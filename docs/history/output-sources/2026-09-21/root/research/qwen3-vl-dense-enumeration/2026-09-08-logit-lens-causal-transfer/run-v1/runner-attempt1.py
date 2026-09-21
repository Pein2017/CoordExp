#!/usr/bin/env python3
"""Bounded Image2299 residual-state causal-transfer probe.

The parent Image2299 Logit Lens helper is imported unchanged and hash-bound.
This runner replays its exact overfit trajectory, captures six decoder block
outputs, and performs the preregistered bidirectional current-token,
full-causal-prefix, and equal-norm random-delta interventions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PARENT_HELPER = ROOT / "scripts/research/probe_image2299_logit_lens.py"
PARENT_HELPER_SHA256 = "e70d33da59b116e0e17194540af3cd6252be62c0c47308c0477fae9685e1f895"
PARENT_OUTPUT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-08-image2299-logit-lens/run-v2"
)
TRAJECTORY = PARENT_OUTPUT / "trajectory-overfit.json"
TRAJECTORY_SHA256 = "537079135d3b12a3bfd72778ea63352112cde60775405e702abbbfa71efce063"
PARENT_RECEIPT = PARENT_OUTPUT / "receipt.json"
PARENT_RECEIPT_SHA256 = "0fc018e26e42d97a08b8a4a6d1e1f56dfd52f279d0637dcbb0c5e49890aa3e58"
DEFAULT_OUTPUT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-08-logit-lens-causal-transfer/run-v1"
)

BLOCKS = (8, 16, 24, 26, 27, 28)  # 1-based decoder block outputs
NONFINAL_BLOCKS = BLOCKS[:-1]
RANDOM_SEEDS = (104729, 104759, 104761, 104773)
EXPECTED_SITE_COUNT = 12
ATOL = 2e-4
RTOL = 2e-4
TOP_K = 5
GPU_BUDGET_SECONDS = 20 * 60
MAX_DEVICE_BYTES = 48 * 1024**3


def _local_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_parent() -> Any:
    if _local_sha256_file(PARENT_HELPER) != PARENT_HELPER_SHA256:
        raise RuntimeError("parent helper hash mismatch before import")
    spec = importlib.util.spec_from_file_location("probe_image2299_logit_lens_frozen", PARENT_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen parent helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parent = _load_parent()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TypeError("decoder output does not support first-tensor replacement")


def _jsonl_append(handle: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _tensor_digest(value: torch.Tensor) -> str:
    return parent.tensor_sha256(value.detach().to(device="cpu").contiguous())


def _nested_tensor_identity(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _tensor_digest(value)}
    if isinstance(value, (tuple, list)):
        return [_nested_tensor_identity(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(type(value))


def coordinate_sites(prompt_ids: Sequence[int], trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = parent.select_sites(
        prompt_token_count=len(prompt_ids), generated_token_ids=trajectory["token_ids"]
    )
    sites = [site for site in selected if any("_coord_" in label for label in site["labels"])]
    require(len(sites) == EXPECTED_SITE_COUNT, f"expected 12 coordinate sites, found {len(sites)}")
    require(len({int(site["position"]) for site in sites}) == len(sites), "coordinate sites are not unique")
    return sites


class LanguageInputCapture:
    """Capture the exact post-vision inputs passed into the text stack once."""

    def __init__(self, text_model: Any) -> None:
        self.text_model = text_model
        self.handle: Any = None
        self.calls = 0
        self.kwargs: dict[str, Any] = {}

    def _hook(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        require(not args, "text model unexpectedly received positional inputs")
        self.calls += 1
        require(self.calls == 1, "text model input capture fired more than once")
        allowed = {
            "attention_mask",
            "position_ids",
            "inputs_embeds",
            "cache_position",
            "visual_pos_masks",
            "deepstack_visual_embeds",
        }
        for key in allowed:
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                self.kwargs[key] = value.detach().clone()
            elif isinstance(value, (tuple, list)):
                self.kwargs[key] = [
                    item.detach().clone() if isinstance(item, torch.Tensor) else item for item in value
                ]
            else:
                self.kwargs[key] = value

    def __enter__(self) -> "LanguageInputCapture":
        self.handle = self.text_model.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        require(self.calls == 1 and isinstance(self.kwargs.get("inputs_embeds"), torch.Tensor), "missing text inputs")


class BlockOutputCapture:
    """Capture complete outputs at only the six frozen blocks."""

    def __init__(self, layers: Sequence[Any]) -> None:
        self.layers = layers
        self.handles: list[Any] = []
        self.states: dict[int, torch.Tensor] = {}
        self.calls: dict[int, int] = {}

    def _hook(self, block: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            hidden = parent._first_tensor(output)
            require(hidden.ndim == 3 and hidden.shape[0] == 1, "decoder hidden shape changed")
            self.calls[block] = self.calls.get(block, 0) + 1
            require(self.calls[block] == 1, f"block {block} capture fired more than once")
            self.states[block] = hidden.detach().to(device="cpu", dtype=torch.float32).contiguous()
            return output

        return hook

    def __enter__(self) -> "BlockOutputCapture":
        for block in BLOCKS:
            self.handles.append(self.layers[block - 1].register_forward_hook(self._hook(block)))
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        require(set(self.states) == set(BLOCKS), f"incomplete block capture: {sorted(self.states)}")


class ResidualPatch:
    """One-shot exact-position decoder-output replacement with corruption checks."""

    def __init__(
        self,
        module: Any,
        *,
        positions: Sequence[int],
        replacement: torch.Tensor,
        expected_before: torch.Tensor,
    ) -> None:
        self.module = module
        self.positions = tuple(int(position) for position in positions)
        self.replacement = replacement.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self.expected_before = expected_before.detach().to(device="cpu", dtype=torch.float32).contiguous()
        require(self.positions and len(set(self.positions)) == len(self.positions), "patch positions must be unique")
        require(self.replacement.ndim == 2, "replacement must have [positions, hidden] shape")
        require(self.replacement.shape == self.expected_before.shape, "replacement/before shape mismatch")
        require(self.replacement.shape[0] == len(self.positions), "patch span length mismatch")
        self.handle: Any = None
        self.calls = 0
        self.before_exact = False
        self.non_target_exact = False
        self.target_max_abs_delta = 0.0

    def _hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = parent._first_tensor(output)
        self.calls += 1
        require(self.calls == 1, "patch hook fired more than once")
        require(hidden.ndim == 3 and hidden.shape[0] == 1, "patch hidden shape changed")
        require(all(0 <= position < hidden.shape[1] for position in self.positions), "patch position out of range")
        positions = torch.tensor(self.positions, dtype=torch.long, device=hidden.device)
        observed = hidden[0, positions, :].detach().to(device="cpu", dtype=torch.float32)
        self.before_exact = bool(torch.equal(observed, self.expected_before))
        require(self.before_exact, "recipient state identity mismatch before patch")
        updated = hidden.clone()
        replacement = self.replacement.to(device=hidden.device, dtype=hidden.dtype)
        updated[0, positions, :] = replacement
        self.target_max_abs_delta = float((updated[0, positions, :] - hidden[0, positions, :]).abs().max().item())
        mask = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        mask[positions] = False
        self.non_target_exact = bool(torch.equal(updated[:, mask, :], hidden[:, mask, :]))
        require(self.non_target_exact, "patch changed a non-target residual position")
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return _replace_first_tensor(output, updated)

    def __enter__(self) -> "ResidualPatch":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        require(self.calls == 1, "patch did not fire exactly once")

    def receipt(self) -> dict[str, Any]:
        return {
            "position_count": len(self.positions),
            "first_position": self.positions[0],
            "last_position": self.positions[-1],
            "hook_calls": self.calls,
            "before_exact": self.before_exact,
            "non_target_residual_exact": self.non_target_exact,
            "target_max_abs_delta": self.target_max_abs_delta,
        }


@dataclass
class ModelBundle:
    name: str
    opened: Any
    model: Any
    text_model: Any
    layers: Sequence[Any]
    head: Any
    language_inputs: dict[str, Any]
    baseline_logits: torch.Tensor
    states: dict[int, torch.Tensor]
    input_receipt: dict[str, Any]
    session_receipt: dict[str, Any]
    baseline_checks: dict[str, Any]


def _text_forward(bundle: ModelBundle, selected: torch.Tensor) -> torch.Tensor:
    kwargs = {key: value for key, value in bundle.language_inputs.items() if value is not None}
    kwargs.update({"use_cache": False, "return_dict": True})
    with torch.inference_mode():
        output = bundle.text_model(**kwargs)
        logits = bundle.head(output.last_hidden_state[:, selected.to(output.last_hidden_state.device), :])
    result = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    require(torch.isfinite(result).all().item(), "nonfinite intervention logits")
    return result


def _open_and_capture(
    *, name: str, adapter: Path, source_gate_root: Path, trajectory: Mapping[str, Any]
) -> ModelBundle:
    opened, frontend, config, resolved = parent._open_session(adapter, source_gate_root=source_gate_root)
    _request, native_inputs, prompt_ids, input_receipt = parent._request_and_inputs(
        opened=opened, frontend=frontend, config=config, request_id=f"logit-lens-causal-transfer-{name}"
    )
    require(parent.sha256_json(list(prompt_ids)) == input_receipt["prompt_token_ids_sha256"], "prompt hash drift")
    sites = coordinate_sites(prompt_ids, trajectory)
    selected = torch.tensor([int(site["position"]) for site in sites], dtype=torch.long)
    model = opened._model  # noqa: SLF001
    layers, _norm, head, seam = parent.resolve_text_stack(model)
    text_model = parent._resolve_path(model, seam["stack_path"])
    require(text_model is not None, "cannot resolve text model owner")
    forwarded = parent._forward_inputs(native_inputs, prompt_ids, trajectory["token_ids"], model)

    # Hooks-off native baseline (the pre-hook only records immutable text-stack inputs).
    with LanguageInputCapture(text_model) as input_capture:
        with torch.inference_mode():
            native = model(**forwarded, logits_to_keep=selected.to(next(model.parameters()).device))
    native_logits = native.logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    require(native_logits.shape[:2] == (1, EXPECTED_SITE_COUNT), "baseline selected-logit shape drift")

    provisional = ModelBundle(
        name=name,
        opened=opened,
        model=model,
        text_model=text_model,
        layers=layers,
        head=head,
        language_inputs=input_capture.kwargs,
        baseline_logits=native_logits,
        states={},
        input_receipt=input_receipt,
        session_receipt=opened.receipt.to_artifact_dict(),
        baseline_checks={},
    )
    with BlockOutputCapture(layers) as capture:
        hooked_logits = _text_forward(provisional, selected)
    maximum = float((hooked_logits - native_logits).abs().max().item())
    require(
        torch.allclose(hooked_logits, native_logits, atol=ATOL, rtol=RTOL),
        f"native/direct hooked baseline mismatch for {name}: max_abs={maximum}",
    )
    provisional.states = capture.states
    provisional.baseline_checks = {
        "native_hooks_off_vs_direct_capture_passed": True,
        "max_absolute_difference": maximum,
        "atol": ATOL,
        "rtol": RTOL,
        "image_encoding_passes": 1,
        "decoder_forward_count": 2,
        "text_input_identity": _nested_tensor_identity(input_capture.kwargs),
        "resolved_stack": seam,
        "resolved_config": resolved.to_artifact_dict(),
    }
    return provisional


def _decode(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def _topk(vector: torch.Tensor, tokenizer: Any) -> list[dict[str, Any]]:
    values, ids = torch.topk(vector, k=TOP_K)
    return [
        {"token_id": int(token), "decoded": _decode(tokenizer, int(token)), "raw_logit": float(value)}
        for token, value in zip(ids.tolist(), values.tolist(), strict=True)
    ]


def _endpoint_summary(vector: torch.Tensor, *, a_id: int, s_id: int, tokenizer: Any) -> dict[str, Any]:
    top_id = int(vector.argmax().item())
    log_z = torch.logsumexp(vector, dim=0)
    return {
        "a_raw_logit": float(vector[a_id].item()),
        "s_raw_logit": float(vector[s_id].item()),
        "margin_a_minus_s": float((vector[a_id] - vector[s_id]).item()),
        "a_probability": float(torch.exp(vector[a_id] - log_z).item()),
        "s_probability": float(torch.exp(vector[s_id] - log_z).item()),
        "top1_token_id": top_id,
        "top1_decoded": _decode(tokenizer, top_id),
        "top_k": _topk(vector, tokenizer),
        "logsumexp": float(log_z.item()),
    }


def _distribution_diagnostics(patch: torch.Tensor, receiver: torch.Tensor, donor: torch.Tensor) -> dict[str, float]:
    return {
        "logit_l2_to_receiver": float((patch - receiver).norm().item()),
        "logit_l2_to_donor": float((patch - donor).norm().item()),
        "logit_max_abs_to_receiver": float((patch - receiver).abs().max().item()),
        "logit_max_abs_to_donor": float((patch - donor).abs().max().item()),
    }


def _random_replacement(
    recipient: torch.Tensor, donor: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    recipient = recipient.detach().to(device="cpu", dtype=torch.float32)
    donor = donor.detach().to(device="cpu", dtype=torch.float32)
    target_delta = donor - recipient
    target_norm = target_delta.norm()
    require(torch.isfinite(target_norm).item(), "nonfinite donor-recipient norm")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    direction = torch.randn(recipient.shape, generator=generator, dtype=torch.float32)
    unit = direction / direction.norm()
    scaled = unit * target_norm
    replacement = recipient + scaled
    actual_norm = (replacement - recipient).norm()
    require(
        torch.allclose(actual_norm, target_norm, atol=1e-5, rtol=1e-5),
        f"random delta norm mismatch seed={seed}",
    )
    return replacement, {
        "seed": int(seed),
        "unit_direction_sha256": _tensor_digest(unit),
        "donor_minus_recipient_l2": float(target_norm.item()),
        "random_delta_l2": float(actual_norm.item()),
        "norm_absolute_difference": float(abs(actual_norm.item() - target_norm.item())),
    }


def _eligible_endpoint(a_id: int, s_id: int) -> tuple[bool, str | None]:
    if not (parent.COORD_START <= a_id < parent.COORD_END):
        return False, "overfit_endpoint_noncoordinate"
    if not (parent.COORD_START <= s_id < parent.COORD_END):
        return False, "source_endpoint_noncoordinate"
    if a_id == s_id:
        return False, "equal_endpoint_tokens"
    return True, None


def build_endpoints(
    *, sites: Sequence[Mapping[str, Any]], source: ModelBundle, overfit: ModelBundle
) -> list[dict[str, Any]]:
    tokenizer = overfit.opened._tokenizer  # noqa: SLF001
    rows: list[dict[str, Any]] = []
    for index, site in enumerate(sites):
        source_vector = source.baseline_logits[0, index]
        overfit_vector = overfit.baseline_logits[0, index]
        s_id = int(source_vector.argmax().item())
        a_id = int(overfit_vector.argmax().item())
        eligible, exclusion = _eligible_endpoint(a_id, s_id)
        rows.append(
            {
                "site_index": index,
                "position": int(site["position"]),
                "labels": list(site["labels"]),
                "trajectory_actual_next_token_id": int(site["actual_next_token_id"]),
                "a_overfit_top1_token_id": a_id,
                "a_decoded": _decode(tokenizer, a_id),
                "s_source_top1_token_id": s_id,
                "s_decoded": _decode(tokenizer, s_id),
                "a_is_coordinate": parent.COORD_START <= a_id < parent.COORD_END,
                "s_is_coordinate": parent.COORD_START <= s_id < parent.COORD_END,
                "equal_endpoint_tokens": a_id == s_id,
                "eligible_for_R": eligible,
                "exclusion_reason": exclusion,
                "source": _endpoint_summary(source_vector, a_id=a_id, s_id=s_id, tokenizer=tokenizer),
                "overfit": _endpoint_summary(overfit_vector, a_id=a_id, s_id=s_id, tokenizer=tokenizer),
            }
        )
    require(len(rows) == EXPECTED_SITE_COUNT, "endpoint population changed")
    return rows


def _row_for_patch(
    *,
    direction: str,
    donor: ModelBundle,
    recipient: ModelBundle,
    endpoint: Mapping[str, Any],
    block: int,
    scope: str,
    patch_vector: torch.Tensor,
    patch_receipt: Mapping[str, Any],
    random_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    index = int(endpoint["site_index"])
    a_id = int(endpoint["a_overfit_top1_token_id"])
    s_id = int(endpoint["s_source_top1_token_id"])
    tokenizer = recipient.opened._tokenizer  # noqa: SLF001
    receiver_vector = recipient.baseline_logits[0, index]
    donor_vector = donor.baseline_logits[0, index]
    summary = _endpoint_summary(patch_vector, a_id=a_id, s_id=s_id, tokenizer=tokenizer)
    m_receiver = float(receiver_vector[a_id] - receiver_vector[s_id])
    m_donor = float(donor_vector[a_id] - donor_vector[s_id])
    m_patch = float(patch_vector[a_id] - patch_vector[s_id])
    denominator = m_donor - m_receiver
    donorward = ((m_patch - m_receiver) * denominator > 0.0) if endpoint["eligible_for_R"] else None
    return {
        "schema_version": "logit_lens_causal_transfer_site.v1",
        "direction": direction,
        "donor": donor.name,
        "recipient": recipient.name,
        "block_1based": int(block),
        "scope": scope,
        "control_seed": None if random_receipt is None else int(random_receipt["seed"]),
        "site_index": index,
        "position": int(endpoint["position"]),
        "labels": list(endpoint["labels"]),
        "a_overfit_top1_token_id": a_id,
        "s_source_top1_token_id": s_id,
        "eligible_for_R": bool(endpoint["eligible_for_R"]),
        "exclusion_reason": endpoint["exclusion_reason"],
        "m_receiver": m_receiver,
        "m_donor": m_donor,
        "m_patch": m_patch,
        "patch_minus_receiver": m_patch - m_receiver,
        "donor_minus_receiver": denominator,
        "donorward": donorward,
        "patch_endpoint": summary,
        "distribution_diagnostics": _distribution_diagnostics(patch_vector, receiver_vector, donor_vector),
        "residual_patch": dict(patch_receipt),
        "random_delta": None if random_receipt is None else dict(random_receipt),
    }


def reduce_trace(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute unclipped grouped R values from raw rows."""

    keys = sorted(
        {
            (str(row["direction"]), int(row["block_1based"]), str(row["scope"]), row["control_seed"])
            for row in rows
        },
        key=lambda key: (key[0], key[1], key[2], -1 if key[3] is None else int(key[3])),
    )
    groups: list[dict[str, Any]] = []
    for direction, block, scope, seed in keys:
        group_rows = [
            row
            for row in rows
            if row["direction"] == direction
            and int(row["block_1based"]) == block
            and row["scope"] == scope
            and row["control_seed"] == seed
        ]
        eligible = [row for row in group_rows if row["eligible_for_R"]]
        numerator = sum(float(row["patch_minus_receiver"]) for row in eligible)
        denominator = sum(float(row["donor_minus_receiver"]) for row in eligible)
        r_value = None if denominator == 0.0 else numerator / denominator
        groups.append(
            {
                "direction": direction,
                "block_1based": block,
                "scope": scope,
                "control_seed": seed,
                "population_site_count": len(group_rows),
                "eligible_site_count": len(eligible),
                "excluded_site_count": len(group_rows) - len(eligible),
                "raw_numerator_sum": numerator,
                "raw_denominator_sum": denominator,
                "R_unclipped": r_value,
                "denominator_small_abs_lt_1e-6": abs(denominator) < 1e-6,
                "donorward_site_count": sum(row["donorward"] is True for row in eligible),
            }
        )

    random_means: list[dict[str, Any]] = []
    full_vs_current: list[dict[str, Any]] = []
    for direction in sorted({str(row["direction"]) for row in rows}):
        for block in sorted({int(row["block_1based"]) for row in rows if row["direction"] == direction}):
            random_groups = [
                group
                for group in groups
                if group["direction"] == direction and group["block_1based"] == block and group["scope"] == "random_delta"
            ]
            if random_groups:
                values = [float(group["R_unclipped"]) for group in random_groups if group["R_unclipped"] is not None]
                random_means.append(
                    {
                        "direction": direction,
                        "block_1based": block,
                        "seed_count": len(random_groups),
                        "R_seed_mean": None if not values else sum(values) / len(values),
                        "R_seed_min": None if not values else min(values),
                        "R_seed_max": None if not values else max(values),
                        "donorward_site_count_mean": sum(group["donorward_site_count"] for group in random_groups)
                        / len(random_groups),
                    }
                )
            current = next(
                (group for group in groups if group["direction"] == direction and group["block_1based"] == block and group["scope"] == "current"),
                None,
            )
            full = next(
                (group for group in groups if group["direction"] == direction and group["block_1based"] == block and group["scope"] == "full_prefix"),
                None,
            )
            if current is not None and full is not None:
                r_difference = None
                if current["R_unclipped"] is not None and full["R_unclipped"] is not None:
                    r_difference = float(full["R_unclipped"]) - float(current["R_unclipped"])
                full_vs_current.append(
                    {
                        "direction": direction,
                        "block_1based": block,
                        "R_full_minus_current": r_difference,
                        "raw_numerator_full_minus_current": float(full["raw_numerator_sum"])
                        - float(current["raw_numerator_sum"]),
                    }
                )
    return {"groups": groups, "random_controls": random_means, "full_vs_current": full_vs_current}


def _run_patch(
    *,
    recipient: ModelBundle,
    block: int,
    positions: Sequence[int],
    replacement: torch.Tensor,
    expected_before: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    patch = ResidualPatch(
        recipient.layers[block - 1],
        positions=positions,
        replacement=replacement,
        expected_before=expected_before,
    )
    with patch:
        logits = _text_forward(recipient, selected)
    return logits, patch.receipt()


def _self_patch_checks(bundle: ModelBundle, selected: torch.Tensor) -> list[dict[str, Any]]:
    sequence_length = int(next(iter(bundle.states.values())).shape[1])
    positions = list(range(sequence_length))
    checks: list[dict[str, Any]] = []
    for block in BLOCKS:
        logits, patch_receipt = _run_patch(
            recipient=bundle,
            block=block,
            positions=positions,
            replacement=bundle.states[block][0],
            expected_before=bundle.states[block][0],
            selected=selected,
        )
        exact = bool(torch.equal(logits, bundle.baseline_logits))
        require(exact, f"self patch changed selected logits for {bundle.name} block {block}")
        checks.append({"checkpoint": bundle.name, "block_1based": block, "selected_logits_exact": exact, **patch_receipt})
    return checks


def _artifact_manifest(output: Path) -> dict[str, Any]:
    result = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"receipt.inprogress.json", "receipt.json"}:
            result[str(path.relative_to(output))] = {"bytes": path.stat().st_size, "sha256": parent.sha256_file(path)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner_sha256_at_launch = parent.sha256_file(Path(__file__))
    started_unix = time.time()
    started_mono = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inprogress: dict[str, Any] = {
        "schema_version": "logit_lens_causal_transfer_receipt.v1",
        "status": "running",
        "stage": "A_image2299_pilot",
        "started_unix": started_unix,
        "output_root": str(output),
        "launch_identity": {
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256_at_launch": runner_sha256_at_launch,
            "parent_helper_path": str(PARENT_HELPER),
            "parent_helper_sha256": PARENT_HELPER_SHA256,
            "trajectory_path": str(TRAJECTORY),
            "trajectory_file_sha256": TRAJECTORY_SHA256,
            "parent_receipt_path": str(PARENT_RECEIPT),
            "parent_receipt_sha256": PARENT_RECEIPT_SHA256,
        },
        "contract": {
            "trajectory": str(TRAJECTORY),
            "site_count": EXPECTED_SITE_COUNT,
            "blocks_1based": list(BLOCKS),
            "nonfinal_blocks_1based": list(NONFINAL_BLOCKS),
            "directions": ["overfit_to_source", "source_to_overfit"],
            "random_seeds": list(RANDOM_SEEDS),
            "atol": ATOL,
            "rtol": RTOL,
            "gpu_budget_seconds": GPU_BUDGET_SECONDS,
            "no_kv_cache": True,
            "full_prefix_equivalence": "one_whole_sequence_post_block_graft_read_at_all_selected_sites_under_causal_mask",
        },
    }
    parent.atomic_json(output / "receipt.inprogress.json", inprogress)
    bundles: list[ModelBundle] = []
    forward_counts = {
        "full_model_with_image_encoding": 0,
        "decoder_baseline_capture": 0,
        "self_patch": 0,
        "current": 0,
        "full_prefix": 0,
        "random_delta": 0,
        "block28_current_identity": 0,
    }
    try:
        require(torch.cuda.is_available(), "CUDA is required")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "GPU-8d43cb78-19ca-2f59-3179-7ea166cb1a4e"}, "not bound to physical GPU0")
        require(parent.sha256_file(PARENT_HELPER) == PARENT_HELPER_SHA256, "parent helper hash mismatch")
        require(parent.sha256_file(TRAJECTORY) == TRAJECTORY_SHA256, "frozen trajectory file hash mismatch")
        require(parent.sha256_file(PARENT_RECEIPT) == PARENT_RECEIPT_SHA256, "parent receipt hash mismatch")
        trajectory = json.loads(TRAJECTORY.read_text())
        require(trajectory["token_count"] == 415 and len(trajectory["token_ids"]) == 415, "trajectory length drift")
        require(trajectory["token_ids_sha256"] == parent.sha256_json(trajectory["token_ids"]), "trajectory token hash drift")

        source_gate_root, source_gate = parent._stage_source_gate(output)
        source = _open_and_capture(
            name="source", adapter=parent.SOURCE_ADAPTER, source_gate_root=source_gate_root, trajectory=trajectory
        )
        bundles.append(source)
        forward_counts["full_model_with_image_encoding"] += 1
        forward_counts["decoder_baseline_capture"] += 1
        require(time.perf_counter() - started_mono < GPU_BUDGET_SECONDS, "GPU budget exhausted after Source capture")
        overfit = _open_and_capture(
            name="overfit", adapter=parent.OVERFIT_ADAPTER, source_gate_root=source_gate_root, trajectory=trajectory
        )
        bundles.append(overfit)
        forward_counts["full_model_with_image_encoding"] += 1
        forward_counts["decoder_baseline_capture"] += 1
        require(source.input_receipt == overfit.input_receipt, "prompt/image identity differs across checkpoints")
        resident_after_two = int(torch.cuda.memory_allocated())
        require(resident_after_two < MAX_DEVICE_BYTES, f"two-model resident allocation exceeds 48 GiB: {resident_after_two}")

        prompt_count = int(source.input_receipt["prompt_token_count"])
        sites = coordinate_sites([0] * prompt_count, trajectory)
        selected = torch.tensor([int(site["position"]) for site in sites], dtype=torch.long)
        endpoints = build_endpoints(sites=sites, source=source, overfit=overfit)
        baseline = {
            "schema_version": "logit_lens_causal_transfer_baseline.v1",
            "trajectory_path": str(TRAJECTORY),
            "trajectory_file_sha256": TRAJECTORY_SHA256,
            "trajectory_token_ids_sha256": trajectory["token_ids_sha256"],
            "input": source.input_receipt,
            "sites": endpoints,
            "checkpoint_checks": {"source": source.baseline_checks, "overfit": overfit.baseline_checks},
        }
        parent.atomic_json(output / "baseline.json", baseline)
        compact = {
            "schema_version": "logit_lens_causal_transfer_compact_tensors.v1",
            "blocks_1based": list(BLOCKS),
            "sites": [{"site_index": i, **dict(site)} for i, site in enumerate(sites)],
            "baseline_logits": {
                "source": source.baseline_logits,
                "overfit": overfit.baseline_logits,
            },
            "selected_residuals": {
                checkpoint.name: torch.stack(
                    [checkpoint.states[block][0, selected, :] for block in BLOCKS]
                )
                for checkpoint in (source, overfit)
            },
        }
        torch.save(compact, output / "compact-selected.pt")
        inprogress["status"] = "baseline_complete"
        inprogress["counts"] = {**forward_counts, "trace_rows": 0}
        parent.atomic_json(output / "receipt.inprogress.json", inprogress)

        self_checks = [*_self_patch_checks(source, selected), *_self_patch_checks(overfit, selected)]
        forward_counts["self_patch"] += len(self_checks)
        trace_rows: list[dict[str, Any]] = []
        with (output / "patch-traces.jsonl").open("x") as trace_handle:
            for direction, donor, recipient in (
                ("overfit_to_source", overfit, source),
                ("source_to_overfit", source, overfit),
            ):
                for block in BLOCKS:
                    require(time.perf_counter() - started_mono < GPU_BUDGET_SECONDS, "GPU budget exhausted before patch group")
                    # Current-state donor transplant: one selected site per decoder forward.
                    for endpoint in endpoints:
                        index = int(endpoint["site_index"])
                        position = int(endpoint["position"])
                        logits, patch_receipt = _run_patch(
                            recipient=recipient,
                            block=block,
                            positions=[position],
                            replacement=donor.states[block][0, position, :].unsqueeze(0),
                            expected_before=recipient.states[block][0, position, :].unsqueeze(0),
                            selected=selected,
                        )
                        forward_counts["block28_current_identity" if block == 28 else "current"] += 1
                        earlier = selected < position
                        earlier_exact = bool(torch.equal(logits[:, earlier, :], recipient.baseline_logits[:, earlier, :]))
                        require(earlier_exact, f"future-position patch contaminated earlier logits block={block} site={index}")
                        if block == 28:
                            other = torch.ones(EXPECTED_SITE_COUNT, dtype=torch.bool)
                            other[index] = False
                            unpatched_exact = bool(torch.equal(logits[:, other, :], recipient.baseline_logits[:, other, :]))
                            require(unpatched_exact, f"block28 changed an unpatched selected logit site={index}")
                            parity = bool(
                                torch.allclose(
                                    logits[0, index], donor.baseline_logits[0, index], atol=ATOL, rtol=RTOL
                                )
                            )
                            parity_max = float((logits[0, index] - donor.baseline_logits[0, index]).abs().max().item())
                            require(parity, f"block28 donor full-vocab parity failed site={index}: {parity_max}")
                            patch_receipt = {
                                **patch_receipt,
                                "all_unpatched_selected_logits_exact": unpatched_exact,
                                "donor_full_vocab_parity_passed": parity,
                                "donor_full_vocab_max_abs": parity_max,
                                "atol": ATOL,
                                "rtol": RTOL,
                            }
                        else:
                            patch_receipt = {
                                **patch_receipt,
                                "all_earlier_selected_logits_exact": earlier_exact,
                                "later_selected_logits_are_not_used_by_this_site_isolated_arm": True,
                            }
                        row = _row_for_patch(
                            direction=direction,
                            donor=donor,
                            recipient=recipient,
                            endpoint=endpoint,
                            block=block,
                            scope="current",
                            patch_vector=logits[0, index],
                            patch_receipt=patch_receipt,
                            random_receipt=None,
                        )
                        trace_rows.append(row)
                        _jsonl_append(trace_handle, [row])

                    if block == 28:
                        continue

                    # Full causal-prefix splice: one whole-sequence graft is
                    # causally equivalent to per-site <=t grafts for selected
                    # readouts because later positions cannot influence earlier logits.
                    sequence_length = recipient.states[block].shape[1]
                    logits, patch_receipt = _run_patch(
                        recipient=recipient,
                        block=block,
                        positions=list(range(sequence_length)),
                        replacement=donor.states[block][0],
                        expected_before=recipient.states[block][0],
                        selected=selected,
                    )
                    forward_counts["full_prefix"] += 1
                    patch_receipt = {
                        **patch_receipt,
                        "scope_semantics": "all_visual_prompt_generated_history_and_current_positions",
                        "causal_equivalence": "whole_sequence_once_read_all_sites_equals_each_prefix_through_t",
                    }
                    rows = [
                        _row_for_patch(
                            direction=direction,
                            donor=donor,
                            recipient=recipient,
                            endpoint=endpoint,
                            block=block,
                            scope="full_prefix",
                            patch_vector=logits[0, int(endpoint["site_index"])],
                            patch_receipt=patch_receipt,
                            random_receipt=None,
                        )
                        for endpoint in endpoints
                    ]
                    trace_rows.extend(rows)
                    _jsonl_append(trace_handle, rows)

                    for endpoint in endpoints:
                        index = int(endpoint["site_index"])
                        position = int(endpoint["position"])
                        recipient_state = recipient.states[block][0, position, :]
                        donor_state = donor.states[block][0, position, :]
                        for seed in RANDOM_SEEDS:
                            replacement, random_receipt = _random_replacement(
                                recipient_state, donor_state, seed=seed
                            )
                            logits, patch_receipt = _run_patch(
                                recipient=recipient,
                                block=block,
                                positions=[position],
                                replacement=replacement.unsqueeze(0),
                                expected_before=recipient_state.unsqueeze(0),
                                selected=selected,
                            )
                            forward_counts["random_delta"] += 1
                            earlier = selected < position
                            earlier_exact = bool(torch.equal(logits[:, earlier, :], recipient.baseline_logits[:, earlier, :]))
                            require(earlier_exact, "random future-position patch contaminated earlier logits")
                            patch_receipt = {
                                **patch_receipt,
                                "all_earlier_selected_logits_exact": earlier_exact,
                                "later_selected_logits_are_not_used_by_this_site_isolated_arm": True,
                            }
                            row = _row_for_patch(
                                direction=direction,
                                donor=donor,
                                recipient=recipient,
                                endpoint=endpoint,
                                block=block,
                                scope="random_delta",
                                patch_vector=logits[0, index],
                                patch_receipt=patch_receipt,
                                random_receipt=random_receipt,
                            )
                            trace_rows.append(row)
                            _jsonl_append(trace_handle, [row])

                    inprogress["status"] = f"patch_complete_{direction}_block_{block}"
                    inprogress["counts"] = {**forward_counts, "trace_rows": len(trace_rows)}
                    parent.atomic_json(output / "receipt.inprogress.json", inprogress)

        elapsed = time.perf_counter() - started_mono
        require(elapsed <= GPU_BUDGET_SECONDS, f"Stage A exceeded GPU budget: {elapsed:.1f}s")
        summary = {
            "schema_version": "logit_lens_causal_transfer_summary.v1",
            "population_site_count": len(endpoints),
            "eligible_site_count": sum(bool(endpoint["eligible_for_R"]) for endpoint in endpoints),
            "excluded_sites": [
                {"site_index": endpoint["site_index"], "reason": endpoint["exclusion_reason"]}
                for endpoint in endpoints
                if not endpoint["eligible_for_R"]
            ],
            **reduce_trace(trace_rows),
        }
        parent.atomic_json(output / "summary.json", summary)

        # Cold readback and exact row/accounting validation before receipt.
        cold_rows = [json.loads(line) for line in (output / "patch-traces.jsonl").read_text().splitlines()]
        require(cold_rows == trace_rows, "cold trace readback differs from in-memory trace")
        cold_compact = torch.load(output / "compact-selected.pt", map_location="cpu", weights_only=False)
        require(cold_compact["schema_version"] == compact["schema_version"], "compact tensor cold readback failed")
        expected_trace_rows = 2 * (
            len(BLOCKS) * EXPECTED_SITE_COUNT
            + len(NONFINAL_BLOCKS) * EXPECTED_SITE_COUNT
            + len(NONFINAL_BLOCKS) * EXPECTED_SITE_COUNT * len(RANDOM_SEEDS)
        )
        require(len(trace_rows) == expected_trace_rows, f"trace row count mismatch: {len(trace_rows)} != {expected_trace_rows}")
        artifacts = _artifact_manifest(output)
        receipt = {
            **inprogress,
            "status": "mechanics_candidate",
            "completed_unix": time.time(),
            "elapsed_seconds": elapsed,
            "identity": {
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256_at_launch": runner_sha256_at_launch,
                "runner_sha256_at_completion": parent.sha256_file(Path(__file__)),
                "parent_helper_path": str(PARENT_HELPER),
                "parent_helper_sha256": PARENT_HELPER_SHA256,
                "parent_receipt_path": str(PARENT_RECEIPT),
                "parent_receipt_sha256": PARENT_RECEIPT_SHA256,
                "trajectory_path": str(TRAJECTORY),
                "trajectory_file_sha256": TRAJECTORY_SHA256,
                "trajectory_token_ids_sha256": trajectory["token_ids_sha256"],
                "source_gate": source_gate,
                "source_adapter": {"path": str(parent.SOURCE_ADAPTER), "sha256": parent.SOURCE_ADAPTER_SHA256},
                "overfit_adapter": {"path": str(parent.OVERFIT_ADAPTER), "sha256": parent.OVERFIT_ADAPTER_SHA256},
                "embedding_delta": {"path": str(parent.SOURCE_DELTA), "sha256": parent.SOURCE_DELTA_SHA256},
            },
            "runtime": parent._runtime_identity(),
            "sessions": {"source": source.session_receipt, "overfit": overfit.session_receipt},
            "input": source.input_receipt,
            "counts": {
                **forward_counts,
                "decoder_forward_total_including_full_model": sum(forward_counts.values()),
                "image_encoding_passes": forward_counts["full_model_with_image_encoding"],
                "trace_rows": len(trace_rows),
                "population_sites": len(endpoints),
                "eligible_sites": summary["eligible_site_count"],
                "self_patch_checks": len(self_checks),
                "resident_models": 2,
            },
            "checks": {
                "parent_hashes_passed": True,
                "same_prompt_and_image_across_models": True,
                "all_native_vs_capture_parity": all(bundle.baseline_checks["native_hooks_off_vs_direct_capture_passed"] for bundle in bundles),
                "all_self_patch_exact": all(check["selected_logits_exact"] for check in self_checks),
                "all_patch_before_identity_exact": all(row["residual_patch"]["before_exact"] for row in trace_rows),
                "all_patch_non_target_residual_exact": all(row["residual_patch"]["non_target_residual_exact"] for row in trace_rows),
                "all_current_earlier_logits_exact": all(
                    row["residual_patch"].get("all_earlier_selected_logits_exact", True)
                    for row in trace_rows
                    if row["scope"] in {"current", "random_delta"} and row["block_1based"] != 28
                ),
                "all_block28_donor_full_vocab_parity": all(
                    row["residual_patch"].get("donor_full_vocab_parity_passed", False)
                    for row in trace_rows
                    if row["block_1based"] == 28
                ),
                "all_block28_unpatched_selected_exact": all(
                    row["residual_patch"].get("all_unpatched_selected_logits_exact", False)
                    for row in trace_rows
                    if row["block_1based"] == 28
                ),
                "all_random_norms_matched": all(
                    row["random_delta"] is None or row["random_delta"]["norm_absolute_difference"] <= 1e-5
                    for row in trace_rows
                ),
                "cold_readback_passed": True,
            },
            "self_patch_checks": self_checks,
            "resource": {
                "two_model_resident_allocated_bytes": resident_after_two,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "artifact_payload_bytes_excluding_receipt": sum(item["bytes"] for item in artifacts.values()),
            },
            "artifacts": artifacts,
            "claim_boundary": {
                "one_training_image": True,
                "teacher_forced_exact_overfit_prefix": True,
                "coordinate_endpoint_preference_not_accuracy": True,
                "no_owner_coverage_or_natural_generation_claim": True,
                "block28_is_algebraic_positive_control": True,
                "full_prefix_splice_may_be_out_of_distribution": True,
            },
            "stop": "stage_A_candidate_complete_stop_GPU_for_lead_decision",
        }
        require(
            receipt["identity"]["runner_sha256_at_completion"] == runner_sha256_at_launch,
            "runner changed during execution",
        )
        require(all(receipt["checks"].values()), "terminal mechanics checks failed")
        parent.atomic_json(output / "receipt.json", receipt)
        (output / "receipt.inprogress.json").unlink()
        print(json.dumps({"status": receipt["status"], "output_root": str(output), "elapsed_seconds": elapsed}, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {
            **inprogress,
            "status": "failed",
            "failed_unix": time.time(),
            "elapsed_seconds": time.perf_counter() - started_mono,
            "counts": {**forward_counts, "trace_rows": inprogress.get("counts", {}).get("trace_rows", 0)},
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
            "resource": {
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
        }
        parent.atomic_json(output / "receipt.failed.json", failure)
        raise
    finally:
        for bundle in reversed(bundles):
            try:
                bundle.opened.close()
            except Exception:
                pass
        bundles.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
