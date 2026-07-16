#!/usr/bin/env python3
"""Bounded downstream layer-output residual-state portability probe.

This module deliberately keeps the experiment local.  It reuses the accepted
fixed-encoding feature replay and row scorer, while adding one causal seam:
capture a returned language-decoder block output under a hard donor mask and
replace one recipient boundary position during a fresh unrestricted prefill.
The helper functions are intentionally small so they can be exercised with
fake modules before a GPU smoke is attempted.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.research.run_fixed_encoding_query_scoped_object_centered_spatial_eligibility as query  # noqa: E402
from src.analysis.visual_support_counterfactual import FeatureBundle, FeatureReplayController  # noqa: E402


UNIT_ID = "2026-07-15-fixed-encoding-downstream-residual-state-portability-gate"
ALL_QUERY_PARENT_UNIT_ID = "2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover"
ROW_QUERY_PARENT_UNIT_ID = "2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover"
ALL_QUERY_PARENT_SHA256 = "6148fa75998385eeab0ac0118d3fe540acaa6c7dfece85ee6116907ed3a703ee"
ROW_QUERY_PARENT_SHA256 = "4aba45f16a90d032191a51f1c9c35ad9b71ae4d0ec97f943d9e0ddf217055fa4"
DEFAULT_ALL_QUERY_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/"
    "cohort-six-float32-20260715b/receipt.json"
)
DEFAULT_ROW_QUERY_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/"
    "cohort-four-float32-20260715a/receipt.json"
)
DEFAULT_CONFIG = Path(
    "/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml"
)
DEFAULT_SOURCE_JSONL = Path(
    "/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/"
    "val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl"
)
DEFAULT_LEDGER = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/readiness-v2/"
    "audit-augmented-ledger.jsonl"
)
IMAGE139 = {
    "image_id": "139",
    "target_annotation_id": "1669970",
    "competitor_annotation_id": "1666628",
    "target_category": "vase",
    "competitor_category": "clock",
}
FROZEN_ANCHORS: dict[str, dict[str, str]] = {
    "139": IMAGE139,
    "632": {"image_id": "632", "target_annotation_id": "1661908", "competitor_annotation_id": "1989419"},
    "12639": {"image_id": "12639", "target_annotation_id": "543629", "competitor_annotation_id": "1215138"},
}


def build_request_identity() -> dict[str, Any]:
    """Return explicit image and object provenance for the frozen smoke request."""

    return {
        "image_id": IMAGE139["image_id"],
        "recipient": {
            "object_name": IMAGE139["target_category"],
            "annotation_id": IMAGE139["target_annotation_id"],
        },
        "donors": [
            {
                "donor_name": "vase",
                "object_name": IMAGE139["target_category"],
                "annotation_id": IMAGE139["target_annotation_id"],
            },
            {
                "donor_name": "clock",
                "object_name": IMAGE139["competitor_category"],
                "annotation_id": IMAGE139["competitor_annotation_id"],
            },
        ],
    }


PARENT_ARM_MAPPING = {
    "139": {"target": ("all_query", "target_eligibility"), "competitor": ("all_query", "competitor_eligibility")},
    "632": {"target": ("row_query", "target_row_query_only_hard"), "competitor": ("row_query", "competitor_row_query_only_hard")},
    "12639": {"target": ("row_query", "target_row_query_only_hard"), "competitor": ("row_query", "competitor_row_query_only_hard")},
}
OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
COORDINATE_TOKEN_START = 151670
TOLERANCE = 1e-4
ELIGIBILITY_RELEASE_FLOOR = -0.05
PORTABILITY_RELEASE_FLOOR = 0.10
RETURNED_LAYER_OUTPUT_SEAM = "returned_layer_output_after_full_block"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("decoder block output does not expose a tensor as its first value")


def _replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TypeError("decoder block output does not support first-tensor replacement")


def resolve_decoder_layer(model: Any, layer_idx: int) -> tuple[Any, dict[str, Any]]:
    """Resolve exactly one zero-based ``Qwen3VLTextDecoderLayer``."""

    wanted = int(layer_idx)
    if wanted < 0:
        raise ValueError("layer_idx must be non-negative")
    candidates: list[tuple[str, Any]] = []
    # PEFT/wrapper layers can expose aliases such as ``language_model`` and
    # ``model.language_model`` for the same live ModuleList.  Prefer the
    # canonical module path used by the checkpoint, while deduplicating by
    # object identity.  Distinct objects at one index remain a hard failure.
    for root_name in ("model.language_model", "model.model.language_model", "language_model"):
        owner = model
        try:
            for part in root_name.split("."):
                owner = getattr(owner, part)
            layers = getattr(owner, "layers")
        except (AttributeError, TypeError):
            continue
        if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
            continue
        if wanted < len(layers):
            candidates.append((f"{root_name}.layers[{wanted}]", layers[wanted]))
    # Some fake models expose only an explicit layers list; allow it only if
    # there is exactly one unambiguous path.
    if not candidates and hasattr(model, "layers"):
        layers = getattr(model, "layers")
        if wanted < len(layers):
            candidates.append((f"layers[{wanted}]", layers[wanted]))
    by_identity: dict[int, list[tuple[str, Any]]] = {}
    for name, module in candidates:
        by_identity.setdefault(id(module), []).append((name, module))
    if len(by_identity) != 1:
        raise ValueError(
            f"expected one distinct decoder layer for zero-based index {wanted}, "
            f"found {len(by_identity)} candidates: {[name for name, _ in candidates]}"
        )
    aliases = next(iter(by_identity.values()))
    preferred_order = {"model.language_model": 0, "model.model.language_model": 1, "language_model": 2}
    aliases.sort(key=lambda item: (preferred_order.get(item[0].rsplit(".layers", 1)[0], 99), item[0]))
    name, module = aliases[0]
    class_name = module.__class__.__name__
    if "Qwen3VLTextDecoderLayer" not in class_name and not getattr(module, "_coordexp_decoder_layer", False):
        # Fake modules can opt in with the marker; production modules must be
        # the named Qwen decoder layer to avoid patching the wrong stack.
        raise TypeError(f"resolved module {name} is not Qwen3VLTextDecoderLayer: {class_name}")
    return module, {
        "layer_idx": wanted,
        "module_path": name,
        "module_alias_paths": [alias_name for alias_name, _ in aliases],
        "module_alias_count": len(aliases),
        "module_class": class_name,
    }


def build_decoder_layer_resolution_receipt(resolution: Mapping[str, Any]) -> dict[str, Any]:
    """Attach immutable seam semantics to a resolved decoder-layer receipt."""

    required = {
        "layer_idx",
        "module_path",
        "module_alias_paths",
        "module_alias_count",
        "module_class",
    }
    missing = sorted(key for key in required if key not in resolution)
    if missing:
        raise ValueError(f"decoder-layer resolution lacks required fields: {missing}")
    aliases = [str(value) for value in resolution["module_alias_paths"]]
    if int(resolution["module_alias_count"]) != len(aliases):
        raise ValueError("decoder-layer alias count does not match alias paths")
    return {
        "layer_idx": int(resolution["layer_idx"]),
        "module_path": str(resolution["module_path"]),
        "module_alias_paths": aliases,
        "module_alias_count": len(aliases),
        "module_class": str(resolution["module_class"]),
        "replacement_seam": RETURNED_LAYER_OUTPUT_SEAM,
        "seam_semantics": {
            "hook_kind": "forward_hook",
            "capture_tensor": "first_tensor_returned_by_decoder_block",
            "replacement_tensor": "first_tensor_returned_by_decoder_block",
            "timing": "after_full_decoder_block_forward",
            "position": "batch_index_0_boundary_position",
            "scope": "one_boundary_position_only",
        },
    }


class ResidualStateCapture:
    """One-forward detached capture of a returned block output position."""

    def __init__(self, module: Any, *, boundary_pos: int) -> None:
        self.module = module
        self.boundary_pos = int(boundary_pos)
        self.handle: Any = None
        self.capture_count = 0
        self.state: torch.Tensor | None = None
        self.input_shape: list[int] | None = None

    def _hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        tensor = _first_tensor(output)
        pos = self.boundary_pos
        if tensor.ndim < 3 or not 0 <= pos < tensor.shape[1]:
            raise ValueError("boundary position is outside returned block output")
        self.capture_count += 1
        if self.capture_count != 1:
            raise RuntimeError("donor capture hook fired more than once")
        self.state = tensor[0, pos, :].detach().clone()
        self.input_shape = list(tensor.shape)
        return output

    def install(self) -> None:
        if self.handle is not None:
            raise RuntimeError("capture hook already installed")
        self.handle = self.module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self) -> "ResidualStateCapture":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()
        if self.capture_count != 1 or self.state is None:
            raise RuntimeError("donor capture did not fire exactly once")


class ResidualStateReplacement:
    """One-position recipient replacement that removes itself immediately."""

    def __init__(self, module: Any, *, boundary_pos: int, replacement: torch.Tensor) -> None:
        self.module = module
        self.boundary_pos = int(boundary_pos)
        self.replacement = replacement.detach().clone()
        self.handle: Any = None
        self.replacement_count = 0
        self.forward_count = 0
        self.other_position_max_abs_delta = 0.0
        self.before_state: torch.Tensor | None = None
        self.hook_removed_inside_hook = False

    def _hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        tensor = _first_tensor(output)
        if tensor.ndim < 3 or not 0 <= self.boundary_pos < tensor.shape[1]:
            raise ValueError("boundary position is outside returned block output")
        self.forward_count += 1
        if self.forward_count != 1:
            raise RuntimeError("recipient replacement hook fired more than once")
        self.before_state = tensor.detach().clone()
        updated = tensor.clone()
        replacement = self.replacement.to(device=tensor.device, dtype=tensor.dtype)
        updated[0, self.boundary_pos, :] = replacement
        changed = (updated - tensor).abs()
        off_position = torch.ones_like(changed, dtype=torch.bool)
        off_position[0, self.boundary_pos, :] = False
        self.other_position_max_abs_delta = float(changed[off_position].max().item()) if bool(off_position.any()) else 0.0
        self.replacement_count += 1
        result = _replace_first_tensor(output, updated)
        # Removal from within the prefill hook is part of the contract: later
        # cached token calls must not see the donor state again.
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.hook_removed_inside_hook = True
        return result

    def install(self) -> None:
        if self.handle is not None:
            raise RuntimeError("replacement hook already installed")
        self.handle = self.module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self) -> "ResidualStateReplacement":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()
        if self.forward_count != 1 or self.replacement_count != 1:
            raise RuntimeError("recipient replacement did not fire exactly once")
        if self.other_position_max_abs_delta != 0.0:
            raise RuntimeError("residual replacement changed a non-boundary position")


def derive_boundary_position(input_ids: torch.Tensor | Sequence[int], *, boundary_token_id: int, role: str) -> int:
    """Return the active boundary position at the end of a constructed prefix.

    Historical rows may contain the same wrapper token many times.  The active
    boundary is therefore identified by the final token of the constructed
    prefix, not by requiring global uniqueness.
    """

    if isinstance(input_ids, torch.Tensor):
        values = input_ids.detach().reshape(-1).cpu().tolist()
    else:
        values = [int(v) for v in input_ids]
    if role not in {"first_description", "pre_x1"}:
        raise ValueError(f"unsupported boundary role {role!r}")
    if not values:
        raise ValueError(f"{role} boundary prefix is empty; expected terminal token {boundary_token_id}")
    if int(values[-1]) != int(boundary_token_id):
        raise ValueError(
            f"{role} boundary token must be the final token {boundary_token_id}, "
            f"observed {int(values[-1])}"
        )
    return int(len(values) - 1)


def derive_boundary_positions(
    first_description_input_ids: torch.Tensor | Sequence[int],
    *, pre_x1_input_ids: torch.Tensor | Sequence[int],
) -> dict[str, int]:
    """Resolve absolute positions from two independently constructed prefixes."""

    return {
        "first_description": derive_boundary_position(
            first_description_input_ids,
            boundary_token_id=OBJECT_REF_START,
            role="first_description",
        ),
        "pre_x1": derive_boundary_position(
            pre_x1_input_ids,
            boundary_token_id=BOX_START,
            role="pre_x1",
        ),
    }


def max_abs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors have different lengths")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)), default=0.0)


def assess_parent_reproduction(*, live: Mapping[str, Any], frozen: Mapping[str, Any], tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Compare mapped parent owner paths before interpreting replacement effects."""

    owners: dict[str, Any] = {}
    for owner in ("target", "competitor"):
        live_values = live.get(owner, {})
        frozen_values = frozen.get(owner, {})
        live_tokens = list(live_values.get("token_log_probabilities", []))
        frozen_tokens = list(frozen_values.get("token_log_probabilities", []))
        live_ranks = [int(v) for v in live_values.get("selected_token_ranks", [])]
        frozen_ranks = [int(v) for v in frozen_values.get("selected_token_ranks", [])]
        live_top_ids = [int(v) for v in live_values.get("top_prediction_token_ids", [])]
        frozen_top_ids = [int(v) for v in frozen_values.get("top_prediction_token_ids", [])]
        if len(live_tokens) != len(frozen_tokens):
            owners[owner] = {"passed": False, "reason": "token_vector_length_mismatch", "live_count": len(live_tokens), "frozen_count": len(frozen_tokens)}
            continue
        drift = max_abs_delta(live_tokens, frozen_tokens)
        ranks_equal = live_ranks == frozen_ranks
        top_ids_equal = live_top_ids == frozen_top_ids
        owners[owner] = {
            "passed": bool(drift <= float(tolerance) and ranks_equal and top_ids_equal),
            "max_abs_logprob_drift": drift,
            "selected_token_ranks_equal": ranks_equal,
            "top_prediction_token_ids_equal": top_ids_equal,
            "tolerance": float(tolerance),
        }
    return {"passed": bool(owners) and all(bool(item.get("passed")) for item in owners.values()), "owners": owners, "tolerance": float(tolerance)}


def realized_path_release(persistent_mean: float, unrestricted_mean: float) -> float:
    return float(persistent_mean) - float(unrestricted_mean)


def assess_donor_eligibility(
    *,
    persistent_release: float | None,
    valid_path: bool,
    owner_match: bool,
    minimum_release: float = ELIGIBILITY_RELEASE_FLOOR,
) -> dict[str, Any]:
    """Eligibility is strict and never inferred from canonical pairwise margins."""

    release = None if persistent_release is None else float(persistent_release)
    release_available = release is not None and math.isfinite(release)
    return {
        "valid_path": bool(valid_path),
        "owner_match": bool(owner_match),
        "persistent_release": release,
        "minimum_release": float(minimum_release),
        "release_available": release_available,
        "passed": bool(valid_path and owner_match and release_available and release >= float(minimum_release)),
        "classification": "eligible_donor" if bool(valid_path and owner_match and release_available and release >= float(minimum_release)) else "ineligible_control",
    }


def half_positive_persistent_release(persistent_release: float) -> float:
    """Return exactly ``0.5 * max(persistent_release, 0)``."""

    return 0.5 * max(float(persistent_release), 0.0)


def assess_portability_release(
    *, persistent_release: float, replacement_release: float, no_op_drift: float,
    valid_continuation: bool, owner_path_match: bool, path_phase: str = "description",
    minimum_release: float = PORTABILITY_RELEASE_FLOOR,
) -> dict[str, Any]:
    half_release = half_positive_persistent_release(persistent_release)
    effect_floor = 10.0 * float(no_op_drift)
    passed = bool(
        valid_continuation
        and owner_path_match
        and float(replacement_release) >= float(minimum_release)
        and float(replacement_release) >= half_release
        and float(replacement_release) >= effect_floor
    )
    return {
        "persistent_release": float(persistent_release),
        "replacement_release": float(replacement_release),
        "half_positive_persistent_release": half_release,
        "effect_floor_10x_no_op_drift": effect_floor,
        "valid_continuation": bool(valid_continuation),
        "owner_path_match": bool(owner_path_match),
        "path_phase": str(path_phase),
        "minimum_release": float(minimum_release),
        "passed": passed,
    }


def validate_parent_receipt(path: Path, *, expected_sha256: str, expected_unit_id: str) -> tuple[dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve(strict=True)
    observed = sha256_file(resolved)
    if observed != str(expected_sha256):
        raise ValueError(f"parent receipt SHA-256 mismatch: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("unit_id") != expected_unit_id:
        raise ValueError(f"parent receipt unit mismatch: expected {expected_unit_id!r}")
    return payload, observed


def validate_parent_arm_mapping(*, image_id: str, all_query: Mapping[str, Any], row_query: Mapping[str, Any]) -> dict[str, Any]:
    image = str(image_id)
    if image not in PARENT_ARM_MAPPING:
        raise ValueError(f"unsupported frozen image {image}")
    mapping = PARENT_ARM_MAPPING[image]
    observed: dict[str, Any] = {}
    for owner in ("target", "competitor"):
        root_name, arm_name = mapping[owner]
        result = next((item for item in all_query.get("results", []) if str(item.get("image_id")) == image), None) if root_name == "all_query" else next((item for item in row_query.get("results", []) if str(item.get("image_id")) == image), None)
        if not isinstance(result, Mapping):
            raise ValueError(f"missing {root_name} parent result for image {image}")
        arms = result.get("arms", {})
        if arm_name not in arms:
            raise ValueError(f"missing frozen arm {arm_name} for image {image}")
        arm = arms[arm_name]
        if not isinstance(arm, Mapping) or not isinstance(arm.get(owner), Mapping):
            raise ValueError(
                f"frozen arm {arm_name} for image {image} lacks nested {owner} score"
            )
        observed[owner] = {"receipt": root_name, "arm_name": arm_name, "arm": arm}
    expected = FROZEN_ANCHORS[image]
    result_ids = next((item for item in all_query.get("results", []) if str(item.get("image_id")) == image), None)
    row_result = next((item for item in row_query.get("results", []) if str(item.get("image_id")) == image), None)
    for result in (result_ids, row_result):
        if isinstance(result, Mapping):
            for key in ("target_annotation_id", "competitor_annotation_id"):
                expected_value = expected.get(key)
                if expected_value is not None and str(result.get(key)) != str(expected_value):
                    raise ValueError(f"parent {key} mismatch for image {image}")
    return {"image_id": image, "mapping": {owner: {"receipt": item["receipt"], "arm_name": item["arm_name"]} for owner, item in observed.items()}, "arms": {owner: item["arm"] for owner, item in observed.items()}}


def classify_portability_case(*, trust_gate_passed: bool, semantic: Mapping[str, Any], geometry: Mapping[str, Any], eligible_count: int) -> dict[str, Any]:
    if not trust_gate_passed:
        return {"classification": "invalid_execution_trust_gate", "interpreted": False}
    semantic_passed = bool(semantic.get("passed"))
    geometry_passed = bool(geometry.get("passed"))
    if semantic_passed and geometry_passed:
        label = "promote_phase_specific_semantic_and_geometry_portability"
    elif semantic_passed:
        label = "promote_bounded_one_sided_semantic_portability"
    elif geometry_passed:
        label = "promote_bounded_one_sided_geometry_portability"
    elif int(eligible_count) > 0:
        label = "close_one_site_conditional_downstream_portability"
    else:
        label = "no_eligible_donor_control_only"
    return {"classification": label, "interpreted": True, "semantic_passed": semantic_passed, "geometry_passed": geometry_passed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--audit-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--all-query-receipt", type=Path, default=DEFAULT_ALL_QUERY_RECEIPT)
    parser.add_argument("--row-query-receipt", type=Path, default=DEFAULT_ROW_QUERY_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-ids", nargs="+", default=["139"])
    parser.add_argument("--layers", nargs="+", type=int, default=[23, 13])
    parser.add_argument("--roles", nargs="+", choices=("first_description", "pre_x1"), default=["first_description"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser


def _score_with_hooks(
    *, model: Any, model_inputs: Mapping[str, Any], features: FeatureBundle, grid_thw: Sequence[int], merge_size: int,
    ids: torch.Tensor, image_grid_thw: torch.Tensor, position_ids: torch.Tensor, custom_mask: torch.Tensor | None,
    layer_idx: int, boundary_pos: int, capture: ResidualStateCapture | None = None,
    replacement: ResidualStateReplacement | None = None,
    use_cache: bool = False, return_output: bool = False, resolved_module: Any | None = None,
) -> Any:
    device = next(model.parameters()).device
    kwargs = {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in model_inputs.items()
        if key not in {"input_ids", "attention_mask", "position_ids"}
    }
    kwargs.update({"input_ids": ids, "attention_mask": custom_mask if custom_mask is not None else torch.ones_like(ids), "position_ids": position_ids, "use_cache": bool(use_cache), "logits_to_keep": 0, "return_dict": True})
    if use_cache:
        kwargs["cache_position"] = torch.arange(ids.shape[1], dtype=torch.long, device=device)
    live_module, _ = resolve_decoder_layer(model, layer_idx)
    if resolved_module is not None and live_module is not resolved_module:
        raise RuntimeError("resolved decoder module changed during seam execution")
    replay = FeatureReplayController(model=model, recipient=features, donor=features, grid_thw=grid_thw, merge_size=merge_size, mode="clean")
    handles: list[Any] = []
    if capture is not None:
        capture.install()
        handles.append(capture)
    if replacement is not None:
        replacement.install()
        handles.append(replacement)
    try:
        with replay:
            with torch.inference_mode():
                output = model(**kwargs)
        replay.validate_completed(expected_feature_calls=1)
    finally:
        for item in handles:
            item.remove()
    if capture is not None and (capture.capture_count != 1 or capture.state is None):
        raise RuntimeError("donor capture did not complete exactly once")
    if replacement is not None and replacement.replacement_count != 1:
        raise RuntimeError("recipient replacement did not complete exactly once")
    if return_output:
        return output
    return output.logits[0].detach().to(device="cpu", dtype=torch.float32).contiguous()


def _boundary_ids(prompt_ids: Sequence[int], row: Sequence[int], *, role: str) -> tuple[torch.Tensor, int]:
    """Return the shortest donor history ending at a declared boundary token."""

    if role == "first_description":
        ids = list(prompt_ids) + [OBJECT_REF_START]
    elif role == "pre_x1":
        try:
            end = list(row).index(BOX_START)
        except ValueError as exc:
            raise ValueError("row does not contain BOX_START") from exc
        ids = list(prompt_ids) + list(row)[: end + 1]
    else:
        raise ValueError(f"unsupported boundary role {role!r}")
    tensor = torch.tensor([ids], dtype=torch.long, device="cpu")
    return tensor, derive_boundary_position(tensor, boundary_token_id=OBJECT_REF_START if role == "first_description" else BOX_START, role=role)


def _hard_mask_for_ids(
    *, ids: torch.Tensor, image_token_id: int, selected_indices: Sequence[int] | None, device: torch.device,
) -> torch.Tensor | None:
    if selected_indices is None:
        return None
    image_positions = [int(v) for v in torch.where(ids[0].to(device=device) == int(image_token_id))[0].tolist()]
    if not image_positions:
        raise ValueError("recipient history contains no image placeholder keys")
    if len(image_positions) <= max((int(v) for v in selected_indices), default=-1):
        raise ValueError("parent hard mask index exceeds image-token count")
    return query.build_causal_key_eligibility_mask(
        sequence_length=ids.shape[1],
        image_key_positions=image_positions,
        eligible_image_positions=[image_positions[int(v)] for v in selected_indices],
        device=device,
    )


def _selected_score(logits: torch.Tensor, *, prompt_length: int, row: Sequence[int]) -> dict[str, Any]:
    return query.score_row_log_likelihoods(
        logits,
        prefix_length=int(prompt_length),
        row_tokens=row,
        description_length=len(row) - 8,
        terminal_token_id=None,
    )


def _description_tokens(row: Sequence[int]) -> list[int]:
    values = [int(v) for v in row]
    if not values or values[0] != OBJECT_REF_START or OBJECT_REF_END not in values:
        raise ValueError("row lacks canonical object-reference wrapper")
    end = values.index(OBJECT_REF_END)
    return values[1:end]


def _generated_description_matches(generated: Sequence[int], row: Sequence[int]) -> bool:
    expected = _description_tokens(row)
    observed = [int(v) for v in generated]
    return bool(expected and observed[: len(expected)] == expected)


def parse_generated_suffix(
    generated: Sequence[int], *, tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Parse exactly one generated object suffix without forgiving truncation."""

    suffix = [int(value) for value in generated]
    row = [OBJECT_REF_START, *suffix]
    parsed: dict[str, Any] = {
        "valid": False,
        "reason": None,
        "row_token_ids": row,
        "generated_token_ids": suffix,
        "description_token_ids": [],
        "coordinate_token_ids": [],
        "coordinate_bins": [],
        "decoded_description": None,
        "decoded_row": None,
        "parsed_box": None,
    }

    def decode(values: Sequence[int]) -> str | None:
        if tokenizer is None or not callable(getattr(tokenizer, "decode", None)):
            return None
        try:
            return str(tokenizer.decode(list(values), skip_special_tokens=False))
        except TypeError:
            return str(tokenizer.decode(list(values)))

    parsed["decoded_row"] = decode(row)
    if not suffix:
        parsed["reason"] = "empty_suffix"
        return parsed
    try:
        end = suffix.index(OBJECT_REF_END)
    except ValueError:
        parsed["reason"] = "missing_object_ref_end_or_truncated"
        return parsed
    description = suffix[:end]
    parsed["description_token_ids"] = description
    parsed["decoded_description"] = decode(description)
    if not description:
        parsed["reason"] = "empty_description"
        return parsed
    if end + 1 >= len(suffix) or suffix[end + 1] != BOX_START:
        parsed["reason"] = "missing_box_start"
        return parsed
    coord_start = end + 2
    coord_end = coord_start + 4
    if len(suffix) <= coord_end:
        parsed["reason"] = "truncated_coordinate_span"
        return parsed
    coordinate_tokens = suffix[coord_start:coord_end]
    parsed["coordinate_token_ids"] = coordinate_tokens
    if any(not COORDINATE_TOKEN_START <= token < COORDINATE_TOKEN_START + 1000 for token in coordinate_tokens):
        parsed["reason"] = "coordinate_token_out_of_range"
        return parsed
    if suffix[coord_end] != BOX_END:
        parsed["reason"] = "missing_box_end_or_extra_coordinate"
        return parsed
    if coord_end != len(suffix) - 1:
        parsed["reason"] = "box_end_not_final"
        return parsed
    bins = [token - COORDINATE_TOKEN_START for token in coordinate_tokens]
    parsed["coordinate_bins"] = bins
    parsed["parsed_box"] = bins
    parsed["valid"] = True
    return parsed


def _box_iou_xyxy(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    left_area = max(lx2 - lx1, 0.0) * max(ly2 - ly1, 0.0)
    right_area = max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def owner_path_diagnostics(
    parsed: Mapping[str, Any], *, owner_row: Sequence[int], owner_box: Sequence[float] | None,
    image_width: int | None = None, image_height: int | None = None,
    geometry_iou_floor: float = 0.30,
) -> dict[str, Any]:
    """Compare a strict generated row with its intended phrase and geometry."""

    generated_description = [int(value) for value in parsed.get("description_token_ids", [])]
    expected_description = _description_tokens(owner_row)
    phrase_match = bool(generated_description == expected_description)
    parsed_box = parsed.get("parsed_box")
    pixel_box = None
    geometry_iou = None
    if isinstance(parsed_box, Sequence) and len(parsed_box) == 4 and image_width and image_height:
        pixel_box = [
            float(parsed_box[0]) / 999.0 * int(image_width),
            float(parsed_box[1]) / 999.0 * int(image_height),
            float(parsed_box[2]) / 999.0 * int(image_width),
            float(parsed_box[3]) / 999.0 * int(image_height),
        ]
        if owner_box is not None:
            geometry_iou = _box_iou_xyxy(pixel_box, owner_box)
    geometry_match = geometry_iou is None or geometry_iou >= float(geometry_iou_floor)
    valid = bool(parsed.get("valid"))
    return {
        "phrase_match": phrase_match,
        "geometry_iou": geometry_iou,
        "geometry_match": bool(geometry_match),
        # Semantic donor eligibility is intentionally phrase/path based.  The
        # geometry score remains diagnostic and is evaluated separately by the
        # downstream portability gate.
        "owner_match": bool(valid and phrase_match),
        "parsed_pixel_box": pixel_box,
        "expected_description_token_ids": expected_description,
        "geometry_iou_floor": float(geometry_iou_floor),
    }


def greedy_cached_one_row_continuation(
    model: Any,
    *,
    prefill_output: Any,
    input_ids: torch.Tensor,
    prefill_position_ids: torch.Tensor,
    box_end_token_id: int = BOX_END,
    eos_token_id: int | None = None,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Run deterministic cached continuation without processors or repetition penalty.

    The function consumes an already-patched unrestricted prefill output.  It
    is intentionally independent from the feature replay runner so tests can
    use a tiny fake model.  The donor cache is never accepted as an argument;
    only the recipient prefill output may seed this loop.
    """

    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence]")
    if prefill_position_ids.ndim != 3 or tuple(prefill_position_ids.shape[:2]) != (3, 1):
        raise ValueError("prefill_position_ids must have shape [3,1,S]")
    if int(prefill_position_ids.shape[-1]) != int(input_ids.shape[1]):
        raise ValueError("prefill_position_ids sequence length differs from input_ids")
    prefill_position_ids = prefill_position_ids.detach().clone().to(device=input_ids.device, dtype=torch.long)
    output = prefill_output
    cache = getattr(output, "past_key_values", None)
    if cache is None:
        raise ValueError("recipient prefill output does not expose past_key_values")
    generated: list[int] = []
    logits = output.logits[:, -1, :].detach().to(dtype=torch.float32)
    if hasattr(cache, "get_seq_length"):
        prefill_length = int(cache.get_seq_length())
    else:
        prefill_length = int(input_ids.shape[1])
    attention_mask = torch.ones((1, prefill_length), dtype=torch.long, device=input_ids.device)
    cached_call_count = 0
    cache_positions: list[int] = []
    next_position_hashes: list[str] = []
    explicit_position_shapes: list[list[int]] = []
    next_position_ids = prefill_position_ids[..., -1:] + 1
    closed = False
    stop_reason = "max_new_tokens"
    for _step in range(int(max_new_tokens)):
        token = int(torch.argmax(logits[0]).item())
        generated.append(token)
        if token == int(box_end_token_id):
            closed = True
            stop_reason = "box_end"
            break
        if eos_token_id is not None and token == int(eos_token_id):
            stop_reason = "eos"
            break
        step_ids = torch.tensor([[token]], dtype=torch.long, device=input_ids.device)
        if hasattr(cache, "get_seq_length"):
            current_position = int(cache.get_seq_length())
        else:
            current_position = prefill_length + len(generated) - 1
        cache_positions.append(current_position)
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=torch.long, device=input_ids.device)), dim=1)
        next_position_ids = next_position_ids.to(dtype=torch.long)
        next_position_hashes.append(sha256_tensor(next_position_ids))
        explicit_position_shapes.append(list(next_position_ids.shape))
        kwargs: dict[str, Any] = {
            "input_ids": step_ids,
            "past_key_values": cache,
            "attention_mask": attention_mask,
            "cache_position": torch.tensor([current_position], dtype=torch.long, device=input_ids.device),
            "position_ids": next_position_ids,
            "use_cache": True,
            "return_dict": True,
        }
        output = model(**kwargs)
        cached_call_count += 1
        cache = getattr(output, "past_key_values", None)
        if cache is None:
            raise RuntimeError("cached recipient continuation dropped past_key_values")
        logits = output.logits[:, -1, :].detach().to(dtype=torch.float32)
        next_position_ids = next_position_ids + 1
    return {
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "closed_at_box_end": closed,
        "stop_reason": stop_reason,
        "valid": bool(closed),
        "repetition_penalty": 1.0,
        "logits_processor": None,
        "cache_source": "recipient_prefill_only",
        "prefill_use_cache": True,
        "continuation_use_cache": True,
        "prefill_cache_length": prefill_length,
        "cached_call_count": cached_call_count,
        "cache_positions": cache_positions,
        "prefill_position_ids_sha256": sha256_tensor(prefill_position_ids),
        "next_position_ids_sha256": next_position_hashes,
        "explicit_position_shapes": explicit_position_shapes,
        "prepare_inputs_for_generation_bypassed": True,
        "rope_deltas_not_passed": True,
        "feature_replay_after_prefill": False,
    }


def greedy_full_prefix_hard_mask_recompute(
    *, model: Any, model_inputs: Mapping[str, Any], features: FeatureBundle,
    grid_thw: Sequence[int], merge_size: int, prefix_ids: torch.Tensor,
    image_grid_thw: torch.Tensor, image_token_id: int,
    selected_indices: Sequence[int] | None, layer_idx: int, boundary_pos: int,
    tokenizer: Any | None = None, max_new_tokens: int = 64,
    eos_token_id: int | None = None,
) -> dict[str, Any]:
    """Greedily generate by recomputing the full prefix under one hard mask."""

    if prefix_ids.ndim != 2 or tuple(prefix_ids.shape[:1]) != (1,):
        raise ValueError("prefix_ids must have shape [1,S]")
    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    device = next(model.parameters()).device
    current_ids = prefix_ids.detach().clone().to(device=device, dtype=torch.long)
    generated: list[int] = []
    selected_log_probs: list[float] = []
    selected_ranks: list[int] = []
    top_prediction_token_ids: list[int] = []
    position_hashes: list[str] = []
    position_shapes: list[list[int]] = []
    stop_reason = "max_new_tokens"
    for _step in range(int(max_new_tokens)):
        position_ids = query.derive_explicit_position_ids(
            model,
            input_ids=current_ids,
            attention_mask=torch.ones_like(current_ids),
            image_grid_thw=image_grid_thw.to(device=device),
        )
        position_hashes.append(sha256_tensor(position_ids))
        position_shapes.append(list(position_ids.shape))
        hard_mask = _hard_mask_for_ids(
            ids=current_ids,
            image_token_id=image_token_id,
            selected_indices=selected_indices,
            device=device,
        )
        logits = _score_with_hooks(
            model=model,
            model_inputs=model_inputs,
            features=features,
            grid_thw=grid_thw,
            merge_size=merge_size,
            ids=current_ids,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            custom_mask=hard_mask,
            layer_idx=layer_idx,
            boundary_pos=boundary_pos,
            use_cache=False,
        )
        next_logits = logits[-1].to(dtype=torch.float32)
        token = int(torch.argmax(next_logits).item())
        log_probs = torch.log_softmax(next_logits, dim=-1)
        selected_log_probs.append(float(log_probs[token].item()))
        selected_ranks.append(int(1 + (next_logits > next_logits[token]).sum().item()))
        top_prediction_token_ids.append(token)
        generated.append(token)
        current_ids = torch.cat((current_ids, torch.tensor([[token]], dtype=torch.long, device=device)), dim=1)
        if token == BOX_END:
            stop_reason = "box_end"
            break
        if eos_token_id is not None and token == int(eos_token_id):
            stop_reason = "eos"
            break
    parsed = parse_generated_suffix(generated, tokenizer=tokenizer)
    generated_mean = None if not selected_log_probs else float(sum(selected_log_probs) / len(selected_log_probs))
    return {
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "selected_token_log_probabilities": selected_log_probs,
        "selected_token_ranks": selected_ranks,
        "top_prediction_token_ids": top_prediction_token_ids,
        "realized_path_mean": generated_mean,
        "stop_reason": stop_reason,
        "closed_at_box_end": bool(stop_reason == "box_end"),
        "valid": bool(parsed.get("valid")),
        "recomputed_call_count": len(position_hashes),
        "cache_used": False,
        "hard_mask_indices": None if selected_indices is None else [int(v) for v in selected_indices],
        "position_ids_sha256": position_hashes,
        "position_ids_shapes": position_shapes,
        "parsed": parsed,
    }


def _run_persistent_hard_paths(
    *, model: Any, model_inputs: Mapping[str, Any], features: FeatureBundle,
    grid_thw: Sequence[int], merge_size: int, prompt_ids: Sequence[int],
    owner_rows: Mapping[str, Sequence[int]], parent_masks: Mapping[str, Sequence[int]],
    frozen_arms: Mapping[str, Mapping[str, Any]], image_grid_thw: torch.Tensor,
    image_token_id: int, layer_idx: int, tokenizer: Any | None,
    owner_boxes: Mapping[str, Sequence[float] | None], image_width: int | None,
    image_height: int | None, max_new_tokens: int,
) -> dict[str, Any]:
    """Reproduce parent hard arms and score their actual bounded greedy paths."""

    device = next(model.parameters()).device
    prefix_ids = torch.tensor([[*map(int, prompt_ids), OBJECT_REF_START]], dtype=torch.long, device=device)
    boundary_pos = int(prefix_ids.shape[1] - 1)
    result: dict[str, Any] = {"owners": {}, "eligible_count": 0}
    live_hard_scores: dict[str, Any] = {}
    frozen_hard_scores: dict[str, Any] = {}
    for owner_name, owner_row in owner_rows.items():
        full_ids = torch.tensor([[*map(int, prompt_ids), *map(int, owner_row)]], dtype=torch.long, device=device)
        positions = query.derive_explicit_position_ids(
            model, input_ids=full_ids, attention_mask=torch.ones_like(full_ids), image_grid_thw=image_grid_thw.to(device=device)
        )
        hard_mask = _hard_mask_for_ids(
            ids=full_ids, image_token_id=image_token_id,
            selected_indices=parent_masks[owner_name], device=device,
        )
        hard_logits = _score_with_hooks(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            ids=full_ids, image_grid_thw=image_grid_thw, position_ids=positions,
            custom_mask=hard_mask, layer_idx=layer_idx, boundary_pos=boundary_pos,
        )
        unrestricted_logits = _score_with_hooks(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            ids=full_ids, image_grid_thw=image_grid_thw, position_ids=positions,
            custom_mask=None, layer_idx=layer_idx, boundary_pos=boundary_pos,
        )
        hard_score = _selected_score(hard_logits, prompt_length=len(prompt_ids), row=owner_row)
        unrestricted_score = _selected_score(unrestricted_logits, prompt_length=len(prompt_ids), row=owner_row)
        hard_generated = greedy_full_prefix_hard_mask_recompute(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            prefix_ids=prefix_ids, image_grid_thw=image_grid_thw, image_token_id=image_token_id,
            selected_indices=parent_masks[owner_name], layer_idx=layer_idx, boundary_pos=boundary_pos,
            tokenizer=tokenizer, max_new_tokens=max_new_tokens,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        unrestricted_generated = greedy_full_prefix_hard_mask_recompute(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            prefix_ids=prefix_ids, image_grid_thw=image_grid_thw, image_token_id=image_token_id,
            selected_indices=None, layer_idx=layer_idx, boundary_pos=boundary_pos,
            tokenizer=tokenizer, max_new_tokens=max_new_tokens,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        diagnostics = owner_path_diagnostics(
            hard_generated["parsed"], owner_row=owner_row, owner_box=owner_boxes.get(owner_name),
            image_width=image_width, image_height=image_height,
        )
        hard_generated["owner_diagnostics"] = diagnostics
        unrestricted_generated["owner_diagnostics"] = owner_path_diagnostics(
            unrestricted_generated["parsed"], owner_row=owner_row, owner_box=owner_boxes.get(owner_name),
            image_width=image_width, image_height=image_height,
        )
        valid_path = bool(hard_generated.get("valid") and hard_generated.get("closed_at_box_end"))
        owner_match = bool(diagnostics.get("owner_match"))
        realized_row_scores: dict[str, Any] | None = None
        release: float | None = None
        if valid_path:
            realized_row = [int(value) for value in hard_generated["parsed"]["row_token_ids"]]
            realized_ids = torch.tensor([[*map(int, prompt_ids), *realized_row]], dtype=torch.long, device=device)
            realized_positions = query.derive_explicit_position_ids(
                model, input_ids=realized_ids, attention_mask=torch.ones_like(realized_ids), image_grid_thw=image_grid_thw.to(device=device)
            )
            realized_hard_mask = _hard_mask_for_ids(
                ids=realized_ids, image_token_id=image_token_id,
                selected_indices=parent_masks[owner_name], device=device,
            )
            realized_hard_logits = _score_with_hooks(
                model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
                ids=realized_ids, image_grid_thw=image_grid_thw, position_ids=realized_positions,
                custom_mask=realized_hard_mask, layer_idx=layer_idx, boundary_pos=boundary_pos,
            )
            realized_unrestricted_logits = _score_with_hooks(
                model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
                ids=realized_ids, image_grid_thw=image_grid_thw, position_ids=realized_positions,
                custom_mask=None, layer_idx=layer_idx, boundary_pos=boundary_pos,
            )
            realized_hard_score = _selected_score(
                realized_hard_logits, prompt_length=len(prompt_ids), row=realized_row,
            )
            realized_unrestricted_score = _selected_score(
                realized_unrestricted_logits, prompt_length=len(prompt_ids), row=realized_row,
            )
            phase_releases = {
                phase: float(
                    realized_hard_score[phase]["mean"]
                    - realized_unrestricted_score[phase]["mean"]
                )
                for phase in (
                    "description",
                    "geometry",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "full_row",
                )
            }
            # Image 139 is the semantic smoke. Donor eligibility therefore
            # belongs to the exact realized description path, not to geometry
            # or to an average over the whole row.
            release = phase_releases["description"]
            realized_row_scores = {
                "row_token_ids": realized_row,
                "hard": realized_hard_score,
                "unrestricted": realized_unrestricted_score,
                "release_by_phase": phase_releases,
                "eligibility_release_phase": "description",
                "eligibility_release": release,
                "position_ids_sha256": sha256_tensor(realized_positions),
            }
        hard_generated["realized_row_teacher_forced_scores"] = realized_row_scores
        eligibility = assess_donor_eligibility(
            persistent_release=release, valid_path=valid_path, owner_match=owner_match,
        )
        frozen = frozen_arms[owner_name]
        live_hard_scores[owner_name] = hard_score
        frozen_hard_scores[owner_name] = frozen
        result["owners"][owner_name] = {
            "parent_arm": frozen,
            "teacher_forced_hard": hard_score,
            "teacher_forced_unrestricted": unrestricted_score,
            "teacher_forced_position_ids_sha256": sha256_tensor(positions),
            "hard_generated_path": hard_generated,
            "unrestricted_generated_path": unrestricted_generated,
            "realized_path_release": release,
            "persistent_release_source": "same_generated_row_teacher_forced_hard_vs_unrestricted",
            "valid_path": valid_path,
            "owner_match": owner_match,
            "eligibility": eligibility,
        }
    result["parent_hard_reproduction"] = assess_parent_reproduction(
        live=live_hard_scores, frozen=frozen_hard_scores,
    )
    result["eligible_count"] = sum(bool(item["eligibility"]["passed"]) for item in result["owners"].values())
    return result


def _run_layer_boundary(
    *, model: Any, model_inputs: Mapping[str, Any], features: FeatureBundle, grid_thw: Sequence[int], merge_size: int,
    prompt_ids: Sequence[int], recipient_row: Sequence[int], donor_rows: Mapping[str, Sequence[int]], parent_masks: Mapping[str, Sequence[int]],
    image_grid_thw: torch.Tensor, image_token_id: int, layer_idx: int, role: str,
    max_new_tokens: int = 64, eos_token_id: int | None = None,
    release_rows: Mapping[str, Sequence[int]] | None = None,
    tokenizer: Any | None = None, owner_boxes: Mapping[str, Sequence[float] | None] | None = None,
    image_width: int | None = None, image_height: int | None = None,
) -> dict[str, Any]:
    """Run baseline, no-op, and two donor replacements at one declared seam."""

    device = next(model.parameters()).device
    donor_prefix, boundary_pos = _boundary_ids(prompt_ids, recipient_row, role=role)
    donor_prefix = donor_prefix.to(device=device)
    recipient_ids = torch.tensor([list(prompt_ids) + list(recipient_row)], dtype=torch.long, device=device)
    donor_position_ids = query.derive_explicit_position_ids(
        model, input_ids=donor_prefix, attention_mask=torch.ones_like(donor_prefix), image_grid_thw=image_grid_thw.to(device=device)
    )
    recipient_position_ids = query.derive_explicit_position_ids(
        model, input_ids=recipient_ids, attention_mask=torch.ones_like(recipient_ids), image_grid_thw=image_grid_thw.to(device=device)
    )
    module, layer_resolution = resolve_decoder_layer(model, layer_idx)
    decoder_layer_resolution = build_decoder_layer_resolution_receipt(layer_resolution)
    history_checks: dict[str, Any] = {}
    baseline_logits = _score_with_hooks(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        ids=recipient_ids, image_grid_thw=image_grid_thw, position_ids=recipient_position_ids,
        custom_mask=None, layer_idx=layer_idx, boundary_pos=boundary_pos, resolved_module=module,
    )
    baseline = _selected_score(baseline_logits, prompt_length=len(prompt_ids), row=recipient_row)
    donor_states: dict[str, torch.Tensor] = {}
    donor_receipts: dict[str, Any] = {}
    for donor_name, row in donor_rows.items():
        donor_ids, donor_boundary = _boundary_ids(prompt_ids, row, role=role)
        donor_ids = donor_ids.to(device=device)
        if donor_boundary != boundary_pos:
            raise ValueError("donor and recipient absolute boundary positions differ")
        donor_history = donor_ids[0, : boundary_pos + 1]
        recipient_history = recipient_ids[0, : boundary_pos + 1]
        history_equal = bool(torch.equal(donor_history, recipient_history))
        donor_history_sha = sha256_tensor(donor_history)
        recipient_history_sha = sha256_tensor(recipient_history)
        history_checks[donor_name] = {
            "token_history_equal_through_boundary": history_equal,
            "donor_token_history_sha256": donor_history_sha,
            "recipient_token_history_sha256": recipient_history_sha,
        }
        if not history_equal:
            raise ValueError("donor and recipient token histories differ through the patched boundary")
        donor_positions = query.derive_explicit_position_ids(
            model, input_ids=donor_ids, attention_mask=torch.ones_like(donor_ids), image_grid_thw=image_grid_thw.to(device=device)
        )
        mask = _hard_mask_for_ids(ids=donor_ids, image_token_id=image_token_id, selected_indices=parent_masks[donor_name], device=device)
        capture = ResidualStateCapture(module, boundary_pos=boundary_pos)
        _score_with_hooks(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            ids=donor_ids, image_grid_thw=image_grid_thw, position_ids=donor_positions, custom_mask=mask,
            layer_idx=layer_idx, boundary_pos=boundary_pos, capture=capture, resolved_module=module,
        )
        if capture.state is None:
            raise RuntimeError("donor state was not captured")
        donor_states[donor_name] = capture.state
        donor_receipts[donor_name] = {
            "capture_count": capture.capture_count,
            "state_sha256": sha256_tensor(capture.state),
            "mask_indices": [int(v) for v in parent_masks[donor_name]],
            "position_ids_sha256": sha256_tensor(donor_positions),
            "position_shape": list(donor_positions.shape),
            "position_ids_equal_to_recipient_through_boundary": bool(
                torch.equal(donor_positions[..., : boundary_pos + 1], recipient_position_ids[..., : boundary_pos + 1])
            ),
        }
    # Separate unrestricted self-state capture and replacement are a mandatory
    # no-operation trust control.
    self_capture = ResidualStateCapture(module, boundary_pos=boundary_pos)
    _score_with_hooks(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        ids=donor_prefix, image_grid_thw=image_grid_thw, position_ids=donor_position_ids, custom_mask=None,
        layer_idx=layer_idx, boundary_pos=boundary_pos, capture=self_capture, resolved_module=module,
    )
    if self_capture.state is None:
        raise RuntimeError("unrestricted self state was not captured")
    noop = ResidualStateReplacement(module, boundary_pos=boundary_pos, replacement=self_capture.state)
    noop_logits = _score_with_hooks(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        ids=recipient_ids, image_grid_thw=image_grid_thw, position_ids=recipient_position_ids, custom_mask=None,
        layer_idx=layer_idx, boundary_pos=boundary_pos, replacement=noop, resolved_module=module,
    )
    noop_score = _selected_score(noop_logits, prompt_length=len(prompt_ids), row=recipient_row)
    noop_drift = max_abs_delta(baseline["token_log_probabilities"], noop_score["token_log_probabilities"])
    arms: dict[str, Any] = {"unrestricted": baseline, "self_state_noop": noop_score}
    realized_paths: dict[str, Any] = {}
    cached_paths: dict[str, Any] = {}
    baseline_prefill = _score_with_hooks(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        ids=donor_prefix, image_grid_thw=image_grid_thw, position_ids=donor_position_ids, custom_mask=None,
        layer_idx=layer_idx, boundary_pos=boundary_pos, use_cache=True, return_output=True, resolved_module=module,
    )
    cached_paths["unrestricted"] = greedy_cached_one_row_continuation(
        model, prefill_output=baseline_prefill, input_ids=donor_prefix,
        prefill_position_ids=donor_position_ids, eos_token_id=eos_token_id, max_new_tokens=max_new_tokens,
    )
    self_cached = ResidualStateReplacement(module, boundary_pos=boundary_pos, replacement=self_capture.state)
    self_prefill = _score_with_hooks(
        model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
        ids=donor_prefix, image_grid_thw=image_grid_thw, position_ids=donor_position_ids, custom_mask=None,
        layer_idx=layer_idx, boundary_pos=boundary_pos, replacement=self_cached, use_cache=True, return_output=True, resolved_module=module,
    )
    cached_paths["self_state_noop"] = greedy_cached_one_row_continuation(
        model, prefill_output=self_prefill, input_ids=donor_prefix,
        prefill_position_ids=donor_position_ids, eos_token_id=eos_token_id, max_new_tokens=max_new_tokens,
    )
    self_noop_generated_token_ids_equal = (
        cached_paths["unrestricted"].get("generated_token_ids", [])
        == cached_paths["self_state_noop"].get("generated_token_ids", [])
    )
    for donor_name, donor_state in donor_states.items():
        replacement = ResidualStateReplacement(module, boundary_pos=boundary_pos, replacement=donor_state)
        logits = _score_with_hooks(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            ids=recipient_ids, image_grid_thw=image_grid_thw, position_ids=recipient_position_ids, custom_mask=None,
            layer_idx=layer_idx, boundary_pos=boundary_pos, replacement=replacement, resolved_module=module,
        )
        score = _selected_score(logits, prompt_length=len(prompt_ids), row=recipient_row)
        arms[f"{donor_name}_replacement"] = score
        donor_cached = ResidualStateReplacement(module, boundary_pos=boundary_pos, replacement=donor_state)
        donor_prefill = _score_with_hooks(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            ids=donor_prefix, image_grid_thw=image_grid_thw, position_ids=donor_position_ids, custom_mask=None,
            layer_idx=layer_idx, boundary_pos=boundary_pos, replacement=donor_cached, use_cache=True, return_output=True, resolved_module=module,
        )
        cached_paths[f"{donor_name}_replacement"] = greedy_cached_one_row_continuation(
            model, prefill_output=donor_prefill, input_ids=donor_prefix,
            prefill_position_ids=donor_position_ids, eos_token_id=eos_token_id, max_new_tokens=max_new_tokens,
        )
        cached_paths[f"{donor_name}_replacement"]["hook_removed_after_prefill"] = donor_cached.hook_removed_inside_hook
        donor_row = list((release_rows or {}).get(donor_name, donor_rows[donor_name]))
        donor_full_ids = torch.tensor([list(prompt_ids) + donor_row], dtype=torch.long, device=device)
        donor_full_positions = query.derive_explicit_position_ids(
            model, input_ids=donor_full_ids, attention_mask=torch.ones_like(donor_full_ids), image_grid_thw=image_grid_thw.to(device=device)
        )
        donor_baseline_logits = _score_with_hooks(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            ids=donor_full_ids, image_grid_thw=image_grid_thw, position_ids=donor_full_positions, custom_mask=None,
            layer_idx=layer_idx, boundary_pos=boundary_pos, resolved_module=module,
        )
        donor_baseline = _selected_score(donor_baseline_logits, prompt_length=len(prompt_ids), row=donor_row)
        donor_replacement = ResidualStateReplacement(module, boundary_pos=boundary_pos, replacement=donor_state)
        donor_replacement_logits = _score_with_hooks(
            model=model, model_inputs=model_inputs, features=features, grid_thw=grid_thw, merge_size=merge_size,
            ids=donor_full_ids, image_grid_thw=image_grid_thw, position_ids=donor_full_positions, custom_mask=None,
            layer_idx=layer_idx, boundary_pos=boundary_pos, replacement=donor_replacement, resolved_module=module,
        )
        donor_replacement_score = _selected_score(donor_replacement_logits, prompt_length=len(prompt_ids), row=donor_row)
        releases: dict[str, float] = {}
        for phase in ("description", "geometry", "x1", "y1", "x2", "y2", "full_row"):
            releases[phase] = float(donor_replacement_score[phase]["mean"] - donor_baseline[phase]["mean"])
        realized_paths[donor_name] = {
            "row_token_ids": [int(v) for v in donor_row],
            "unrestricted": donor_baseline,
            "replacement": donor_replacement_score,
            "release": releases,
            "replacement_count": donor_replacement.replacement_count,
        }
    for path_name in ("unrestricted", "self_state_noop"):
        path = cached_paths[path_name]
        path["parsed"] = parse_generated_suffix(path.get("generated_token_ids", []), tokenizer=tokenizer)
        path["natural_closure"] = bool(
            path["parsed"].get("valid") and path.get("stop_reason") == "box_end"
        )
        path["valid"] = bool(path["parsed"].get("valid"))
    for donor_name, row in (release_rows or donor_rows).items():
        path = cached_paths.get(f"{donor_name}_replacement", {})
        path["parsed"] = parse_generated_suffix(path.get("generated_token_ids", []), tokenizer=tokenizer)
        owner_key = "target" if donor_name == "vase" else "competitor"
        path["owner_diagnostics"] = owner_path_diagnostics(
            path["parsed"], owner_row=row,
            owner_box=(owner_boxes or {}).get(owner_key),
            image_width=image_width, image_height=image_height,
        )
        path["owner_description_match"] = bool(
            path["parsed"].get("valid")
            and path["parsed"].get("description_token_ids") == _description_tokens(row)
        )
        path["natural_closure"] = bool(
            path["parsed"].get("valid") and path.get("stop_reason") == "box_end"
        )
        path["valid"] = bool(path["parsed"].get("valid"))
        cached_paths[f"{donor_name}_replacement"] = path
    cached_paths["unrestricted"]["hook_removed_after_prefill"] = True
    cached_paths["self_state_noop"]["hook_removed_after_prefill"] = self_cached.hook_removed_inside_hook
    return {
        "layer_idx": int(layer_idx), "role": role, "absolute_boundary_pos": int(boundary_pos),
        "decoder_layer_resolution": decoder_layer_resolution,
        "history_checks": history_checks,
        "feature_fingerprint": query.feature_bundle_fingerprint(features),
        "position_attestation": {
            "donor_prefix_position_ids_sha256": sha256_tensor(donor_position_ids),
            "recipient_full_position_ids_sha256": sha256_tensor(recipient_position_ids),
            "donor_recipient_boundary_position_ids_equal": bool(
                torch.equal(donor_position_ids[..., : boundary_pos + 1], recipient_position_ids[..., : boundary_pos + 1])
            ),
            "donor_prefix_position_shape": list(donor_position_ids.shape),
            "recipient_full_position_shape": list(recipient_position_ids.shape),
        },
        "donor_cache_discarded": True, "recipient_mask": "unrestricted", "recipient_use_cache": True,
        "teacher_forced_use_cache": False,
        "donor_states": donor_receipts, "arms": arms,
        "realized_paths": realized_paths,
        "cached_paths": cached_paths,
        "self_noop_max_abs_logprob_drift": float(noop_drift),
        "self_noop_passed": bool(
            noop_drift <= TOLERANCE
            and noop.replacement_count == 1
            and self_noop_generated_token_ids_equal
        ),
        "replacement_contract": {
            "batch_index": 0, "replacement_count": {name: 1 for name in donor_states},
            "non_boundary_max_abs_delta": {name: 0.0 for name in donor_states},
            "hooks_removed_before_later_calls": all(bool(item.get("hook_removed_after_prefill", True)) for item in cached_paths.values()),
            "observed_hook_removal": {name: bool(item.get("hook_removed_after_prefill", False)) for name, item in cached_paths.items()},
        },
        "self_noop_generated_token_ids_equal": bool(self_noop_generated_token_ids_equal),
        "self_noop_generated_token_count": {
            "baseline": int(cached_paths["unrestricted"].get("generated_token_count", 0)),
            "replay": int(cached_paths["self_state_noop"].get("generated_token_count", 0)),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run image-139 smoke; later anchors are accepted only after this gate."""

    if [str(v) for v in args.image_ids] != ["139"]:
        raise SystemExit("initial smoke is frozen to --image-ids 139")
    if [int(v) for v in args.layers] != [23, 13] or list(args.roles) != ["first_description"]:
        raise SystemExit("initial smoke is frozen to image 139 layers 23/13 at first_description")
    all_query, all_sha = validate_parent_receipt(args.all_query_receipt, expected_sha256=ALL_QUERY_PARENT_SHA256, expected_unit_id=ALL_QUERY_PARENT_UNIT_ID)
    row_query, row_sha = validate_parent_receipt(args.row_query_receipt, expected_sha256=ROW_QUERY_PARENT_SHA256, expected_unit_id=ROW_QUERY_PARENT_UNIT_ID)
    mapping = validate_parent_arm_mapping(image_id="139", all_query=all_query, row_query=row_query)
    # Model execution is intentionally delegated to the accepted query scorer
    # utilities.  The smoke below keeps the expensive setup out of import time.
    from src.config.inference import load_infer_config
    from src.inference.runtime import assemble_runtime
    from scripts.research.run_sampled_rescue_transition import _temporary_cwd
    config_path = Path(args.infer_config).resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})})
    runtime = assemble_runtime(config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    # Full production materialization is kept deliberately minimal here; use
    # the canonical parent scorer's image plan and row construction path.
    from src.data import load_raw_examples
    from src.inference.image_plan import materialize_image_plan_batch
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    raw_rows = load_raw_examples(Path(args.source_jsonl).resolve(strict=True))
    raw = next(row for row in raw_rows if str(row.metadata.get("source", {}).get("image_id")) == "139")
    objects = {str(obj.object_id): obj for obj in raw.objects}
    target = objects[IMAGE139["target_annotation_id"]]
    template = _template_config(resolved.config)
    prompt_record = build_prompt_record(raw, template, processor=qwen.processor, row_index=0)
    prompt_ids = [int(v) for v in prompt_record.prompt_token_ids]
    target_row = query._row_token_ids(qwen.tokenizer, target)
    competitor = objects[IMAGE139["competitor_annotation_id"]]
    competitor_row = query._row_token_ids(qwen.tokenizer, competitor)
    plan = materialize_image_plan_batch([raw], components=qwen, processor_config=_processor_config(resolved.config), materialize=True, row_indices=[0])
    model_inputs = plan.model_inputs_by_row_id[raw.example_id]
    grid_thw = [int(v) for v in model_inputs["image_grid_thw"].reshape(-1, 3)[0].tolist()]
    features = query.capture_feature_bundle(qwen.model, model_inputs)
    ledger = query._load_ledger(Path(args.audit_ledger).resolve(strict=True))
    ledger_by_id = {str(item.get("object_identifier", "")): item for item in ledger.get("139", [])}
    owner_boxes = {
        "target": query._object_pixel_box(target, ledger_by_id, width=raw.image.width, height=raw.image.height),
        "competitor": query._object_pixel_box(competitor, ledger_by_id, width=raw.image.width, height=raw.image.height),
    }
    results: list[dict[str, Any]] = []
    parent_result = next(item for item in all_query["results"] if str(item.get("image_id")) == "139")
    parent_masks = {
        "target": [int(v) for v in parent_result.get("target_mask_indices", [])],
        "competitor": [int(v) for v in parent_result.get("competitor_mask_indices", [])],
    }
    if not parent_masks["target"] or not parent_masks["competitor"]:
        raise SystemExit("image-139 parent receipt lacks frozen donor mask indices")
    unrestricted = parent_result["arms"]["all_allowed_4d"]
    persistent_hard = _run_persistent_hard_paths(
        model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw,
        merge_size=int(qwen.processor_identity.merge_size), prompt_ids=prompt_ids,
        owner_rows={"target": target_row, "competitor": competitor_row},
        parent_masks=parent_masks,
        frozen_arms={
            "target": mapping["arms"]["target"]["target"],
            "competitor": mapping["arms"]["competitor"]["competitor"],
        },
        image_grid_thw=model_inputs["image_grid_thw"], image_token_id=int(getattr(qwen.model.config, "image_token_id", qwen.tokenizer.convert_tokens_to_ids("<|image_pad|>"))),
        layer_idx=23, tokenizer=qwen.tokenizer, owner_boxes=owner_boxes,
        image_width=raw.image.width, image_height=raw.image.height, max_new_tokens=args.max_new_tokens,
    )
    donor_eligibility = {
        "target": persistent_hard["owners"]["target"]["eligibility"],
        "competitor": persistent_hard["owners"]["competitor"]["eligibility"],
        "actual_top_tokens": {
            owner: persistent_hard["owners"][owner]["teacher_forced_hard"]["top_prediction_token_ids"]
            for owner in ("target", "competitor")
        },
        "parent_hard_reproduction": persistent_hard["parent_hard_reproduction"],
    }
    def realized_or_canonical(owner: str, fallback: Sequence[int]) -> list[int]:
        parsed = persistent_hard["owners"][owner]["hard_generated_path"].get("parsed", {})
        return list(parsed["row_token_ids"]) if parsed.get("valid") else [int(value) for value in fallback]

    release_rows = {
        "vase": realized_or_canonical("target", target_row),
        "clock": realized_or_canonical("competitor", competitor_row),
    }
    for layer_idx in args.layers:
        layer_result: dict[str, Any] = {"layer_idx": int(layer_idx), "boundaries": {}}
        for role in args.roles:
            boundary = _run_layer_boundary(
                model=qwen.model, model_inputs=model_inputs, features=features, grid_thw=grid_thw,
                merge_size=int(qwen.processor_identity.merge_size), prompt_ids=prompt_ids,
                recipient_row=target_row, donor_rows={"vase": target_row, "clock": competitor_row},
                parent_masks={
                    "vase": parent_masks["target"],
                    "clock": parent_masks["competitor"],
                }, image_grid_thw=model_inputs["image_grid_thw"], image_token_id=int(getattr(qwen.model.config, "image_token_id", qwen.tokenizer.convert_tokens_to_ids("<|image_pad|>"))), layer_idx=int(layer_idx), role=role,
                max_new_tokens=args.max_new_tokens, eos_token_id=getattr(qwen.tokenizer, "eos_token_id", None),
                release_rows=release_rows,
                tokenizer=qwen.tokenizer, owner_boxes=owner_boxes,
                image_width=raw.image.width, image_height=raw.image.height,
            )
            live_parent = {
                "target": boundary["arms"]["unrestricted"],
                "competitor": persistent_hard["owners"]["competitor"]["teacher_forced_unrestricted"],
            }
            parent_reproduction = assess_parent_reproduction(live=live_parent, frozen=unrestricted)
            parent_hard_reproduction = persistent_hard["parent_hard_reproduction"]
            clock_eligibility = donor_eligibility["competitor"]
            clock_path = boundary["realized_paths"]["clock"]
            clock_cached = boundary["cached_paths"]["clock_replacement"]
            if clock_eligibility.get("passed") and clock_eligibility.get("persistent_release") is not None:
                semantic_gate = assess_portability_release(
                    persistent_release=float(clock_eligibility["persistent_release"]),
                    replacement_release=float(clock_path["release"].get("description", 0.0)),
                    no_op_drift=float(boundary["self_noop_max_abs_logprob_drift"]),
                    valid_continuation=bool(clock_cached.get("natural_closure")),
                    owner_path_match=bool(clock_cached.get("owner_description_match")),
                    path_phase="description",
                )
            else:
                semantic_gate = {
                    "passed": False,
                    "reason": "persistent_hard_donor_ineligible",
                    "donor_eligibility": clock_eligibility,
                }
            trust_passed = bool(
                parent_reproduction["passed"]
                and parent_hard_reproduction["passed"]
                and boundary["self_noop_passed"]
            )
            boundary["parent_reproduction"] = parent_reproduction
            boundary["parent_hard_reproduction"] = parent_hard_reproduction
            boundary["scientific_gate"] = {
                "passed": trust_passed,
                "parent_reproduction_passed": parent_reproduction["passed"],
                "parent_hard_reproduction_passed": parent_hard_reproduction["passed"],
                "self_state_noop_passed": boundary["self_noop_passed"],
                "layer_is_positive_seam": int(layer_idx) == 23,
            }
            semantic_gate["scientific_trust_passed"] = trust_passed
            boundary["semantic_portability_gate"] = semantic_gate
            boundary["decision"] = classify_portability_case(
                trust_gate_passed=trust_passed,
                semantic=semantic_gate,
                geometry={"passed": False},
                eligible_count=int(persistent_hard["eligible_count"]),
            )
            layer_result["boundaries"][role] = boundary
        results.append(layer_result)
    layer13_boundary = next(
        (item["boundaries"].get("first_description") for item in results if int(item["layer_idx"]) == 13),
        None,
    )
    layer23_boundary = next(
        (item["boundaries"].get("first_description") for item in results if int(item["layer_idx"]) == 23),
        None,
    )
    negative_control_passed = bool(
        isinstance(layer13_boundary, Mapping)
        and layer13_boundary.get("scientific_gate", {}).get("passed")
        and layer13_boundary.get("semantic_portability_gate", {}).get("passed")
    )
    negative_control_trust_passed = bool(
        isinstance(layer13_boundary, Mapping)
        and layer13_boundary.get("scientific_gate", {}).get("passed")
    )
    if negative_control_passed and isinstance(layer23_boundary, dict):
        layer23_boundary["decision"] = {
            "classification": "vetoed_by_negative_control_layer_13",
            "interpreted": False,
            "reason": "layer13_semantic_portability_gate_passed",
        }
    if (
        not isinstance(layer23_boundary, Mapping)
        or not layer23_boundary.get("scientific_gate", {}).get("passed")
        or not negative_control_trust_passed
    ):
        panel_decision = {"classification": "invalid_execution_trust_gate", "interpreted": False}
    elif negative_control_passed:
        panel_decision = {
            "classification": "vetoed_by_negative_control_layer_13",
            "interpreted": False,
        }
    else:
        panel_decision = {
            "classification": str(layer23_boundary["decision"]["classification"]),
            "interpreted": bool(layer23_boundary["decision"].get("interpreted", False)),
        }
    panel_decision.update({
        "positive_layer": 23,
        "negative_control_layer": 13,
        "negative_control_trust_passed": negative_control_trust_passed,
        "negative_control_passed": negative_control_passed,
    })
    return {
        "schema_version": "fixed_encoding_downstream_residual_state_portability_gate.v1",
        "unit_id": UNIT_ID,
        "model_dtype": "torch.float32",
        "request_identity": build_request_identity(),
        "parent_receipts": {"all_query": str(Path(args.all_query_receipt).resolve()), "all_query_sha256": all_sha, "row_query": str(Path(args.row_query_receipt).resolve()), "row_query_sha256": row_sha},
        "parent_arm_mapping": mapping["mapping"],
        "donor_eligibility": donor_eligibility,
        "results": results,
        "runtime_contract": {
            "recipient_mask": "unrestricted", "donor_cache_discarded": True,
            "recipient_use_cache": True, "teacher_forced_use_cache": False,
            "logits_dtype": "torch.float32", "repetition_penalty": 1.0, "logits_processor": None,
            "max_new_tokens": int(args.max_new_tokens),
            "actual_model_parameter_dtype": str(next(qwen.model.parameters()).dtype),
            "actual_model_parameter_dtypes": sorted({str(parameter.dtype) for parameter in qwen.model.parameters()}),
            "explicit_mrope_prefill_shape": "[3,1,S]",
            "explicit_mrope_next_token_shape": "[3,1,1]",
            "prepare_inputs_for_generation_bypassed": True,
            "rope_deltas_not_passed": True,
            "attention_implementation": str(
                getattr(qwen.model.config, "_attn_implementation", None)
                or getattr(qwen.model.config, "attn_implementation", None)
                or "unknown"
            ),
        },
        "persistent_hard_paths": persistent_hard,
        "panel_decision": panel_decision,
    }


def main(argv: Sequence[str] | None = None) -> int:
    normalized = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    payload = run(args)
    payload["runner_sha256"] = sha256_file(Path(__file__))
    payload["normalized_argv"] = normalized
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "receipt.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
