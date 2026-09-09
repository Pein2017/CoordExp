#!/usr/bin/env python3
"""Final fixed radius-versus-direction residual-state factorial."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
CAUSAL_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-08-logit-lens-causal-transfer"
)
STAGE_A_ROOT = CAUSAL_ROOT / "run-v1"
STAGE_A_ACCEPTED = STAGE_A_ROOT / "evaluation-v2/receipt.json"
STAGE_A_ACCEPTED_SHA256 = "082ffdb2ba027869168dac268764559f90e156ed5ed5a84ee94560781d167db1"
STAGE_B_ROOT = CAUSAL_ROOT / "stage-b-v1"
STAGE_B_ACCEPTED = STAGE_B_ROOT / "receipt.json"
STAGE_B_ACCEPTED_SHA256 = "974980e16baaa45fd292c503449ca8676f84bd2e3436f0d42a90ff743b44231a"
IMAGE2299_TRAJECTORY = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-08-image2299-logit-lens/run-v2/trajectory-overfit.json"
)
IMAGE2299_TRAJECTORY_SHA256 = "537079135d3b12a3bfd72778ea63352112cde60775405e702abbbfa71efce063"
DEFAULT_OUTPUT = Path("/tmp/coordexp-logit-lens-radius-new")

IMAGE_IDS = (2299, *tuple(int(value) for value in (1584, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228)))
BLOCKS = (24, 27)
CAPTURE_BLOCKS = (24, 27, 28)
SCOPES = ("radius_only", "direction_only", "full_current")
ATOL = 2e-4
RTOL = 2e-4
GPU_BUDGET_SECONDS = 20 * 60
MAX_DEVICE_BYTES = 48 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


from probes.logit_lens import causal, base as parent

PARENT_HELPER = Path(parent.__file__)
PARENT_HELPER_SHA256 = parent.sha256_file(PARENT_HELPER)
CAUSAL_HELPER = Path(causal.__file__)
CAUSAL_HELPER_SHA256 = parent.sha256_file(CAUSAL_HELPER)


def atomic_json(path: Path, value: Any) -> None:
    parent.atomic_json(path, value)


def append_jsonl(handle: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    causal._jsonl_append(handle, rows)


def radius_direction_corners(
    recipient: torch.Tensor, donor: torch.Tensor
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return the three frozen non-native corners for one or many rows."""

    recipient = recipient.detach().to(device="cpu", dtype=torch.float32).contiguous()
    donor = donor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    require(recipient.shape == donor.shape and recipient.ndim in {1, 2}, "state shape mismatch")
    was_vector = recipient.ndim == 1
    if was_vector:
        recipient = recipient.unsqueeze(0)
        donor = donor.unsqueeze(0)
    r_recipient = recipient.norm(dim=-1, keepdim=True)
    r_donor = donor.norm(dim=-1, keepdim=True)
    require(torch.isfinite(r_recipient).all().item(), "nonfinite recipient radius")
    require(torch.isfinite(r_donor).all().item(), "nonfinite donor radius")
    require(bool((r_recipient > 0).all()), "zero recipient radius")
    require(bool((r_donor > 0).all()), "zero donor radius")
    u_recipient = recipient / r_recipient
    u_donor = donor / r_donor
    corners = {
        "radius_only": r_donor * u_recipient,
        "direction_only": r_recipient * u_donor,
        "full_current": r_donor * u_donor,
    }
    if was_vector:
        corners = {key: value[0] for key, value in corners.items()}
    geometry = {
        "recipient_radius": [float(value) for value in r_recipient[:, 0].tolist()],
        "donor_radius": [float(value) for value in r_donor[:, 0].tolist()],
        "donor_to_recipient_radius_ratio": [
            float(value) for value in (r_donor[:, 0] / r_recipient[:, 0]).tolist()
        ],
        "unit_direction_cosine": [
            float(value) for value in (u_recipient * u_donor).sum(dim=-1).tolist()
        ],
    }
    return corners, geometry


def validate_corner(
    state: torch.Tensor,
    *,
    expected_radius: torch.Tensor | float,
    expected_unit: torch.Tensor,
    scope: str,
) -> dict[str, Any]:
    """Fail closed unless both intended radius and unit direction are preserved."""

    state = state.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    expected_unit = expected_unit.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    radius = state.norm()
    require(torch.isfinite(radius).item() and float(radius) > 0.0, f"{scope} has zero/nonfinite radius")
    target_radius = torch.as_tensor(expected_radius, dtype=torch.float32).reshape(())
    unit = state / radius
    radius_passed = bool(torch.allclose(radius, target_radius, atol=ATOL, rtol=RTOL))
    direction_passed = bool(torch.allclose(unit, expected_unit, atol=ATOL, rtol=RTOL))
    require(radius_passed, f"{scope} radius mismatch")
    require(direction_passed, f"{scope} unit direction mismatch")
    return {
        "scope": scope,
        "actual_radius": float(radius.item()),
        "expected_radius": float(target_radius.item()),
        "radius_absolute_difference": float(abs(radius.item() - target_radius.item())),
        "unit_direction_max_absolute_difference": float((unit - expected_unit).abs().max().item()),
        "radius_passed": radius_passed,
        "unit_direction_passed": direction_passed,
        "atol": ATOL,
        "rtol": RTOL,
    }


def project_norm_head(norm: Any, head: Any, state: torch.Tensor) -> torch.Tensor:
    """Apply a final norm/head without a decoder forward."""

    device = next(norm.parameters()).device
    value = state.detach().to(device=device, dtype=next(norm.parameters()).dtype)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    with torch.inference_mode():
        logits = head(norm(value))
    return logits.detach().to(device="cpu", dtype=torch.float32).contiguous()


def project_final_state(bundle: Any, state: torch.Tensor) -> torch.Tensor:
    """Apply only the recipient's shared final norm/head to block-28 states."""

    _layers, norm, head, _seam = parent.resolve_text_stack(bundle.model)
    return project_norm_head(norm, head, state)


def _artifact_manifest(root: Path, *, exclude_receipts: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if exclude_receipts and path.name in {"receipt.json", "receipt.inprogress.json"}:
            continue
        result[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _prior_paths(image_id: int) -> dict[str, Path]:
    if image_id == 2299:
        return {
            "trajectory": IMAGE2299_TRAJECTORY,
            "baseline": STAGE_A_ROOT / "baseline.json",
            "compact": STAGE_A_ROOT / "compact-selected.pt",
            "trace": STAGE_A_ROOT / "patch-traces.jsonl",
        }
    image_root = STAGE_B_ROOT / "images" / f"image-{image_id:012d}"
    return {
        "trajectory": image_root / "trajectory-overfit.json",
        "baseline": image_root / "baseline.json",
        "compact": image_root / "compact-selected.pt",
        "trace": image_root / "patch-traces.jsonl",
    }


def _verify_prior_path_hashes(image_id: int, paths: Mapping[str, Path]) -> dict[str, Any]:
    if image_id == 2299:
        accepted = json.loads(STAGE_A_ACCEPTED.read_text())
        expected = accepted["raw_artifacts"]
        keys = {
            "baseline": "run-v1/baseline.json",
            "compact": "run-v1/compact-selected.pt",
            "trace": "run-v1/patch-traces.jsonl",
        }
        result = {}
        for role, key in keys.items():
            digest = sha256_file(paths[role])
            require(digest == expected[key]["sha256"], f"Stage A {role} hash mismatch")
            result[role] = {"path": str(paths[role]), "sha256": digest}
        digest = sha256_file(paths["trajectory"])
        require(digest == IMAGE2299_TRAJECTORY_SHA256, "Image2299 trajectory hash mismatch")
        result["trajectory"] = {"path": str(paths["trajectory"]), "sha256": digest}
        return result
    accepted = json.loads(STAGE_B_ACCEPTED.read_text())
    result = {}
    for role, path in paths.items():
        key = str(path.relative_to(STAGE_B_ROOT))
        expected = accepted["artifacts"][key]["sha256"]
        digest = sha256_file(path)
        require(digest == expected, f"Stage B {role} hash mismatch image={image_id}")
        result[role] = {"path": str(path), "sha256": digest}
    return result


def _load_prior(image_id: int) -> dict[str, Any]:
    paths = _prior_paths(image_id)
    identities = _verify_prior_path_hashes(image_id, paths)
    trajectory = json.loads(paths["trajectory"].read_text())
    baseline = json.loads(paths["baseline"].read_text())
    compact = torch.load(paths["compact"], map_location="cpu", weights_only=False)
    trace = [json.loads(line) for line in paths["trace"].read_text().splitlines()]
    sites = [dict(item) for item in baseline["sites"]]
    expected_count = 12 if image_id == 2299 else 4
    require(len(sites) == expected_count, f"prior site count changed image={image_id}")
    require(
        len({int(site["position"]) for site in sites}) == expected_count,
        f"prior positions are not unique image={image_id}",
    )
    prior_current: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in trace:
        if row["scope"] != "current" or int(row["block_1based"]) not in BLOCKS:
            continue
        if image_id != 2299:
            require(int(row["image_id"]) == image_id, "prior trace image identity mismatch")
        key = (str(row["direction"]), int(row["block_1based"]), int(row["site_index"]))
        require(key not in prior_current, f"duplicate prior current anchor {key}")
        prior_current[key] = row
    require(len(prior_current) == 2 * len(BLOCKS) * expected_count, "prior current anchor count changed")
    return {
        "image_id": image_id,
        "paths": paths,
        "identities": identities,
        "trajectory": trajectory,
        "baseline": baseline,
        "compact": compact,
        "sites": sites,
        "prior_current": prior_current,
    }


def _scalar_close(left: float, right: float) -> bool:
    return bool(
        torch.allclose(
            torch.tensor(float(left), dtype=torch.float64),
            torch.tensor(float(right), dtype=torch.float64),
            atol=ATOL,
            rtol=RTOL,
        )
    )


def _baseline_parity(
    *, prior: Mapping[str, Any], source: Any, overfit: Any, fresh_endpoints: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    prior_sites = prior["sites"]
    require(len(prior_sites) == len(fresh_endpoints), "fresh/prior endpoint count mismatch")
    identity_checks = []
    for old, new in zip(prior_sites, fresh_endpoints, strict=True):
        identity_checks.append(
            int(old["site_index"]) == int(new["site_index"])
            and int(old["position"]) == int(new["position"])
            and old["labels"] == new["labels"]
            and int(old["a_overfit_top1_token_id"]) == int(new["a_overfit_top1_token_id"])
            and int(old["s_source_top1_token_id"]) == int(new["s_source_top1_token_id"])
            and bool(old["eligible_for_R"]) == bool(new["eligible_for_R"])
            and old["exclusion_reason"] == new["exclusion_reason"]
        )
    prior_source = prior["compact"]["baseline_logits"]["source"].float()
    prior_overfit = prior["compact"]["baseline_logits"]["overfit"].float()
    source_max = float((source.baseline_logits - prior_source).abs().max().item())
    overfit_max = float((overfit.baseline_logits - prior_overfit).abs().max().item())
    source_passed = bool(torch.allclose(source.baseline_logits, prior_source, atol=ATOL, rtol=RTOL))
    overfit_passed = bool(torch.allclose(overfit.baseline_logits, prior_overfit, atol=ATOL, rtol=RTOL))
    require(all(identity_checks), "fresh endpoint identity differs from frozen prior")
    require(source_passed and overfit_passed, "fresh native full-vocab logits differ from prior")
    return {
        "endpoint_identity_passed": True,
        "source_full_vocab_passed": source_passed,
        "source_full_vocab_max_absolute_difference": source_max,
        "overfit_full_vocab_passed": overfit_passed,
        "overfit_full_vocab_max_absolute_difference": overfit_max,
        "atol": ATOL,
        "rtol": RTOL,
    }


def _corner_geometry(
    recipient_state: torch.Tensor,
    donor_state: torch.Tensor,
    corners: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rr = recipient_state.float().norm()
    rd = donor_state.float().norm()
    require(float(rr) > 0.0 and float(rd) > 0.0, "zero radius at intervention site")
    ur = recipient_state.float() / rr
    ud = donor_state.float() / rd
    checks = {
        "radius_only": validate_corner(
            corners["radius_only"], expected_radius=rd, expected_unit=ur, scope="radius_only"
        ),
        "direction_only": validate_corner(
            corners["direction_only"], expected_radius=rr, expected_unit=ud, scope="direction_only"
        ),
        "full_current": validate_corner(
            corners["full_current"], expected_radius=rd, expected_unit=ud, scope="full_current"
        ),
    }
    full_state_passed = bool(
        torch.allclose(corners["full_current"].float(), donor_state.float(), atol=ATOL, rtol=RTOL)
    )
    require(full_state_passed, "full-current formula does not reconstruct donor state")
    geometry = {
        "recipient_radius": float(rr.item()),
        "donor_radius": float(rd.item()),
        "donor_to_recipient_radius_ratio": float((rd / rr).item()),
        "unit_direction_cosine": float(torch.dot(ur, ud).item()),
        "full_state_matches_donor": full_state_passed,
    }
    return checks, geometry


def _block28_controls(
    *, image_id: int, source: Any, overfit: Any, endpoints: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    projection_calls = 0
    selected = torch.tensor([int(endpoint["position"]) for endpoint in endpoints], dtype=torch.long)
    for direction, donor, recipient in (
        ("overfit_to_source", overfit, source),
        ("source_to_overfit", source, overfit),
    ):
        recipient_states = recipient.states[28][0, selected, :]
        donor_states = donor.states[28][0, selected, :]
        corners, _geometry = radius_direction_corners(recipient_states, donor_states)
        radius_logits = project_final_state(recipient, corners["radius_only"])
        direction_logits = project_final_state(recipient, corners["direction_only"])
        projection_calls += 2
        for index, endpoint in enumerate(endpoints):
            radius_max = float((radius_logits[index] - recipient.baseline_logits[0, index]).abs().max().item())
            direction_max = float((direction_logits[index] - donor.baseline_logits[0, index]).abs().max().item())
            radius_passed = bool(
                torch.allclose(
                    radius_logits[index], recipient.baseline_logits[0, index], atol=ATOL, rtol=RTOL
                )
            )
            direction_passed = bool(
                torch.allclose(
                    direction_logits[index], donor.baseline_logits[0, index], atol=ATOL, rtol=RTOL
                )
            )
            require(radius_passed, "block28 radius-only did not preserve recipient logits")
            require(direction_passed, "block28 direction-only did not reproduce donor logits")
            records.append(
                {
                    "image_id": image_id,
                    "direction": direction,
                    "site_index": int(endpoint["site_index"]),
                    "position": int(endpoint["position"]),
                    "radius_only_matches_recipient_full_vocab": radius_passed,
                    "radius_only_max_absolute_difference": radius_max,
                    "direction_only_matches_donor_full_vocab": direction_passed,
                    "direction_only_max_absolute_difference": direction_max,
                    "atol": ATOL,
                    "rtol": RTOL,
                    "semantics": "direct_shared_final_norm_and_head_projection_no_decoder_forward",
                }
            )
    return records, projection_calls


def _reduce_image(rows: Sequence[Mapping[str, Any]], *, image_id: int) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for direction in ("overfit_to_source", "source_to_overfit"):
        for block in BLOCKS:
            for scope in SCOPES:
                selected = [
                    row
                    for row in rows
                    if row["direction"] == direction
                    and int(row["block_1based"]) == block
                    and row["scope"] == scope
                ]
                eligible = [row for row in selected if row["eligible_for_R"]]
                numerator = sum(float(row["patch_minus_receiver"]) for row in eligible)
                denominator = sum(float(row["donor_minus_receiver"]) for row in eligible)
                groups.append(
                    {
                        "image_id": image_id,
                        "direction": direction,
                        "block_1based": block,
                        "scope": scope,
                        "population_site_count": len(selected),
                        "eligible_site_count": len(eligible),
                        "raw_numerator_sum": numerator,
                        "raw_denominator_sum": denominator,
                        "R_unclipped": None if denominator == 0.0 else numerator / denominator,
                        "donor_top1_count": sum(row["top1_is_donor"] for row in eligible),
                        "third_token_top1_count": sum(row["top1_is_third_token"] for row in eligible),
                    }
                )
    interactions: list[dict[str, Any]] = []
    group_ids = sorted({str(row["factorial_group_id"]) for row in rows})
    for group_id in group_ids:
        group = [row for row in rows if row["factorial_group_id"] == group_id]
        require(len(group) == 3 and {row["scope"] for row in group} == set(SCOPES), "factorial group incomplete")
        by_scope = {row["scope"]: row for row in group}
        native = float(group[0]["m_receiver"])
        interaction = (
            float(by_scope["full_current"]["m_patch"])
            - float(by_scope["radius_only"]["m_patch"])
            - float(by_scope["direction_only"]["m_patch"])
            + native
        )
        interactions.append(
            {
                "factorial_group_id": group_id,
                "image_id": image_id,
                "direction": group[0]["direction"],
                "block_1based": group[0]["block_1based"],
                "site_index": group[0]["site_index"],
                "eligible_for_R": group[0]["eligible_for_R"],
                "interaction_raw_margin": interaction,
            }
        )
    return {
        "schema_version": "logit_lens_radius_direction_image_summary.v1",
        "image_id": image_id,
        "groups": groups,
        "interactions": interactions,
    }


def _run_image(
    *,
    image_id: int,
    output_root: Path,
    source_opened: Any,
    source_components: Any,
    source_frontend: Any,
    source_config: Any,
    overfit_opened: Any,
    overfit_components: Any,
    overfit_frontend: Any,
    overfit_config: Any,
    stage_started: float,
) -> dict[str, Any]:
    image_root = output_root / "images" / f"image-{image_id:012d}"
    image_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    progress: dict[str, Any] = {
        "schema_version": "logit_lens_radius_direction_image_receipt.v1",
        "status": "running",
        "image_id": image_id,
    }
    atomic_json(image_root / "receipt.inprogress.json", progress)
    counts = {
        "full_model_teacher_forced": 0,
        "decoder_direct_baseline_capture": 0,
        "self_patch": 0,
        "radius_only": 0,
        "direction_only": 0,
        "full_current": 0,
        "vision_encoder_forward_calls": 0,
        "block28_head_projection_calls": 0,
        "block28_projected_site_arms": 0,
        "trace_rows": 0,
    }
    try:
        require(time.perf_counter() - stage_started < GPU_BUDGET_SECONDS, "GPU budget exhausted before image")
        prior = _load_prior(image_id)
        trajectory = prior["trajectory"]
        source_request, source_native, source_prompt, source_input = causal.request_and_inputs_for_image(
            components=source_components,
            frontend=source_frontend,
            config=source_config,
            image_id=image_id,
            request_id=f"logit-lens-radius-direction-source-{image_id}",
        )
        overfit_request, overfit_native, overfit_prompt, overfit_input = causal.request_and_inputs_for_image(
            components=overfit_components,
            frontend=overfit_frontend,
            config=overfit_config,
            image_id=image_id,
            request_id=f"logit-lens-radius-direction-overfit-{image_id}",
        )
        require(source_input == overfit_input, "fresh source/overfit input identity mismatch")
        prior_input = prior["baseline"]["input"]
        require(source_input == prior_input, "fresh input identity differs from frozen prior")
        require(source_prompt == overfit_prompt, "fresh prompt token IDs differ across checkpoints")
        require(parent.sha256_json(source_prompt) == source_input["prompt_token_ids_sha256"], "prompt hash mismatch")
        del source_request, overfit_request
        atomic_json(
            image_root / "input.json",
            {
                "schema_version": "logit_lens_radius_direction_input.v1",
                "image_id": image_id,
                "fresh_source": source_input,
                "fresh_overfit": overfit_input,
                "prior": prior_input,
                "all_identical": True,
                "prior_artifacts": prior["identities"],
            },
        )
        selected_sites = prior["sites"]
        site_positions = [int(site["position"]) for site in selected_sites]
        if image_id == 2299:
            fresh_sites = causal.coordinate_sites(source_prompt, trajectory)
        else:
            fresh_sites, exclusion = causal.middle_coordinate_sites(source_prompt, trajectory)
            require(exclusion is None, f"frozen Stage B sites are no longer selectable: {exclusion}")
        require(
            [int(site["position"]) for site in fresh_sites] == site_positions,
            "fresh deterministic selector differs from frozen positions",
        )

        source_visual, source_visual_aliases = causal.resolve_visual_module(source_components.model)  # noqa: SLF001
        overfit_visual, overfit_visual_aliases = causal.resolve_visual_module(overfit_components.model)  # noqa: SLF001
        with causal.ModuleCallCounter(source_visual) as source_visual_counter:
            source = causal._capture_existing_session(
                name="source",
                opened=source_opened, components=source_components,
                native_inputs=source_native,
                prompt_ids=source_prompt,
                trajectory=trajectory,
                sites=fresh_sites,
                input_receipt=source_input,
                blocks=CAPTURE_BLOCKS,
            )
        with causal.ModuleCallCounter(overfit_visual) as overfit_visual_counter:
            overfit = causal._capture_existing_session(
                name="overfit",
                opened=overfit_opened, components=overfit_components,
                native_inputs=overfit_native,
                prompt_ids=overfit_prompt,
                trajectory=trajectory,
                sites=fresh_sites,
                input_receipt=overfit_input,
                blocks=CAPTURE_BLOCKS,
            )
        require(source_visual_counter.calls == 1 and overfit_visual_counter.calls == 1, "vision pass count changed")
        counts["full_model_teacher_forced"] = 2
        counts["decoder_direct_baseline_capture"] = 2
        counts["vision_encoder_forward_calls"] = 2
        fresh_endpoints = causal.build_endpoints(
            sites=fresh_sites,
            source=source,
            overfit=overfit,
            expected_site_count=len(fresh_sites),
        )
        for endpoint in fresh_endpoints:
            endpoint["image_id"] = image_id
        baseline_parity = _baseline_parity(
            prior=prior, source=source, overfit=overfit, fresh_endpoints=fresh_endpoints
        )
        baseline = {
            "schema_version": "logit_lens_radius_direction_baseline.v1",
            "image_id": image_id,
            "trajectory": {
                "path": str(prior["paths"]["trajectory"]),
                "file_sha256": prior["identities"]["trajectory"]["sha256"],
                "token_ids_sha256": trajectory["token_ids_sha256"],
                "token_count": trajectory["token_count"],
            },
            "input": source_input,
            "sites": fresh_endpoints,
            "fresh_vs_prior": baseline_parity,
            "checkpoint_checks": {
                "source": source.baseline_checks,
                "overfit": overfit.baseline_checks,
            },
            "visual_module_aliases": {
                "source": source_visual_aliases,
                "overfit": overfit_visual_aliases,
            },
        }
        atomic_json(image_root / "baseline.json", baseline)
        selected = torch.tensor(site_positions, dtype=torch.long)
        compact = {
            "schema_version": "logit_lens_radius_direction_compact_tensors.v1",
            "image_id": image_id,
            "blocks_1based": list(CAPTURE_BLOCKS),
            "sites": fresh_endpoints,
            "baseline_logits": {
                "source": source.baseline_logits,
                "overfit": overfit.baseline_logits,
            },
            "selected_residuals": {
                bundle.name: torch.stack([bundle.states[block][0, selected, :] for block in CAPTURE_BLOCKS])
                for bundle in (source, overfit)
            },
        }
        torch.save(compact, image_root / "compact-selected.pt")
        progress["status"] = "baseline_complete"
        progress["counts"] = dict(counts)
        atomic_json(image_root / "receipt.inprogress.json", progress)

        self_checks = [
            *causal._self_patch_checks_for_blocks(source, selected, blocks=(28,)),
            *causal._self_patch_checks_for_blocks(overfit, selected, blocks=(28,)),
        ]
        for check in self_checks:
            check["image_id"] = image_id
        counts["self_patch"] = len(self_checks)
        block28_controls, projection_calls = _block28_controls(
            image_id=image_id, source=source, overfit=overfit, endpoints=fresh_endpoints
        )
        counts["block28_head_projection_calls"] = projection_calls
        counts["block28_projected_site_arms"] = len(block28_controls) * 2

        trace_rows: list[dict[str, Any]] = []
        prior_anchor_checks: list[dict[str, Any]] = []
        with (image_root / "trace.jsonl").open("x") as trace_handle:
            for direction, donor, recipient in (
                ("overfit_to_source", overfit, source),
                ("source_to_overfit", source, overfit),
            ):
                donor_top1_key = (
                    "a_overfit_top1_token_id" if direction == "overfit_to_source" else "s_source_top1_token_id"
                )
                for block in BLOCKS:
                    for endpoint in fresh_endpoints:
                        require(
                            time.perf_counter() - stage_started < GPU_BUDGET_SECONDS,
                            "GPU budget exhausted before factorial site",
                        )
                        index = int(endpoint["site_index"])
                        position = int(endpoint["position"])
                        recipient_state = recipient.states[block][0, position, :]
                        donor_state = donor.states[block][0, position, :]
                        corners, _ = radius_direction_corners(recipient_state, donor_state)
                        geometry_checks, geometry = _corner_geometry(recipient_state, donor_state, corners)
                        site_rows: list[dict[str, Any]] = []
                        for scope in SCOPES:
                            logits, patch_receipt = causal._run_patch(
                                recipient=recipient,
                                block=block,
                                positions=[position],
                                replacement=corners[scope].unsqueeze(0),
                                expected_before=recipient_state.unsqueeze(0),
                                selected=selected,
                            )
                            counts[scope] += 1
                            earlier = selected < position
                            earlier_exact = bool(
                                torch.equal(logits[:, earlier, :], recipient.baseline_logits[:, earlier, :])
                            )
                            require(earlier_exact, "future-position factorial patch contaminated earlier logits")
                            patch_receipt = {
                                **patch_receipt,
                                "all_earlier_selected_logits_exact": earlier_exact,
                                "later_selected_logits_are_not_used_by_this_site_isolated_arm": True,
                            }
                            row = causal._row_for_patch(
                                direction=direction,
                                donor=donor,
                                recipient=recipient,
                                endpoint=endpoint,
                                block=block,
                                scope=scope,
                                patch_vector=logits[0, index],
                                patch_receipt=patch_receipt,
                                random_receipt=None,
                            )
                            top1 = int(row["patch_endpoint"]["top1_token_id"])
                            a_id = int(endpoint["a_overfit_top1_token_id"])
                            s_id = int(endpoint["s_source_top1_token_id"])
                            row.update(
                                {
                                    "schema_version": "logit_lens_radius_direction_trace.v1",
                                    "image_id": image_id,
                                    "factorial_group_id": f"{image_id}:{direction}:{block}:{index}",
                                    "corner_geometry": geometry,
                                    "corner_validation": geometry_checks[scope],
                                    "donor_top1_token_id": int(endpoint[donor_top1_key]),
                                    "top1_is_donor": top1 == int(endpoint[donor_top1_key]),
                                    "top1_is_third_token": top1 not in {a_id, s_id},
                                }
                            )
                            site_rows.append(row)
                        by_scope = {row["scope"]: row for row in site_rows}
                        interaction = (
                            float(by_scope["full_current"]["m_patch"])
                            - float(by_scope["radius_only"]["m_patch"])
                            - float(by_scope["direction_only"]["m_patch"])
                            + float(by_scope["full_current"]["m_receiver"])
                        )
                        by_scope["full_current"]["factorial_interaction_raw_margin"] = interaction
                        prior_row = prior["prior_current"][(direction, block, index)]
                        full_row = by_scope["full_current"]
                        comparisons = {
                            "m_receiver": _scalar_close(full_row["m_receiver"], prior_row["m_receiver"]),
                            "m_donor": _scalar_close(full_row["m_donor"], prior_row["m_donor"]),
                            "m_patch": _scalar_close(full_row["m_patch"], prior_row["m_patch"]),
                            "a_raw_logit": _scalar_close(
                                full_row["patch_endpoint"]["a_raw_logit"],
                                prior_row["patch_endpoint"]["a_raw_logit"],
                            ),
                            "s_raw_logit": _scalar_close(
                                full_row["patch_endpoint"]["s_raw_logit"],
                                prior_row["patch_endpoint"]["s_raw_logit"],
                            ),
                            "top1_token_id": int(full_row["patch_endpoint"]["top1_token_id"])
                            == int(prior_row["patch_endpoint"]["top1_token_id"]),
                        }
                        prior_passed = all(comparisons.values())
                        require(prior_passed, f"fresh full-current differs from prior anchor {direction}/{block}/{index}")
                        anchor = {
                            "image_id": image_id,
                            "direction": direction,
                            "block_1based": block,
                            "site_index": index,
                            "passed": prior_passed,
                            "comparisons": comparisons,
                            "prior_m_patch": prior_row["m_patch"],
                            "fresh_m_patch": full_row["m_patch"],
                            "absolute_m_patch_difference": abs(
                                float(full_row["m_patch"]) - float(prior_row["m_patch"])
                            ),
                            "atol": ATOL,
                            "rtol": RTOL,
                        }
                        full_row["prior_full_current_anchor"] = anchor
                        prior_anchor_checks.append(anchor)
                        trace_rows.extend(site_rows)
                        append_jsonl(trace_handle, site_rows)
                    progress["status"] = f"patch_complete_{direction}_block_{block}"
                    counts["trace_rows"] = len(trace_rows)
                    progress["counts"] = dict(counts)
                    atomic_json(image_root / "receipt.inprogress.json", progress)

        expected_trace = len(fresh_endpoints) * 2 * len(BLOCKS) * len(SCOPES)
        require(len(trace_rows) == expected_trace, "factorial trace count changed")
        counts["trace_rows"] = len(trace_rows)
        summary = _reduce_image(trace_rows, image_id=image_id)
        atomic_json(image_root / "summary.json", summary)
        checks = {
            "schema_version": "logit_lens_radius_direction_checks.v1",
            "image_id": image_id,
            "checks": {
                "input_matches_prior": True,
                "selector_matches_prior": True,
                "native_full_vocab_matches_prior": all(
                    value
                    for key, value in baseline_parity.items()
                    if key.endswith("passed")
                ),
                "native_vs_text_replay_parity": all(
                    bundle.baseline_checks["native_hooks_off_vs_direct_capture_passed"]
                    for bundle in (source, overfit)
                ),
                "self_patch_exact": all(check["selected_logits_exact"] for check in self_checks),
                "block28_radius_and_direction_controls": all(
                    item["radius_only_matches_recipient_full_vocab"]
                    and item["direction_only_matches_donor_full_vocab"]
                    for item in block28_controls
                ),
                "all_corner_radius_direction_checks": all(
                    row["corner_validation"]["radius_passed"]
                    and row["corner_validation"]["unit_direction_passed"]
                    for row in trace_rows
                ),
                "all_recipient_states_exact_before_patch": all(
                    row["residual_patch"]["before_exact"] for row in trace_rows
                ),
                "all_non_target_residuals_exact": all(
                    row["residual_patch"]["non_target_residual_exact"] for row in trace_rows
                ),
                "all_earlier_selected_logits_exact": all(
                    row["residual_patch"]["all_earlier_selected_logits_exact"] for row in trace_rows
                ),
                "all_prior_full_current_anchors_match": all(item["passed"] for item in prior_anchor_checks),
            },
            "baseline_parity": baseline_parity,
            "self_patch_checks": self_checks,
            "block28_projection_controls": block28_controls,
            "prior_full_current_anchors": prior_anchor_checks,
        }
        require(all(checks["checks"].values()), f"image checks failed {image_id}")
        atomic_json(image_root / "checks.json", checks)
        cold_rows = [json.loads(line) for line in (image_root / "trace.jsonl").read_text().splitlines()]
        require(cold_rows == trace_rows, "cold trace readback mismatch")
        cold_compact = torch.load(image_root / "compact-selected.pt", map_location="cpu", weights_only=False)
        require(cold_compact["image_id"] == image_id, "cold compact readback mismatch")
        teacher_forced_total = (
            counts["full_model_teacher_forced"]
            + counts["decoder_direct_baseline_capture"]
            + counts["self_patch"]
            + counts["radius_only"]
            + counts["direction_only"]
            + counts["full_current"]
        )
        artifacts = _artifact_manifest(image_root)
        receipt = {
            **progress,
            "status": "mechanics_candidate",
            "elapsed_seconds": time.perf_counter() - started,
            "prior_artifacts": prior["identities"],
            "trajectory_token_ids_sha256": trajectory["token_ids_sha256"],
            "input": source_input,
            "counts": {
                **counts,
                "teacher_forced_decoder_forward_total": teacher_forced_total,
                "population_sites": len(fresh_endpoints),
                "eligible_sites": sum(bool(item["eligible_for_R"]) for item in fresh_endpoints),
                "prior_anchor_checks": len(prior_anchor_checks),
                "self_patch_checks": len(self_checks),
            },
            "checks": checks["checks"],
            "checks_path": str(image_root / "checks.json"),
            "checks_sha256": sha256_file(image_root / "checks.json"),
            "artifacts": artifacts,
            "resource": {
                "peak_cuda_allocated_bytes_stage_so_far": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes_stage_so_far": int(torch.cuda.max_memory_reserved()),
                "peak_host_rss_kib_stage_so_far": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "artifact_payload_bytes_excluding_receipt": sum(item["bytes"] for item in artifacts.values()),
            },
            "claim_boundary": {
                "fixed_training_image_prefix_and_sites": True,
                "current_position_factorial_only": True,
                "block28_is_algebraic_control_only": True,
                "no_accuracy_owner_recovery_or_natural_generation_claim": True,
            },
        }
        atomic_json(image_root / "receipt.json", receipt)
        (image_root / "receipt.inprogress.json").unlink()
        return receipt
    except BaseException as error:
        failure = {
            **progress,
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        atomic_json(image_root / "receipt.failed.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started_unix = time.time()
    started = time.perf_counter()
    runner_hash = sha256_file(Path(__file__))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    progress: dict[str, Any] = {
        "schema_version": "logit_lens_radius_direction_receipt.v1",
        "status": "running",
        "started_unix": started_unix,
        "output_root": str(output),
        "image_ids": list(IMAGE_IDS),
        "runner_sha256_at_launch": runner_hash,
        "contract": {
            "blocks_1based": list(BLOCKS),
            "capture_blocks_1based": list(CAPTURE_BLOCKS),
            "scopes": list(SCOPES),
            "population_images": 13,
            "population_sites": 60,
            "prior_eligible_sites": 59,
            "atol": ATOL,
            "rtol": RTOL,
            "gpu_budget_seconds": GPU_BUDGET_SECONDS,
            "no_new_generation": True,
            "no_kv_cache": True,
        },
    }
    atomic_json(output / "receipt.inprogress.json", progress)
    source_opened: Any | None = None
    overfit_opened: Any | None = None
    try:
        require(torch.cuda.is_available(), "CUDA is required")
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "GPU-8d43cb78-19ca-2f59-3179-7ea166cb1a4e"},
            "not bound to physical GPU0",
        )
        require(sha256_file(PARENT_HELPER) == PARENT_HELPER_SHA256, "parent helper hash mismatch")
        require(sha256_file(CAUSAL_HELPER) == CAUSAL_HELPER_SHA256, "causal helper hash mismatch")
        require(sha256_file(STAGE_A_ACCEPTED) == STAGE_A_ACCEPTED_SHA256, "Stage A receipt hash mismatch")
        require(sha256_file(STAGE_B_ACCEPTED) == STAGE_B_ACCEPTED_SHA256, "Stage B receipt hash mismatch")
        source_gate_root, source_gate = parent._stage_source_gate(output)
        source_opened, source_frontend, source_config, source_resolved, source_components = parent._open_session(
            parent.SOURCE_ADAPTER, source_gate_root=source_gate_root
        )
        overfit_opened, overfit_frontend, overfit_config, overfit_resolved, overfit_components = parent._open_session(
            parent.OVERFIT_ADAPTER, source_gate_root=source_gate_root
        )
        require(source_resolved.fingerprint == overfit_resolved.fingerprint, "config drift across sessions")
        resident_after_two = int(torch.cuda.memory_allocated())
        require(resident_after_two < MAX_DEVICE_BYTES, "two-model allocation exceeds 48 GiB")
        receipts: list[dict[str, Any]] = []
        for image_id in IMAGE_IDS:
            receipt = _run_image(
                image_id=image_id,
                output_root=output,
                source_opened=source_opened,
            source_components=source_components,
                source_frontend=source_frontend,
                source_config=source_config,
                overfit_opened=overfit_opened,
            overfit_components=overfit_components,
                overfit_frontend=overfit_frontend,
                overfit_config=overfit_config,
                stage_started=started,
            )
            receipts.append(receipt)
            progress["status"] = f"image_{image_id}_complete"
            progress["completed_image_ids"] = [int(item["image_id"]) for item in receipts]
            atomic_json(output / "receipt.inprogress.json", progress)
        elapsed = time.perf_counter() - started
        require(elapsed <= GPU_BUDGET_SECONDS, f"GPU budget exceeded: {elapsed:.1f}s")
        aggregate: dict[str, int] = {}
        for receipt in receipts:
            for key, value in receipt["counts"].items():
                if isinstance(value, int):
                    aggregate[key] = aggregate.get(key, 0) + value
        artifacts = _artifact_manifest(output)
        top_receipt = {
            **progress,
            "status": "mechanics_candidate",
            "completed_unix": time.time(),
            "elapsed_seconds": elapsed,
            "identity": {
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256_at_launch": runner_hash,
                "runner_sha256_at_completion": sha256_file(Path(__file__)),
                "causal_helper_path": str(CAUSAL_HELPER),
                "causal_helper_sha256": CAUSAL_HELPER_SHA256,
                "helper_binding_scope": "maintained_package_sources_at_launch",
                "parent_helper_path": str(PARENT_HELPER),
                "parent_helper_sha256": PARENT_HELPER_SHA256,
                "stage_a_accepted_receipt": str(STAGE_A_ACCEPTED),
                "stage_a_accepted_receipt_sha256": STAGE_A_ACCEPTED_SHA256,
                "stage_b_accepted_receipt": str(STAGE_B_ACCEPTED),
                "stage_b_accepted_receipt_sha256": STAGE_B_ACCEPTED_SHA256,
                "source_adapter": {"path": str(parent.SOURCE_ADAPTER), "sha256": parent.SOURCE_ADAPTER_SHA256},
                "overfit_adapter": {"path": str(parent.OVERFIT_ADAPTER), "sha256": parent.OVERFIT_ADAPTER_SHA256},
                "embedding_delta": {"path": str(parent.SOURCE_DELTA), "sha256": parent.SOURCE_DELTA_SHA256},
                "source_gate": source_gate,
            },
            "runtime": parent._runtime_identity(),
            "sessions": {
                "source": source_opened.receipt.to_artifact_dict(),
                "overfit": overfit_opened.receipt.to_artifact_dict(),
            },
            "counts": {
                **aggregate,
                "images": len(receipts),
                "resident_models": 2,
            },
            "image_receipts": [
                {
                    "image_id": int(receipt["image_id"]),
                    "path": str(
                        output / "images" / f"image-{int(receipt['image_id']):012d}" / "receipt.json"
                    ),
                    "sha256": sha256_file(
                        output / "images" / f"image-{int(receipt['image_id']):012d}" / "receipt.json"
                    ),
                }
                for receipt in receipts
            ],
            "checks": {
                "runner_unchanged": sha256_file(Path(__file__)) == runner_hash,
                "all_13_images_complete": len(receipts) == len(IMAGE_IDS),
                "image_ids_exact": tuple(int(receipt["image_id"]) for receipt in receipts) == IMAGE_IDS,
                "all_image_checks_passed": all(all(receipt["checks"].values()) for receipt in receipts),
                "population_sites_exact": aggregate["population_sites"] == 60,
                "eligible_sites_exact": aggregate["eligible_sites"] == 59,
                "trace_rows_exact": aggregate["trace_rows"] == 720,
                "no_generation": True,
                "gpu_budget_passed": elapsed <= GPU_BUDGET_SECONDS,
                "device_allocation_cap_passed": int(torch.cuda.max_memory_allocated()) < MAX_DEVICE_BYTES,
            },
            "resource": {
                "two_model_resident_allocated_bytes": resident_after_two,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "artifact_payload_bytes_excluding_receipt": sum(item["bytes"] for item in artifacts.values()),
            },
            "artifacts": artifacts,
            "claim_boundary": {
                "fixed_13_training_image_prefixes_and_60_sites": True,
                "radius_direction_current_state_factorial_only": True,
                "block28_controls_are_algebraic_only": True,
                "no_behavior_accuracy_or_natural_generation_claim": True,
            },
            "stop": "final_round_candidate_complete_no_successor",
        }
        require(all(top_receipt["checks"].values()), "terminal checks failed")
        atomic_json(output / "receipt.json", top_receipt)
        (output / "receipt.inprogress.json").unlink()
        print(
            json.dumps(
                {
                    "status": top_receipt["status"],
                    "output_root": str(output),
                    "elapsed_seconds": elapsed,
                    "images": len(receipts),
                    "sites": aggregate["population_sites"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        failure = {
            **progress,
            "status": "failed",
            "failed_unix": time.time(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "resource": {
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
        }
        atomic_json(output / "receipt.failed.json", failure)
        raise
    finally:
        for opened in (overfit_opened, source_opened):
            if opened is not None:
                try:
                    opened.close()
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
