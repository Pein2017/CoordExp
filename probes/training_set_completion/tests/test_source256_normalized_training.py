from __future__ import annotations

from collections import Counter
import json

import pytest
import torch

from probes.training_set_completion import source256_normalized_training as normalized
from probes.training_set_completion import source256_training as predecessor


EOS = 99
HINGE = {
    "coordinate_token_ids": list(range(128)),
    "coordinate_bin_values": list(range(128)),
    "margin": 1 / 999,
}


def _route(*, kind: str, image_id: int = 0) -> dict:
    canonical_suffix = [40, 41, 42, 43, 44, 45, 46, 47, 48, EOS]
    if kind == "canonical":
        continuation = canonical_suffix
        weights = [1] * len(continuation)
        prefix = []
        prefix_owners: list[str] = []
        suffix_owners = [f"{image_id}:owner-{index}" for index in range(2)]
        route_kind = "canonical"
    else:
        prefix = [30, 31]
        completion_suffix = [40, 41, 42, 43, EOS]
        continuation = prefix + completion_suffix
        weights = [0, 0] + [1] * 5
        prefix_owners = [f"{image_id}:owner-0"]
        suffix_owners = [f"{image_id}:owner-1"]
        route_kind = "fixed_source_prefix_completion"
    bank = prefix_owners + suffix_owners
    return {
        "route_id": f"route-{image_id}-{kind}",
        "image_id": image_id,
        "example_id": str(image_id),
        "case": {"image_path": f"/images/{image_id}.jpg"},
        "image_identity": {"image_path": f"/images/{image_id}.jpg"},
        "prompt_token_ids": [10, 11],
        "continuation_token_ids": continuation,
        "ce_weights": weights,
        "trusted_boxes": [],
        "provenance": {
            "route_kind": route_kind,
            "bank_owner_ids": bank,
            "prefix_owner_ids": prefix_owners,
            "suffix_owner_ids": suffix_owners,
            "prefix_token_ids": prefix,
            "suffix_token_ids": continuation[len(prefix) :],
            "source_greedy_generated_token_ids_sha256": "a" * 64,
        },
    }


def _terms(route: dict, *, canonical_count: int | None = None):
    logits = torch.randn(
        len(route["continuation_token_ids"]), 128, dtype=torch.float64, requires_grad=True
    )
    return normalized._route_terms(
        logits, route, HINGE, canonical_full_target_token_count=canonical_count
    )


def test_canonical_ce_is_unchanged_and_completion_uses_same_image_canonical_count() -> None:
    canonical = _route(kind="canonical")
    completion = _route(kind="completion")
    canonical_logits = torch.zeros(
        len(canonical["continuation_token_ids"]), 128, dtype=torch.float64, requires_grad=True
    )
    completion_logits = torch.zeros(
        len(completion["continuation_token_ids"]), 128, dtype=torch.float64, requires_grad=True
    )

    old_canonical, old_metrics = normalized.training.masked_ce_loss(
        canonical_logits,
        torch.tensor(canonical["continuation_token_ids"]),
        canonical["ce_weights"],
    )
    new_canonical, _, canonical_active, canonical_card = normalized._route_terms(
        canonical_logits, canonical, HINGE
    )
    new_completion, _, completion_active, completion_card = normalized._route_terms(
        completion_logits,
        completion,
        HINGE,
        canonical_full_target_token_count=len(canonical["continuation_token_ids"]),
    )

    torch.testing.assert_close(new_canonical, old_canonical)
    assert canonical_active == old_metrics["active_tokens"] == 10
    assert canonical_card["old_ce_denominator_active_tokens"] == 10
    assert canonical_card["new_ce_denominator_canonical_full_target_tokens"] == 10
    assert completion_active == 5
    assert completion_card["ce_numerator"] == pytest.approx(5 * torch.log(torch.tensor(128.0)).item())
    assert completion_card["old_ce_denominator_active_tokens"] == 5
    assert completion_card["new_ce_denominator_canonical_full_target_tokens"] == 10
    assert completion_card["ce_scale_active_over_canonical"] == pytest.approx(0.5)
    torch.testing.assert_close(
        new_completion,
        torch.tensor(completion_card["active_token_mean_ce"] * 0.5, dtype=new_completion.dtype),
    )


def test_completion_masks_remain_binary_and_prefix_context_stays_differentiable() -> None:
    route = _route(kind="completion")
    bad = dict(route)
    bad["ce_weights"] = [0.5, 0.5] + [1] * 5
    with pytest.raises(ValueError, match="binary 0/1"):
        _terms(bad, canonical_count=10)

    prefix_signal = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    vocab = torch.arange(128, dtype=torch.float64).view(1, -1)
    logits = prefix_signal * vocab.expand(len(route["continuation_token_ids"]), -1)
    ce, _, active, card = normalized._route_terms(
        logits, route, HINGE, canonical_full_target_token_count=10
    )
    assert active == 5
    assert card["normalized_canonical_target_mean_ce"] == pytest.approx(float(ce.detach()))
    ce.backward()
    assert prefix_signal.grad is not None and bool(torch.isfinite(prefix_signal.grad))
    assert prefix_signal.grad.abs() > 0


def test_geometry_is_not_scaled_with_completion_ce() -> None:
    route = _route(kind="completion")
    sentinel = torch.tensor(7.0, dtype=torch.float64)
    original = normalized.training.raw_axis_validity_hinge
    normalized.training.raw_axis_validity_hinge = lambda *args, **kwargs: sentinel
    try:
        ce, hinge, _, card = _terms(route, canonical_count=10)
    finally:
        normalized.training.raw_axis_validity_hinge = original
    assert float(hinge) == 7.0
    assert card["geometry_scale"] == 1.0
    combined = normalized.objective_from_presentation_terms(
        [ce], [hinge], ["common"], geometry_weight=0.01
    )
    expected = 0.5 * (ce + 0.01 * hinge) / normalized.BRANCH_IMAGE_COUNT
    torch.testing.assert_close(combined, expected)
    whole_loss_scaled = 0.5 * (ce + 0.01 * hinge) / normalized.BRANCH_IMAGE_COUNT
    wrong = 0.5 * (ce + 0.01 * hinge) * 0.5 / normalized.BRANCH_IMAGE_COUNT
    assert not torch.equal(whole_loss_scaled, wrong)


def test_four_rank_branch_compensation_remains_exact() -> None:
    parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    common = [parameter * float(index + 1) for index in range(32)]
    variable = [parameter.square() * float(index + 1) for index in range(32)]
    common_hinges = [parameter.square() for _ in common]
    variable_hinges = [2 * parameter.square() for _ in variable]
    local = parameter.new_zeros(())
    for rank in range(4):
        start, stop = rank * 8, (rank + 1) * 8
        local = local + normalized.objective_from_presentation_terms(
            common[start:stop] + variable[start:stop],
            common_hinges[start:stop] + variable_hinges[start:stop],
            ["common"] * 8 + ["variable"] * 8,
        )
    expected = 0.5 * sum(common) / 32 + 0.5 * sum(variable) / 32
    expected = expected + 0.01 * (
        0.5 * sum(common_hinges) / 32 + 0.5 * sum(variable_hinges) / 32
    )
    torch.testing.assert_close(local, expected)
    torch.testing.assert_close(
        torch.autograd.grad(local, parameter, retain_graph=True)[0],
        torch.autograd.grad(expected, parameter)[0],
    )


def test_normalization_receipt_logs_numerator_denominators_and_scale_distribution() -> None:
    cards = []
    for branch, scales in (("common", (1.0, 1.0)), ("variable", (0.25, 0.5))):
        for index, scale in enumerate(scales):
            cards.append(
                {
                    "branch": branch,
                    "ce_numerator": float(index + 1),
                    "old_ce_denominator_active_tokens": 10,
                    "new_ce_denominator_canonical_full_target_tokens": int(10 / scale),
                    "ce_scale_active_over_canonical": scale,
                }
            )
    summary = normalized.summarize_normalization(cards)
    assert summary["variable"]["ce_numerator_sum"] == 3.0
    assert summary["variable"]["old_active_token_denominator_sum"] == 20
    assert summary["variable"]["new_canonical_full_target_denominator_sum"] == 60
    assert summary["variable"]["scale_distribution"]["min"] == 0.25
    assert summary["variable"]["scale_distribution"]["max"] == 0.5
    assert summary["variable"]["scale_distribution"]["median"] == 0.375


def test_normalized_resolver_preserves_completion_and_fallback_selection() -> None:
    records = []
    for image_id in range(normalized.IMAGE_COUNT):
        eligible = image_id != 0
        records.append(
            {
                "image_id": image_id,
                "example_id": str(image_id),
                "canonical_route": _route(kind="canonical", image_id=image_id),
                "completion_route": _route(kind="completion", image_id=image_id)
                if eligible
                else None,
                "eligibility": {
                    "fully_eligible": eligible,
                    "fallback_reason": None if eligible else "no_trusted_prefix",
                },
            }
        )
    update = {
        "step": 1,
        "common_image_ids": list(range(32)),
        "variable_image_ids": list(range(64, 96)),
    }
    selected = normalized.resolve_update_presentations(
        records, update, arm=normalized.B_NORMALIZED
    )
    assert Counter(row["route_kind"] for row in selected) == {
        "canonical": 32,
        "completion": 32,
    }
    completion = next(row for row in selected if row["route_kind"] == "completion")
    assert completion["canonical_full_target_token_count"] == 10
    assert completion["route"]["_canonical_full_target_token_count"] == 10
    fallback_update = {
        "step": 1,
        "common_image_ids": list(range(32)),
        "variable_image_ids": [0] + list(range(96, 127)),
    }
    fallback = normalized.resolve_update_presentations(
        records, fallback_update, arm=normalized.B_NORMALIZED
    )
    row = next(item for item in fallback if item["branch"] == "variable" and item["image_id"] == 0)
    assert row["route_kind"] == "canonical"
    assert row["fallback_reason"] == "no_trusted_prefix"


def test_actual_entry_requires_lead_release_before_manifest_or_cuda_work(tmp_path) -> None:
    with pytest.raises(ValueError, match="lead qualification release receipt"):
        normalized.run(
            tmp_path / "missing-manifest.json",
            output=tmp_path / "training",
        )


def test_qualification_receipt_binds_qualification_and_main_entries(tmp_path) -> None:
    preparation = {"path": "/fixed/preparation.json", "sha256": "a" * 64, "size_bytes": 1}
    qualification_manifest = tmp_path / "qualification-manifest.json"
    qualification_manifest.write_text(json.dumps({"preparation": preparation}))
    main_manifest = tmp_path / "main-manifest.json"
    main_manifest.write_text("{}")
    terminal = tmp_path / "qualification-terminal.json"
    terminal.write_text("{}")
    receipt = {
        "schema": normalized.QUALIFICATION_SCHEMA,
        "status": "accepted_actual_entry",
        "unit_id": normalized.UNIT_ID,
        "arm": normalized.B_NORMALIZED,
        "normalization": normalized.QUALIFICATION_NORMALIZATION,
        "qualification_manifest": normalized.training.binding(qualification_manifest),
        "main_training_manifest": normalized.training.binding(main_manifest),
        "training_terminal": normalized.training.binding(terminal),
        "qualification_terminal": normalized.training.binding(terminal),
        "preparation": preparation,
    }
    assert normalized.validate_qualification_receipt(
        receipt,
        manifest_path=qualification_manifest,
        main_manifest_path=main_manifest,
    ) == receipt


def test_actual_entry_passes_explicit_variant_functions_without_changing_predecessor(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    manifest = {"mode": "main", "arm": normalized.B_NORMALIZED}
    calls = []
    original_terms = normalized.replay.route_terms
    original_resolver = predecessor.resolve_update_presentations
    monkeypatch.setattr(normalized, "validate_training_manifest", lambda value: manifest)
    monkeypatch.setattr(normalized, "validate_release", lambda *args, **kwargs: {})
    monkeypatch.setattr(normalized, "_dependency_bindings", lambda: {"normalized": {"sha256": "bound"}})
    def execute(path, **kwargs):
        calls.append((path, kwargs))
        return {"status": "completed"}
    monkeypatch.setattr(predecessor, "run_paired_training", execute)
    assert normalized.run(manifest_path, output=tmp_path / "output", release_receipt=tmp_path / "release") == {"status": "completed"}
    path, options = calls.pop()
    assert path == manifest_path.resolve()
    assert options["receipt_schema"] == normalized.SCHEMA
    assert options["validate_manifest"] is normalized.validate_training_manifest
    assert options["resolve_presentations"] is normalized.resolve_update_presentations
    assert options["route_terms"] is normalized._route_terms
    assert options["enrich_update"] is normalized._enrich_update
    assert options["dependency_bindings"] == {"normalized": {"sha256": "bound"}}
    assert normalized.replay.route_terms is original_terms
    assert predecessor.resolve_update_presentations is original_resolver


def test_normalized_projection_preserves_original_variant_input_and_digest() -> None:
    # Exercise the projection boundary independently of production artifacts.
    original = {"arm": normalized.B_NORMALIZED, "ce_normalization": {"completion_only": True},
                "objective": {}, "schema": normalized.MANIFEST_SCHEMA}
    shadow = normalized._shadow_for_predecessor_validation(original)
    assert shadow["arm"] == "B" and "ce_normalization" not in shadow
    assert original["arm"] == normalized.B_NORMALIZED
    assert shadow["content_sha256"] == normalized.training.digest(
        {key: value for key, value in shadow.items() if key != "content_sha256"})
