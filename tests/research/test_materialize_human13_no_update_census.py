from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.research.materialize_human13_no_update_census import (
    DiscoveryRecord,
    ExactLogitEvidence,
    materialize_census_plan,
    run_census_with_exact_logits,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _trace(token_ids: tuple[int, ...], texts: tuple[str, ...]) -> DiscoveryRecord:
    return DiscoveryRecord(
        image_id=1584,
        trajectory_id="unused",
        prompt_token_ids_sha256="a" * 64,
        chat_text_sha256="b" * 64,
        token_ids=token_ids,
        token_texts=texts,
    )


def _fixture() -> tuple[SimpleNamespace, tuple[DiscoveryRecord, ...]]:
    row_a_ids = tuple(range(100, 109))
    row_a_text = (
        "<|object_ref_start|>",
        "cat",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
        "<|box_end|>",
    )
    row_b_ids = tuple(range(200, 210))
    row_b_text = (
        "<|object_ref_start|>",
        "traffic ",
        "light",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|coord_5|>",
        "<|coord_6|>",
        "<|coord_7|>",
        "<|coord_8|>",
        "<|box_end|>",
    )
    stop = 999
    clean_prefix = (50, 51)
    source = SimpleNamespace(
        trajectory_id="human13:1584:source",
        raw_token_ids=(*clean_prefix, stop),
        rows=(),
        prefix=SimpleNamespace(clean_token_ids=clean_prefix),
    )
    k_a = SimpleNamespace(
        trajectory_id="human13:1584:k16:21001",
        raw_token_ids=(*row_a_ids, stop),
        rows=(SimpleNamespace(row_id="row-a", token_start=0, token_end=9),),
    )
    k_b = SimpleNamespace(
        trajectory_id="human13:1584:k16:21002",
        raw_token_ids=(*row_b_ids, stop),
        rows=(SimpleNamespace(row_id="row-b", token_start=0, token_end=10),),
    )
    image = SimpleNamespace(
        image_id=1584,
        trajectories=(source, k_a, k_b),
        selected_rows=(
            SimpleNamespace(
                owner_id="gt:1584:1",
                row_id="row-a",
                trajectory_id=k_a.trajectory_id,
                row_index=0,
                token_ids=row_a_ids,
                target_token_mask=(True,) * len(row_a_ids),
            ),
            SimpleNamespace(
                owner_id="gt:1584:4",
                row_id="row-b",
                trajectory_id=k_b.trajectory_id,
                row_index=0,
                token_ids=row_b_ids,
                target_token_mask=(True,) * len(row_b_ids),
            ),
        ),
        h_owner_ids=("gt:1584:1", "gt:1584:4"),
        g_owner_ids=("gt:1584:0",),
        m_owner_ids=("gt:1584:2",),
    )
    manifest = SimpleNamespace(images=(image,), full_panel=True)
    records = (
        replace(
            _trace(
                (*clean_prefix, stop),
                ("source-a", "source-b", "<|im_end|>"),
            ),
            trajectory_id=source.trajectory_id,
        ),
        replace(
            _trace((*row_a_ids, stop), (*row_a_text, "<|im_end|>")),
            trajectory_id=k_a.trajectory_id,
        ),
        replace(
            _trace((*row_b_ids, stop), (*row_b_text, "<|im_end|>")),
            trajectory_id=k_b.trajectory_id,
        ),
    )
    return manifest, records


def test_materializes_native_rows_and_one_coherent_a1_chain_with_exact_roles() -> None:
    manifest, records = _fixture()

    plan = materialize_census_plan(
        manifest=manifest,
        manifest_sha256="c" * 64,
        discovery_records=records,
    )

    assert [
        (row.owner_id, row.row_id, row.token_ids) for row in plan.selected_rows
    ] == [
        ("gt:1584:1", "row-a", tuple(range(100, 109))),
        ("gt:1584:4", "row-b", tuple(range(200, 210))),
    ]
    assert len(plan.segments) == 1
    segment = plan.segments[0]
    assert segment.segment_id == "a1:1584"
    assert segment.segment_role == "a1_full_h"
    assert segment.fixed_prefix_token_ids == (50, 51)
    assert segment.token_ids == (50, 51, *range(100, 109), *range(200, 210))
    assert segment.row_ids == ("row-a", "row-b")

    sites = plan.coherent_sites
    assert sites[0].segment_prefix_token_ids == (50, 51)
    assert sites[0].target_token_id == 100
    assert sites[0].token_role == "boundary"
    assert sites[1].token_role == "description"
    assert sites[2].token_role == "schema"
    assert sites[3].token_role == "schema"
    assert {site.token_role for site in sites[4:8]} == {"coordinate"}
    assert sites[8].token_role == "row_terminator"
    assert sites[9].segment_prefix_token_ids == (50, 51, *range(100, 109))
    assert sites[9].target_token_id == 200
    assert [site.token_role for site in sites[10:12]] == [
        "description",
        "description",
    ]
    assert all(site.segment_id == "a1:1584" for site in sites)
    assert all(site.prompt_token_ids_sha256 == "a" * 64 for site in sites)
    root = next(site for site in plan.trie_sites if site.prefix_token_ids == ())
    assert root.model_prefix_token_ids == (50, 51)
    assert plan.actions == {
        "model_imports": 0,
        "model_loads": 0,
        "forwards": 0,
        "gpu_allocations": 0,
        "artifact_writes": 0,
    }


def _observed_evidence(plan: object) -> tuple[ExactLogitEvidence, ...]:
    evidence: list[ExactLogitEvidence] = []
    for site in plan.trie_sites:
        logits = torch.zeros(1024)
        logits[site.viable_child_token_ids[0]] = 2.0
        evidence.append(
            ExactLogitEvidence.observed(
                plan=plan,
                site=site,
                surface="trie",
                logits=logits,
            )
        )
    for site in plan.coherent_sites:
        logits = torch.zeros(1024)
        logits[site.target_token_id] = 3.0
        for surface in ("packed", "hf"):
            evidence.append(
                ExactLogitEvidence.observed(
                    plan=plan,
                    site=site,
                    surface=surface,
                    logits=logits.clone(),
                )
            )
    return tuple(evidence)


def test_exact_logit_injection_delegates_to_existing_census_semantics() -> None:
    manifest, records = _fixture()
    plan = materialize_census_plan(
        manifest=manifest,
        manifest_sha256="c" * 64,
        discovery_records=records,
    )

    result = run_census_with_exact_logits(
        plan=plan,
        evidence=_observed_evidence(plan),
    )

    assert result["schema_version"] == "human13_k_union_no_update_census.v1"
    assert result["trie"]["original_row_count"] == 2
    assert result["coherent_chain"]["site_count"] == 19
    assert result["coherent_chain"]["first_non_argmax_site"] is None
    assert result["aligned_surface"]["maximum_absolute_margin_drift"] == 0.0
    assert result["frozen_targets"]["byte_identical"] is True


@pytest.mark.parametrize("surface", ["trie", "packed", "hf"])
def test_missing_any_exact_logit_surface_fails_closed(surface: str) -> None:
    manifest, records = _fixture()
    plan = materialize_census_plan(
        manifest=manifest,
        manifest_sha256="c" * 64,
        discovery_records=records,
    )
    evidence = list(_observed_evidence(plan))
    evidence.pop(
        next(index for index, item in enumerate(evidence) if item.surface == surface)
    )

    with pytest.raises(ValueError, match="exact logit evidence coverage"):
        run_census_with_exact_logits(plan=plan, evidence=evidence)


def test_logit_evidence_is_content_bound_to_the_exact_site_and_plan() -> None:
    manifest, records = _fixture()
    plan = materialize_census_plan(
        manifest=manifest,
        manifest_sha256="c" * 64,
        discovery_records=records,
    )
    evidence = list(_observed_evidence(plan))
    evidence[0] = replace(evidence[0], site_binding_sha256="0" * 64)

    with pytest.raises(ValueError, match="content binding"):
        run_census_with_exact_logits(plan=plan, evidence=evidence)


def test_logit_vector_mutated_after_capture_fails_content_binding() -> None:
    manifest, records = _fixture()
    plan = materialize_census_plan(
        manifest=manifest,
        manifest_sha256="c" * 64,
        discovery_records=records,
    )
    evidence = list(_observed_evidence(plan))
    evidence[0].logits[0] = 123.0

    with pytest.raises(ValueError, match="logit vector content"):
        run_census_with_exact_logits(plan=plan, evidence=evidence)


def test_discovery_trace_or_prompt_mismatch_fails_closed() -> None:
    manifest, records = _fixture()
    bad_token = replace(records[1], token_ids=(777, *records[1].token_ids[1:]))
    with pytest.raises(ValueError, match="token IDs differ"):
        materialize_census_plan(
            manifest=manifest,
            manifest_sha256="c" * 64,
            discovery_records=(records[0], bad_token, records[2]),
        )

    bad_prompt = replace(records[2], prompt_token_ids_sha256="f" * 64)
    with pytest.raises(ValueError, match="prompt identity differs"):
        materialize_census_plan(
            manifest=manifest,
            manifest_sha256="c" * 64,
            discovery_records=(records[0], records[1], bad_prompt),
        )


def test_noncanonical_or_ambiguous_row_token_shape_fails_closed() -> None:
    manifest, records = _fixture()
    bad_text = replace(
        records[1],
        token_texts=("<|object_ref_start|>cat", *records[1].token_texts[1:]),
    )

    with pytest.raises(ValueError, match="canonical compact row token shape"):
        materialize_census_plan(
            manifest=manifest,
            manifest_sha256="c" * 64,
            discovery_records=(records[0], bad_text, records[2]),
        )


def test_direct_cli_entry_resolves_repository_imports_without_runtime_actions(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/research/materialize_human13_no_update_census.py"),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
