from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

import scripts.research.analyze_trajectory_owner_set_adjudication_salvage_gate as gate
import scripts.research.select_trajectory_owner_set_adjudication_review_sample as selector
from scripts.research.assemble_constant_dose_breadth_state_banks import AssemblyError
from src.config.fingerprint import sha256_file, sha256_json
from src.inference.backend import token_ids_sha256


def _state(
    candidate_id: str,
    token: str,
    categories: list[str],
    *,
    usable: bool = True,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "projected_token_sha256": token * 64,
        "projection_status": "accepted_natural_end",
        "parser_status": "accepted",
        "invalid_before_b16_count": 0,
        "ordered_categories": categories,
        "parser_usable": usable,
    }


def _search(*states: dict[str, Any]) -> dict[str, Any]:
    return gate.search_category_possibility(gate.collapse_exact_token_states(states))


def _adapter_route(
    *,
    image_id: str,
    candidate_id: str,
    token_id: int,
    categories: list[str],
    malformed_count: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    token_ids = [token_id]
    token_hash = token_ids_sha256(token_ids)
    provenance = {
        "status": "accepted_natural_end",
        "projected_token_ids_sha256": token_hash,
        "projected_token_count": len(token_ids),
        "raw_generated_token_ids_sha256": token_hash,
        "raw_generated_token_count": len(token_ids),
    }
    route_row = {
        "generated_token_ids": token_ids,
        "generated_token_ids_sha256": token_hash,
        (
            "_source_b16_provenance"
            if candidate_id == "source-b16"
            else "_sampled_b16_provenance"
        ): provenance,
    }
    route_evidence = {"parser": {"parse_status": "accepted"}}
    assignment = {
        "malformed_row_count": malformed_count,
        "row_counts": {"malformed": malformed_count},
        "row_assignment_receipts": [
            {
                "generated_row_index": index,
                "prediction_id": f"{image_id}:{candidate_id}:{index}",
                "category": category,
            }
            for index, category in enumerate(categories)
        ],
    }
    return route_row, route_evidence, assignment


def _three_image_adapter() -> dict[str, Any]:
    adapter: dict[str, Any] = {
        "source_rows": {},
        "sampled_rows": {},
        "image_results": {},
    }
    for image_id in ("1", "2", "3"):
        evidence: dict[str, Any] = {}
        assignments: dict[str, Any] = {}
        for route_index, candidate_id in enumerate(
            ["source-b16", *(f"sample-{index:02d}" for index in range(16))]
        ):
            if image_id == "1":
                if route_index == 0:
                    token_id, categories, malformed = 10, ["cat", "dog"], 0
                elif route_index == 1:
                    token_id, categories, malformed = 11, ["dog", "cat"], 0
                else:
                    token_id, categories, malformed = 12, ["cat"], 0
            elif image_id == "2":
                token_id, categories, malformed = 20, ["cat"], 0
            elif route_index == 0:
                token_id, categories, malformed = 30, ["cat"], 1
            else:
                token_id, categories, malformed = 31, ["cat"], 0
            route_row, route_evidence, assignment = _adapter_route(
                image_id=image_id,
                candidate_id=candidate_id,
                token_id=token_id,
                categories=categories,
                malformed_count=malformed,
            )
            if candidate_id == "source-b16":
                adapter["source_rows"][(image_id, 0)] = route_row
            else:
                adapter["sampled_rows"][(image_id, route_index - 1)] = route_row
            evidence[candidate_id] = route_evidence
            assignments[candidate_id] = assignment
        adapter["image_results"][image_id] = {
            "trajectory_evidence": evidence,
            "budgets": [{"trajectory_assignments": assignments}],
        }
    return adapter


def _synthetic_success_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    monkeypatch.setattr(gate, "EXPECTED_POPULATION_COUNT", 3)
    monkeypatch.setattr(gate, "_verify_source_snapshot", lambda *args: None)
    monkeypatch.setattr(gate, "_stage_zero_git_identity", lambda *args: {})
    categories, possibilities, summary = gate.analyze_stage_zero(
        _three_image_adapter(), ["1", "2", "3"], reverse_input=False
    )
    root = tmp_path / "stage-zero"
    root.mkdir()
    gate._write_jsonl(root / "category-state.jsonl", categories)
    gate._write_jsonl(root / "possibility-census.jsonl", possibilities)
    gate._write_json(root / "summary.json", summary)
    possible_pool = gate._pool_payload(["1"], pool_role="possible")
    impossible_pool = gate._pool_payload(
        ["2", "3"], pool_role="certified_impossible"
    )
    gate._write_json(root / "possible-pool.json", possible_pool)
    gate._write_json(root / "impossible-pool.json", impossible_pool)
    snapshot_root = root / "source-snapshot"
    snapshot_root.mkdir()
    gate._write_json(snapshot_root / "manifest.json", {"synthetic": True})
    outputs = gate._output_inventory(root)
    output_directories = gate._directory_inventory(root)
    receipt = {
        "schema_version": gate.SCHEMA_VERSION,
        "terminal_status": "completed",
        "population_scope": "train_only_censored_nonpassing_U",
        "outputs": outputs,
        "output_directories": output_directories,
        "population": {
            "count": 3,
            "ordered_image_ids_sha256": sha256_json(["1", "2", "3"]),
        },
        "row_counts": {
            "category_state": 3,
            "possibility_census": 3,
            "possible_pool": 1,
            "impossible_pool": 2,
        },
        "determinism": {
            "forward_category_state_sha256": sha256_json(categories),
            "reversed_category_state_sha256": sha256_json(categories),
            "forward_possibility_census_sha256": sha256_json(possibilities),
            "reversed_possibility_census_sha256": sha256_json(possibilities),
            "forward_summary_sha256": sha256_json(summary),
            "reversed_summary_sha256": sha256_json(summary),
        },
        "possible_pool_count": 1,
        "ordered_possible_image_ids_sha256": possible_pool[
            "ordered_image_ids_sha256"
        ],
        "possible_pool_artifact_sha256": outputs["possible-pool.json"]["sha256"],
        "artifact_integrity_checks": {"synthetic_proof": True},
        "source_snapshot": {},
        "git_identity": {},
    }
    gate._write_json(root / "receipt.json", receipt)
    return root, receipt


def _write_manifest_panel(root: Path, *, mode: str) -> tuple[list[str], Path]:
    image_ids: list[str] = []
    first_batch: Path | None = None
    for worker_index in range(8):
        worker = root / f"worker-{worker_index:02d}-of-08"
        worker.mkdir(parents=True)
        batches: list[dict[str, Any]] = []
        for local_index in range(19):
            batch_index = worker_index * 19 + local_index
            batch_ids = [str(batch_index * 16 + offset + 1) for offset in range(16)]
            image_ids.extend(batch_ids)
            filename = f"{mode}-batch-{batch_index:05d}.json"
            batch_path = worker / filename
            batch_path.write_text(
                json.dumps({"batch_index": batch_index}), encoding="utf-8"
            )
            if first_batch is None:
                first_batch = batch_path
            batches.append(
                {
                    "batch_index": batch_index,
                    "image_ids": batch_ids,
                    "artifacts": {
                        mode: {"path": filename, "sha256": sha256_file(batch_path)}
                    },
                }
            )
        (worker / "manifest.json").write_text(
            json.dumps(
                {
                    "worker_index": worker_index,
                    "worker_count": 8,
                    "status": "completed",
                    "batches": batches,
                }
            ),
            encoding="utf-8",
        )
    assert first_batch is not None
    return image_ids, first_batch


def test_parser_boundary_allows_after_boundary_drop_but_not_invalid_prefix() -> None:
    projected_ids = [1, 2, 3]
    projected_hash = token_ids_sha256(projected_ids)
    route: dict[str, Any] = {
        "generated_token_ids": projected_ids,
        "generated_token_ids_sha256": projected_hash,
        "_sampled_b16_provenance": {
            "status": "accepted_budget",
            "projected_token_ids_sha256": projected_hash,
            "projected_token_count": len(projected_ids),
            "raw_generated_token_ids_sha256": "b" * 64,
            "raw_generated_token_count": len(projected_ids) + 1,
        },
    }
    evidence = {
        "parser": {
            "parse_status": "accepted_with_drops",
            # Adapter's malformed_row_count, not this full-completion count,
            # owns whether a drop occurred before the B16 boundary.
            "dropped_prediction_count": 1,
        }
    }
    assignment = {
        "malformed_row_count": 0,
        "row_assignment_receipts": [
            {
                "generated_row_index": 0,
                "prediction_id": "row-0",
                "category": "Traffic_Light",
            }
        ],
    }

    accepted = gate._candidate_state(
        candidate_id="sample-00",
        route_row=route,
        route_evidence=evidence,
        assignment=assignment,
    )
    assert accepted["parser_usable"] is True
    assert accepted["ordered_categories"] == ["traffic light"]

    invalid_assignment = {**assignment, "malformed_row_count": 1}
    assert (
        gate._candidate_state(
            candidate_id="sample-00",
            route_row=route,
            route_evidence=evidence,
            assignment=invalid_assignment,
        )["parser_usable"]
        is False
    )
    invalid_row_counts = {**assignment, "row_counts": {"malformed": 1}}
    assert (
        gate._candidate_state(
            candidate_id="sample-00",
            route_row=route,
            route_evidence=evidence,
            assignment=invalid_row_counts,
        )["parser_usable"]
        is False
    )
    invalid_projection = {
        **route,
        "_sampled_b16_provenance": {
            **route["_sampled_b16_provenance"],
            "status": "failed_invalid_before_budget",
        },
    }
    assert (
        gate._candidate_state(
            candidate_id="sample-00",
            route_row=invalid_projection,
            route_evidence=evidence,
            assignment=assignment,
        )["parser_usable"]
        is False
    )
    invalid_token_provenance = {
        **route,
        "_sampled_b16_provenance": {
            **route["_sampled_b16_provenance"],
            "projected_token_ids_sha256": "0" * 64,
        },
    }
    with pytest.raises(AssemblyError, match="B16 token provenance differs"):
        gate._candidate_state(
            candidate_id="sample-00",
            route_row=invalid_token_provenance,
            route_evidence=evidence,
            assignment=assignment,
        )


def test_exact_token_aliases_collapse_and_disagreement_fails() -> None:
    left = _state("source-b16", "a", ["cat", "dog"])
    right = _state("sample-00", "a", ["cat", "dog"])

    representatives = gate.collapse_exact_token_states([right, left])

    assert len(representatives) == 1
    assert representatives[0]["representative_candidate_id"] == "source-b16"
    assert representatives[0]["alias_candidate_ids"] == ["source-b16", "sample-00"]
    with pytest.raises(AssemblyError, match="exact-token aliases disagree"):
        gate.collapse_exact_token_states(
            [left, _state("sample-00", "a", ["cat"])]
        )
    with pytest.raises(AssemblyError, match="exact-token aliases disagree"):
        gate.collapse_exact_token_states(
            [left, _state("sample-00", "a", ["cat", "dog"], usable=False)]
        )


def test_different_first_categories_and_proper_lower_support_are_possible() -> None:
    result = _search(
        _state("sample-00", "a", ["cat", "dog"]),
        _state("sample-01", "b", ["dog", "cat"]),
        _state("sample-02", "c", ["cat"]),
    )

    assert result["possible"] is True
    witness = result["canonical_witness"]
    assert witness["d_token_sha256"] == "a" * 64
    assert witness["d_candidate_id"] == "sample-00"
    assert witness["empty_lower_vacuity"] is False


def test_same_first_category_requires_two_owner_capacity() -> None:
    impossible = _search(
        _state("sample-00", "a", ["cat"]),
        _state("sample-01", "b", ["cat"]),
        _state("sample-02", "c", []),
    )
    possible = _search(
        _state("sample-00", "a", ["cat", "cat"]),
        _state("sample-01", "b", ["cat", "cat"]),
        _state("sample-02", "c", []),
    )

    assert impossible["possible"] is False
    assert possible["possible"] is True
    assert possible["canonical_witness"]["higher_category_capacity"] == {"cat": 2}


def test_equal_support_requires_strict_multiplicity() -> None:
    impossible = _search(
        _state("sample-00", "a", ["cat", "dog"]),
        _state("sample-01", "b", ["dog", "cat"]),
        _state("sample-02", "c", ["cat", "dog"]),
    )
    possible = _search(
        _state("sample-00", "a", ["cat", "cat", "dog"]),
        _state("sample-01", "b", ["dog", "cat", "cat"]),
        _state("sample-02", "c", ["cat", "dog"]),
    )

    assert impossible["possible"] is False
    assert possible["possible"] is True


def test_empty_lower_is_vacuous_and_needs_no_d() -> None:
    result = _search(
        _state("sample-00", "a", ["cat", "dog"]),
        _state("sample-01", "b", ["dog", "cat"]),
        _state("sample-02", "c", []),
    )

    assert result["possible"] is True
    assert result["canonical_witness"]["d_token_sha256"] is None
    assert result["canonical_witness"]["d_candidate_id"] is None
    assert result["canonical_witness"]["empty_lower_vacuity"] is True


def test_nonempty_lower_allows_d_to_equal_a_or_b() -> None:
    result = _search(
        _state("sample-00", "a", ["cat", "dog"]),
        _state("sample-01", "b", ["dog", "cat"]),
        _state("sample-02", "c", ["cat"]),
    )

    witness = result["canonical_witness"]
    assert witness["d_token_sha256"] in {
        witness["a_token_sha256"],
        witness["b_token_sha256"],
    }
    assert witness["d_token_sha256"] != witness["k_token_sha256"]


def test_impossibility_certificate_replays_and_rejects_tampering() -> None:
    representatives = gate.collapse_exact_token_states(
        [
            _state("sample-00", "a", ["cat"]),
            _state("sample-01", "b", ["cat"]),
            _state("sample-02", "c", ["cat"]),
        ]
    )
    result = gate.search_category_possibility(representatives)

    assert result["possible"] is False
    certificate = result["impossibility_certificate"]
    assert certificate["attempt_count"] > 0
    assert gate.replay_impossibility_certificate(representatives, certificate)
    tampered = {**certificate, "attempt_count": certificate["attempt_count"] + 1}
    assert not gate.replay_impossibility_certificate(representatives, tampered)


def test_candidate_order_reversal_preserves_witness_and_certificate() -> None:
    possible_states = [
        _state("sample-00", "a", ["cat", "dog"]),
        _state("sample-01", "b", ["dog", "cat"]),
        _state("sample-02", "c", ["cat"]),
    ]
    impossible_states = [
        _state("sample-00", "d", ["cat"]),
        _state("sample-01", "e", ["cat"]),
        _state("sample-02", "f", ["cat"]),
    ]

    assert _search(*possible_states) == _search(*reversed(possible_states))
    assert _search(*impossible_states) == _search(*reversed(impossible_states))


def test_three_image_stage_zero_partition_reconciles_end_to_end() -> None:
    adapter = _three_image_adapter()

    categories, possibilities, summary = gate.analyze_stage_zero(
        adapter, ["3", "1", "2"], reverse_input=False
    )
    reversed_outputs = gate.analyze_stage_zero(
        adapter, ["1", "2", "3"], reverse_input=True
    )

    assert (categories, possibilities, summary) == reversed_outputs
    assert [row["image_id"] for row in categories] == ["1", "2", "3"]
    assert [row["image_id"] for row in possibilities if row["possible"]] == ["1"]
    assert [row["image_id"] for row in possibilities if not row["possible"]] == [
        "2",
        "3",
    ]
    parser_invalid = categories[2]
    assert parser_invalid["parser_usable_candidate_count"] == 16
    assert parser_invalid["candidate_states"][0]["parser_usable"] is False
    for category_row, possibility_row in zip(categories, possibilities, strict=True):
        if not possibility_row["possible"]:
            assert gate.replay_impossibility_certificate(
                category_row["exact_token_representatives"],
                possibility_row["impossibility_certificate"],
            )
    assert summary["population_count"] == 3
    assert summary["possible_pool_count"] == 1
    assert summary["impossible_pool_count"] == 2
    assert summary["possible_image_ids"] == ["1"]
    assert summary["impossible_image_ids"] == ["2", "3"]
    assert summary["category_state_sha256"] == sha256_json(categories)
    assert summary["possibility_census_sha256"] == sha256_json(possibilities)


def test_derive_population_is_exact_and_order_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "EXPECTED_CENSUS_IMAGE_COUNT", 3)
    monkeypatch.setattr(gate, "EXPECTED_POPULATION_COUNT", 2)
    records = [
        {
            "image_id": "10",
            "fully_adjudicable": False,
            "admission": {"primary_natural_alias_admitted": False},
        },
        {
            "image_id": "2",
            "fully_adjudicable": False,
            "admission": {"primary_natural_alias_admitted": False},
        },
        {
            "image_id": "1",
            "fully_adjudicable": True,
            "admission": {"primary_natural_alias_admitted": False},
        },
    ]

    assert gate.derive_population(records) == ["2", "10"]
    assert gate.derive_population(list(reversed(records))) == ["2", "10"]


def test_possible_pool_payload_uses_selector_canonical_identifiers() -> None:
    payload = gate._pool_payload(["10", "2"], pool_role="possible")

    assert payload["schema_version"] == gate.SCHEMA_VERSION
    assert payload["terminal_status"] == "completed"
    assert payload["population_scope"] == "train_only_censored_nonpassing_U"
    assert payload["pool_role"] == "possible"
    assert payload["ordered_image_ids"] == ["2", "10"]
    assert payload["count"] == 2
    assert payload["ordered_image_ids_sha256"] == sha256_json(["2", "10"])
    assert selector.canonicalize_image_ids(payload["ordered_image_ids"]) == (
        "2",
        "10",
    )


def test_exact_file_pin_rejects_byte_identical_alternate_path(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.json"
    alternate = tmp_path / "alternate.json"
    frozen.write_text("{}\n", encoding="utf-8")
    alternate.write_bytes(frozen.read_bytes())
    digest = sha256_file(frozen)

    assert gate._validate_exact_file(frozen, frozen, digest) == frozen.resolve()
    with pytest.raises(AssemblyError, match="input path is not frozen"):
        gate._validate_exact_file(alternate, frozen, digest)
    with pytest.raises(AssemblyError, match="input hash differs"):
        gate._validate_exact_file(frozen, frozen, "0" * 64)


def test_fixed_input_post_revalidation_detects_mutation(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.json"
    frozen.write_text("{}\n", encoding="utf-8")
    digest = sha256_file(frozen)
    assert gate._observe_fixed_inputs({"frozen": frozen}, {"frozen": digest})

    frozen.write_text('{"mutated":true}\n', encoding="utf-8")

    with pytest.raises(AssemblyError, match="frozen input changed"):
        gate._observe_fixed_inputs({"frozen": frozen}, {"frozen": digest})


def test_exact_manifest_revalidation_detects_bound_batch_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sampled"
    image_ids, first_batch = _write_manifest_panel(root, mode="sampled")
    receipt = gate._validate_panel_manifests_exact(
        root, mode="sampled", expected_pool_ids=image_ids
    )
    assert receipt["manifest_count"] == 8
    assert receipt["batch_count"] == 152
    assert receipt["image_inventory_count"] == 2_432

    first_batch.write_text('{"mutated":true}', encoding="utf-8")

    with pytest.raises(AssemblyError, match="hash differs before JSON decoding"):
        gate._validate_panel_manifests_exact(
            root, mode="sampled", expected_pool_ids=image_ids
        )


def test_adapter_provenance_must_exactly_equal_manifest_inventory() -> None:
    sampled = {
        "batches": [{"path": "/sampled.json", "sha256": "a" * 64}]
    }
    source = {
        "batches": [{"path": "/source.json", "sha256": "b" * 64}]
    }
    provenance = [
        {
            "mode": "sampled",
            "path": "/sampled.json",
            "sha256": "a" * 64,
        },
        {
            "mode": "source_b16",
            "path": "/source.json",
            "sha256": "b" * 64,
        },
    ]
    adapter = {"census": {"artifact_provenance": provenance}}

    assert gate._validate_adapter_artifact_inventory(
        adapter, sampled_manifests=sampled, source_manifests=source
    ) == sorted(provenance, key=lambda item: (item["mode"], item["path"]))
    adapter["census"]["artifact_provenance"] = [
        *provenance,
        {"mode": "sampled", "path": "/extra.json", "sha256": "c" * 64},
    ]
    with pytest.raises(AssemblyError, match="differs from manifest-bound inventory"):
        gate._validate_adapter_artifact_inventory(
            adapter, sampled_manifests=sampled, source_manifests=source
        )


def test_success_receipt_revalidates_impossible_pool_and_exact_root_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, receipt = _synthetic_success_root(tmp_path, monkeypatch)
    gate._validate_success_receipt_readback(
        receipt_path=root / "receipt.json",
        expected_receipt=receipt,
        output_root=root,
        repo=tmp_path,
    )

    gate._write_json(
        root / "impossible-pool.json",
        gate._pool_payload(["2"], pool_role="certified_impossible"),
    )
    with pytest.raises(AssemblyError, match="output identity differs"):
        gate._validate_success_receipt_readback(
            receipt_path=root / "receipt.json",
            expected_receipt=receipt,
            output_root=root,
            repo=tmp_path,
        )
    gate._write_json(
        root / "impossible-pool.json",
        gate._pool_payload(["2", "3"], pool_role="certified_impossible"),
    )
    (root / "unreceipted.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(AssemblyError, match="root inventory is not exact"):
        gate._validate_success_receipt_readback(
            receipt_path=root / "receipt.json",
            expected_receipt=receipt,
            output_root=root,
            repo=tmp_path,
        )


def test_source_snapshot_revalidation_detects_live_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    repo.mkdir()
    output.mkdir()
    source = repo / "task.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(gate, "SOURCE_SNAPSHOT_PATHS", ("task.py",))
    monkeypatch.setattr(
        gate.census, "_repo_loaded_source_paths", lambda _: ["task.py"]
    )
    snapshot = gate._materialize_source_snapshot(repo, output)
    gate._verify_source_snapshot(repo, output, snapshot)

    source.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(AssemblyError, match="source or snapshot drift"):
        gate._verify_source_snapshot(repo, output, snapshot)


def test_failed_publication_contains_only_immutable_receipt(tmp_path: Path) -> None:
    output = tmp_path / "failed-output"
    analyzer = Path(gate.__file__).resolve()

    gate._publish_failed_output(
        output,
        analyzer=analyzer,
        failed_validation_stage="injected",
        failure=AssemblyError("injected failure"),
    )

    assert sorted(item.name for item in output.iterdir()) == ["receipt.json"]
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["terminal_status"] == "failed"
    assert receipt["failed_validation_stage"] == "injected"
    assert receipt["failure_message"] == "injected failure"
    assert (output.stat().st_mode & 0o777) == 0o555
    assert ((output / "receipt.json").stat().st_mode & 0o777) == 0o444


def test_failed_publication_preserves_existing_target(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(AssemblyError, match="refusing to overwrite"):
        gate._publish_failed_output(
            output,
            analyzer=Path(gate.__file__).resolve(),
            failed_validation_stage="injected",
            failure=AssemblyError("injected"),
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    assert not list(tmp_path.glob(".existing.failed-*"))


def test_failed_publication_chmod_failure_never_exposes_canonical_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failed-output"

    def fail_chmod(*args: object, **kwargs: object) -> None:
        raise OSError("injected chmod failure")

    monkeypatch.setattr(gate.os, "chmod", fail_chmod)
    with pytest.raises(OSError, match="injected chmod failure"):
        gate._publish_failed_output(
            output,
            analyzer=Path(gate.__file__).resolve(),
            failed_validation_stage="injected",
            failure=AssemblyError("injected"),
        )

    assert not output.exists()


def test_failed_publication_rejects_chmod_that_does_not_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failed-output"
    monkeypatch.setattr(gate.os, "chmod", lambda *args, **kwargs: None)

    with pytest.raises(AssemblyError, match="immutable root mode differs"):
        gate._publish_failed_output(
            output,
            analyzer=Path(gate.__file__).resolve(),
            failed_validation_stage="injected",
            failure=AssemblyError("injected"),
        )

    assert not output.exists()


def test_success_publication_is_atomic_and_immutable(tmp_path: Path) -> None:
    output = tmp_path / "stage-zero-v1"
    staging = tmp_path / ".stage-zero-v1.staging-test"
    nested = staging / "source-snapshot"
    nested.mkdir(parents=True)
    (staging / "receipt.json").write_text("{}\n", encoding="utf-8")
    (nested / "manifest.json").write_text("{}\n", encoding="utf-8")

    gate._finalize_staging(
        staging,
        output,
        analyzer=Path(gate.__file__).resolve(),
        validate_completed_root=gate._assert_immutable_tree,
    )

    assert output.is_dir()
    assert not staging.exists()
    assert not list(tmp_path.glob(".stage-zero-v1.staging-*"))
    assert (output.stat().st_mode & 0o777) == 0o555
    assert ((output / "source-snapshot").stat().st_mode & 0o777) == 0o555
    assert ((output / "receipt.json").stat().st_mode & 0o777) == 0o444
    assert (
        (output / "source-snapshot" / "manifest.json").stat().st_mode & 0o777
    ) == 0o444


def test_late_finalize_failure_cleans_frozen_staging_and_publishes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "stage-zero-v1"
    staging = tmp_path / ".stage-zero-v1.staging-test"
    staging.mkdir()
    (staging / "receipt.json").write_text("{}\n", encoding="utf-8")
    real_replace = gate.os.replace
    real_rmtree = gate.shutil.rmtree

    def injected_replace(source: Path | str, destination: Path | str) -> None:
        if Path(source) == staging and Path(destination) == output:
            raise OSError("injected late rename failure")
        real_replace(source, destination)

    def guarded_rmtree(path: Path | str) -> None:
        assert Path(path).stat().st_mode & 0o200
        real_rmtree(path)

    monkeypatch.setattr(gate.os, "replace", injected_replace)
    monkeypatch.setattr(gate.shutil, "rmtree", guarded_rmtree)

    with pytest.raises(OSError, match="injected late rename failure"):
        gate._finalize_staging(
            staging,
            output,
            analyzer=Path(gate.__file__).resolve(),
            validate_completed_root=gate._assert_immutable_tree,
        )

    assert not staging.exists()
    assert not list(tmp_path.glob(".stage-zero-v1.staging-*"))
    assert not list(tmp_path.glob(".stage-zero-v1.failed-*"))
    failed = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert failed["terminal_status"] == "failed"
    assert failed["failed_validation_stage"] == "finalize_staging"
    assert (output.stat().st_mode & 0o777) == 0o555


def test_post_rename_corruption_never_leaves_completed_canonical_root(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stage-zero-v1"
    staging = tmp_path / ".stage-zero-v1.staging-test"
    staging.mkdir()
    receipt_path = staging / "receipt.json"
    receipt_path.write_text('{"terminal_status":"completed"}\n', encoding="utf-8")
    expected_hash = sha256_file(receipt_path)

    def corrupt_after_replace(root: Path) -> None:
        canonical_receipt = root / "receipt.json"
        canonical_receipt.write_text('{"terminal_status":"completed","corrupt":true}\n')

    def validate_completed(root: Path) -> None:
        gate._assert_immutable_tree(root)
        if sha256_file(root / "receipt.json") != expected_hash:
            raise AssemblyError("injected post-rename receipt corruption")

    with pytest.raises(
        AssemblyError, match="injected post-rename receipt corruption"
    ):
        gate._finalize_staging(
            staging,
            output,
            analyzer=Path(gate.__file__).resolve(),
            validate_completed_root=validate_completed,
            post_replace_hook=corrupt_after_replace,
        )

    assert output.is_dir()
    assert sorted(path.name for path in output.iterdir()) == ["receipt.json"]
    failed = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert failed["terminal_status"] == "failed"
    assert failed["failed_validation_stage"] == "post_rename_validate_completed_root"
    gate._validate_failed_output(output, failed)
    assert not list(tmp_path.glob(".stage-zero-v1.staging-*"))
    assert not list(tmp_path.glob(".stage-zero-v1.quarantine-*"))


def test_post_rename_output_corruption_fails_full_receipt_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built_root, receipt = _synthetic_success_root(tmp_path, monkeypatch)
    output = tmp_path / "stage-zero-v1"
    staging = tmp_path / ".stage-zero-v1.staging-test"
    built_root.rename(staging)

    def corrupt_after_replace(root: Path) -> None:
        (root / "impossible-pool.json").write_text(
            '{"terminal_status":"completed","corrupt":true}\n', encoding="utf-8"
        )

    def validate_completed(root: Path) -> None:
        gate._validate_success_receipt_readback(
            receipt_path=root / "receipt.json",
            expected_receipt=receipt,
            output_root=root,
            repo=tmp_path,
        )

    with pytest.raises(AssemblyError, match="output identity differs"):
        gate._finalize_staging(
            staging,
            output,
            analyzer=Path(gate.__file__).resolve(),
            validate_completed_root=validate_completed,
            post_replace_hook=corrupt_after_replace,
        )

    assert sorted(path.name for path in output.iterdir()) == ["receipt.json"]
    failed = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert failed["terminal_status"] == "failed"
    gate._validate_failed_output(output, failed)


def test_post_rename_empty_directory_corruption_fails_recursive_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built_root, receipt = _synthetic_success_root(tmp_path, monkeypatch)
    output = tmp_path / "stage-zero-v1"
    staging = tmp_path / ".stage-zero-v1.staging-test"
    built_root.rename(staging)

    def add_empty_directory(root: Path) -> None:
        extra = root / "unreceipted-empty-directory"
        extra.mkdir()
        gate.os.chmod(extra, 0o555)

    def validate_completed(root: Path) -> None:
        gate._validate_success_receipt_readback(
            receipt_path=root / "receipt.json",
            expected_receipt=receipt,
            output_root=root,
            repo=tmp_path,
        )

    with pytest.raises(AssemblyError, match="directory inventory is not exact"):
        gate._finalize_staging(
            staging,
            output,
            analyzer=Path(gate.__file__).resolve(),
            validate_completed_root=validate_completed,
            post_replace_hook=add_empty_directory,
        )

    assert sorted(path.name for path in output.iterdir()) == ["receipt.json"]
    failed = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert failed["terminal_status"] == "failed"
    gate._validate_failed_output(output, failed)


@pytest.mark.parametrize(
    "drift_field",
    [
        "commit",
        "scoped_status_sha256",
        "scoped_tracked_dirty_diff_sha256",
        "scoped_cached_diff_sha256",
    ],
)
@pytest.mark.parametrize("drift_boundary", ["last_pre_rename", "post_rename"])
def test_final_git_boundary_detects_head_status_worktree_and_index_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_field: str,
    drift_boundary: str,
) -> None:
    expected = {
        "commit": "a" * 40,
        "dirty": False,
        "scope": ["task.py"],
        "scoped_status_sha256": "b" * 64,
        "scoped_tracked_dirty_diff_sha256": "c" * 64,
        "scoped_status_lines": [],
        "scoped_cached_diff_command": [
            "git",
            "diff",
            "--cached",
            "--binary",
            "HEAD",
            "--",
            "task.py",
        ],
        "scoped_cached_diff_exit_code": 0,
        "scoped_cached_diff_sha256": "e" * 64,
        "scoped_cached_diff_size_bytes": 0,
        "scoped_cached_diff_stderr_sha256": "f" * 64,
    }
    drifted = {**expected, drift_field: "d" * len(str(expected[drift_field]))}
    observations = iter(
        [drifted] if drift_boundary == "last_pre_rename" else [expected, drifted]
    )
    monkeypatch.setattr(gate, "_stage_zero_git_identity", lambda *args: next(observations))
    output = tmp_path / "stage-zero-v1"
    staging = tmp_path / ".stage-zero-v1.staging-test"
    staging.mkdir()
    (staging / "receipt.json").write_text("{}\n", encoding="utf-8")

    def validate_pre_rename(_: Path) -> None:
        gate._require_exact_git_identity(
            tmp_path, ["task.py"], expected, boundary="last_pre_rename"
        )

    def validate_post_rename(_: Path) -> None:
        gate._require_exact_git_identity(
            tmp_path, ["task.py"], expected, boundary="post_rename"
        )

    with pytest.raises(AssemblyError, match=f"{drift_boundary} Git identity differs"):
        gate._finalize_staging(
            staging,
            output,
            analyzer=Path(gate.__file__).resolve(),
            validate_completed_root=validate_post_rename,
            validate_pre_rename=validate_pre_rename,
        )
    failed = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert failed["terminal_status"] == "failed"
    expected_stage = (
        "finalize_staging"
        if drift_boundary == "last_pre_rename"
        else "post_rename_validate_completed_root"
    )
    assert failed["failed_validation_stage"] == expected_stage


def test_stage_zero_git_identity_detects_staged_blob_only_change(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str, input_bytes: bytes | None = None) -> bytes:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout

    git("init", "-q")
    git("config", "user.email", "stage-zero@example.invalid")
    git("config", "user.name", "Stage Zero Test")
    task = repo / "task.py"
    task.write_text("HEAD = 1\n", encoding="utf-8")
    git("add", "task.py")
    git("commit", "-q", "-m", "initial")

    task.write_text("INDEX = 'A'\n", encoding="utf-8")
    git("add", "task.py")
    worktree_bytes = b"WORKTREE = 'constant'\n"
    task.write_bytes(worktree_bytes)
    head_before = git("rev-parse", "HEAD")
    status_before = git("status", "--porcelain=v1", "--", "task.py")
    first = gate._stage_zero_git_identity(repo, ["task.py"])

    staged_blob = git("hash-object", "-w", "--stdin", input_bytes=b"INDEX = 'B'\n")
    git(
        "update-index",
        "--cacheinfo",
        f"100644,{staged_blob.decode('ascii').strip()},task.py",
    )
    second = gate._stage_zero_git_identity(repo, ["task.py"])

    assert git("rev-parse", "HEAD") == head_before
    assert task.read_bytes() == worktree_bytes
    assert git("status", "--porcelain=v1", "--", "task.py") == status_before
    first_base = {
        key: value
        for key, value in first.items()
        if not key.startswith("scoped_cached_diff_")
    }
    second_base = {
        key: value
        for key, value in second.items()
        if not key.startswith("scoped_cached_diff_")
    }
    assert first_base == second_base
    assert first["scoped_cached_diff_sha256"] != second[
        "scoped_cached_diff_sha256"
    ]
    assert first["scoped_cached_diff_command"] == [
        "git",
        "diff",
        "--cached",
        "--binary",
        "HEAD",
        "--",
        "task.py",
    ]
    assert first["scoped_cached_diff_exit_code"] == 0
    assert first["scoped_cached_diff_size_bytes"] > 0
    assert len(str(first["scoped_cached_diff_stderr_sha256"])) == 64
