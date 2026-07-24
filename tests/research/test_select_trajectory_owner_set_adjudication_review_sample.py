from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
from typing import Any

from PIL import Image
import pytest

import scripts.research.analyze_trajectory_owner_set_adjudication_salvage_gate as gate
import scripts.research.select_trajectory_owner_set_adjudication_review_sample as selector


TEST_SOURCE_LABELS = frozenset({"ontology_state", "reviewer_packet"})
ZERO_ENTROPY = bytes(selector.ENTROPY_BYTE_COUNT)
REALIZED_POSSIBLE_COUNT = 1_106


@dataclass(frozen=True)
class StageFixture:
    root: Path
    audit: Path
    binding: selector.StageZeroBinding


@dataclass
class CountingEntropy:
    payload: bytes
    calls: int = 0

    def __call__(self, byte_count: int) -> bytes:
        assert byte_count == selector.ENTROPY_BYTE_COUNT
        self.calls += 1
        return self.payload


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _thaw_tree(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)


def _state(candidate_id: str, token: str, categories: list[str]) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "projected_token_sha256": token * 64,
        "projection_status": "accepted_natural_end",
        "parser_status": "accepted",
        "invalid_before_b16_count": 0,
        "ordered_categories": categories,
        "parser_usable": True,
    }


def _build_stage_fixture(root: Path, audit: Path) -> StageFixture:
    possible_representatives = gate.collapse_exact_token_states(
        [
            _state("sample-00", "a", ["cat", "dog"]),
            _state("sample-01", "b", ["dog", "cat"]),
            _state("sample-02", "c", ["cat"]),
        ]
    )
    impossible_representatives = gate.collapse_exact_token_states(
        [
            _state("sample-00", "d", ["cat"]),
            _state("sample-01", "e", ["cat"]),
            _state("sample-02", "f", ["cat"]),
        ]
    )
    possible_search = gate.search_category_possibility(possible_representatives)
    impossible_search = gate.search_category_possibility(impossible_representatives)
    assert possible_search["possible"] is True
    assert impossible_search["possible"] is False

    categories: list[dict[str, Any]] = []
    possibilities: list[dict[str, Any]] = []
    for value in range(1, selector.EXPECTED_STAGE_ZERO_POPULATION_COUNT + 1):
        image_id = str(value)
        is_possible = value <= REALIZED_POSSIBLE_COUNT
        representatives = (
            possible_representatives if is_possible else impossible_representatives
        )
        search = possible_search if is_possible else impossible_search
        categories.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "image_id": image_id,
                "exact_token_representatives": copy.deepcopy(representatives),
            }
        )
        possibilities.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "image_id": image_id,
                **copy.deepcopy(search),
            }
        )

    possible_ids = [row["image_id"] for row in possibilities if bool(row["possible"])]
    impossible_ids = [
        row["image_id"] for row in possibilities if not bool(row["possible"])
    ]
    _write_jsonl(root / "category-state.jsonl", categories)
    _write_jsonl(root / "possibility-census.jsonl", possibilities)
    possible_pool = gate._pool_payload(possible_ids, pool_role="possible")
    impossible_pool = gate._pool_payload(
        impossible_ids, pool_role="certified_impossible"
    )
    _write_json(root / "possible-pool.json", possible_pool)
    _write_json(root / "impossible-pool.json", impossible_pool)
    summary = {
        "schema_version": gate.SCHEMA_VERSION,
        "population_scope": selector.STAGE_ZERO_POPULATION_SCOPE,
        "population_count": len(categories),
        "possible_image_ids": possible_ids,
        "impossible_image_ids": impossible_ids,
        "possible_pool_count": len(possible_ids),
        "impossible_pool_count": len(impossible_ids),
        "ordered_possible_image_ids_sha256": gate.sha256_json(possible_ids),
        "ordered_impossible_image_ids_sha256": gate.sha256_json(impossible_ids),
        "category_state_sha256": gate.sha256_json(categories),
        "possibility_census_sha256": gate.sha256_json(possibilities),
    }
    _write_json(root / "summary.json", summary)
    snapshot = gate._materialize_source_snapshot(selector.REPOSITORY_ROOT, root)
    outputs = gate._output_inventory(root)
    summary_hash = gate.sha256_json(summary)
    receipt = {
        "schema_version": gate.SCHEMA_VERSION,
        "terminal_status": "completed",
        "population_scope": selector.STAGE_ZERO_POPULATION_SCOPE,
        "analyzer": {
            "path": str(selector.STAGE_ZERO_PRODUCER_PATH),
            "sha256": selector.sha256_file(selector.STAGE_ZERO_PRODUCER_PATH),
        },
        "inputs": {},
        "population": {
            "count": len(categories),
            "ordered_image_ids_sha256": gate.sha256_json(
                [row["image_id"] for row in categories]
            ),
        },
        "possible_pool_count": len(possible_ids),
        "ordered_possible_image_ids_sha256": possible_pool["ordered_image_ids_sha256"],
        "possible_pool_artifact_sha256": outputs["possible-pool.json"]["sha256"],
        "row_counts": {
            "category_state": len(categories),
            "possibility_census": len(possibilities),
            "possible_pool": len(possible_ids),
            "impossible_pool": len(impossible_ids),
        },
        "determinism": {
            "forward_category_state_sha256": gate.sha256_json(categories),
            "reversed_category_state_sha256": gate.sha256_json(categories),
            "forward_possibility_census_sha256": gate.sha256_json(possibilities),
            "reversed_possibility_census_sha256": gate.sha256_json(possibilities),
            "forward_summary_sha256": summary_hash,
            "reversed_summary_sha256": summary_hash,
        },
        "source_snapshot": snapshot,
        "git_identity": gate._stage_zero_git_identity(
            selector.REPOSITORY_ROOT, snapshot["snapshotted_repo_paths"]
        ),
        "outputs": outputs,
        "output_directories": gate._directory_inventory(root),
        "artifact_integrity_checks": {
            "producer_schema_interop_fixture": True,
            "full_certificate_replay": True,
            "exact_inventory": True,
        },
        "publication_contract": {
            "method": "same_parent_staging_then_os_replace",
            "completed_file_mode": "0444",
            "completed_directory_mode": "0555",
        },
    }
    _write_json(root / "receipt.json", receipt)
    _freeze_tree(root)

    inventory = selector._root_inventory(root)
    audit_receipt = {
        "schema_version": selector.STAGE_ZERO_AUDIT_SCHEMA_VERSION,
        "terminal_status": "completed",
        "population_scope": selector.STAGE_ZERO_POPULATION_SCOPE,
        "review_scope": "full_witness_and_certificate_replay",
        "verdict": "approved",
        "stage_zero_receipt_sha256": selector.sha256_file(root / "receipt.json"),
        "stage_zero_root_inventory_sha256": selector.sha256_json(inventory),
        "population_count": len(categories),
        "possible_pool_count": len(possible_ids),
        "impossible_pool_count": len(impossible_ids),
        "ordered_possible_image_ids_sha256": selector.sha256_json(possible_ids),
        "ordered_impossible_image_ids_sha256": selector.sha256_json(impossible_ids),
        "category_state_sha256": summary["category_state_sha256"],
        "possibility_census_sha256": summary["possibility_census_sha256"],
        "replayed_possible_witness_count": len(possible_ids),
        "replayed_impossibility_certificate_count": len(impossible_ids),
        "full_certificate_replay": True,
    }
    _write_json(audit, audit_receipt)
    audit.chmod(0o444)
    receipt_raw = (root / "receipt.json").read_bytes()
    audit_raw = audit.read_bytes()
    binding = selector.StageZeroBinding(
        root.resolve(),
        receipt,
        selector.sha256_bytes(receipt_raw),
        inventory,
        selector.sha256_json(inventory),
        tuple(possible_ids),
        tuple(impossible_ids),
        selector.sha256_json(possible_ids),
        audit.resolve(),
        audit_receipt,
        selector.sha256_bytes(audit_raw),
    )
    return StageFixture(root, audit, binding)


@pytest.fixture(scope="session")
def stage_fixture(tmp_path_factory: pytest.TempPathFactory) -> StageFixture:
    base = tmp_path_factory.mktemp("selector-stage-zero")
    return _build_stage_fixture(base / "stage-zero-v1", base / "stage-zero-audit.json")


def _named_sources(tmp_path: Path) -> tuple[list[selector.NamedSource], frozenset[str]]:
    sources: list[selector.NamedSource] = []
    for label in sorted(TEST_SOURCE_LABELS):
        path = tmp_path / f"{label}.json"
        path.write_text(f'{{"label":"{label}"}}\n', encoding="utf-8")
        sources.append(selector.NamedSource(label, path, selector.sha256_file(path)))
    return sources, TEST_SOURCE_LABELS


def _image_path(tmp_path: Path) -> Path:
    path = tmp_path / "source.png"
    Image.new("RGB", (7, 5), color=(11, 22, 33)).save(path)
    return path


def _member_provider(
    binding: selector.StageZeroBinding, image_path: Path
) -> Callable[[], Mapping[str, Any]]:
    def provide() -> Mapping[str, Any]:
        return selector.build_synthetic_member_manifest(
            binding.possible_image_ids, image_path=image_path
        )

    return provide


def _patch_stage(
    monkeypatch: pytest.MonkeyPatch, binding: selector.StageZeroBinding
) -> None:
    def cached(
        stage_zero_root: Path,
        audit_path: Path,
        *,
        synthetic_test_only: bool = False,
    ) -> selector.StageZeroBinding:
        del stage_zero_root, audit_path, synthetic_test_only
        return binding

    monkeypatch.setattr(selector, "validate_stage_zero_root", cached)


def _assert_immutable_tree(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        assert path.stat().st_mode & 0o222 == 0


def _forbidden_entropy(_: int) -> bytes:
    raise AssertionError("entropy factory must not be invoked")


def _rejected_entropy_bytes(binding: selector.StageZeroBinding) -> bytes:
    permutation_count = selector.falling_factorial(
        len(binding.possible_image_ids), selector.SAMPLE_SIZE
    )
    _, limit = selector.rejection_sampling_parameters(
        entropy_space_size=selector.ENTROPY_SPACE_SIZE,
        permutation_count=permutation_count,
    )
    return limit.to_bytes(selector.ENTROPY_BYTE_COUNT, "big")


def test_exhaustive_small_unranking_is_bijective_and_conditionally_uniform() -> None:
    for population_size in range(1, 7):
        pool = [str(value) for value in range(1, population_size + 1)]
        for sample_size in range(0, min(population_size, 4) + 1):
            permutation_count = selector.falling_factorial(population_size, sample_size)
            observed: set[tuple[str, ...]] = set()
            for rank in range(permutation_count):
                sample, trace = selector.unrank_ordered_sample(pool, sample_size, rank)
                assert selector.rank_ordered_sample(pool, sample) == rank
                assert len(trace) == sample_size
                observed.add(sample)
            assert len(observed) == permutation_count
            finite_space = permutation_count * 3 + permutation_count // 2
            limit = (finite_space // permutation_count) * permutation_count
            residue_counts = [
                sum(value % permutation_count == residue for value in range(limit))
                for residue in range(permutation_count)
            ]
            assert residue_counts == [limit // permutation_count] * permutation_count


def test_exact_64_byte_rejection_sampling_and_trace() -> None:
    pool = [str(value) for value in range(1, 33)]
    accepted = selector.randomization_from_entropy(pool, ZERO_ENTROPY)
    assert accepted.accepted
    assert accepted.entropy_integer_R == 0
    assert accepted.accepted_initial_rank == 0
    assert accepted.ordered_sample == tuple(pool)
    assert [row["selected_image_id"] for row in accepted.unranking_trace] == pool
    permutation_count = selector.falling_factorial(len(pool), selector.SAMPLE_SIZE)
    _, acceptance_limit = selector.rejection_sampling_parameters(
        entropy_space_size=selector.ENTROPY_SPACE_SIZE,
        permutation_count=permutation_count,
    )
    rejected_entropy = acceptance_limit.to_bytes(selector.ENTROPY_BYTE_COUNT, "big")
    rejected = selector.randomization_from_entropy(pool, rejected_entropy)
    assert rejected.permutation_count_M == permutation_count
    assert rejected.entropy_integer_R == acceptance_limit
    assert rejected.acceptance_limit_L == acceptance_limit
    assert not rejected.accepted
    assert rejected.accepted_initial_rank is None
    assert rejected.ordered_sample == ()
    assert rejected.unranking_trace == ()
    with pytest.raises(selector.SelectionError, match="exactly 64"):
        selector.randomization_from_entropy(pool, bytes(63))


def test_exact_dynamic_hypergeometric_cutoffs_include_minus_one() -> None:
    reconnaissance = selector.hypergeometric_design(1_106)
    assert [row["largest_rejection_success_count"] for row in reconnaissance] == [
        0,
        2,
    ]
    assert reconnaissance[0]["boundary_lower_tail_probability"] == {
        "numerator": 121588884438900766472300859,
        "denominator": 7293679104777741335314444853,
        "decimal": (
            "0.016670446107130444911969318492712778628198417856277851760642146326471591974599946"
        ),
    }
    realized = selector.hypergeometric_design(1_622)
    assert [row["largest_rejection_success_count"] for row in realized] == [-1, 0]
    assert realized[0]["boundary_lower_tail_probability"] == {
        "numerator": 0,
        "denominator": 1,
        "decimal": "0",
    }
    assert realized[1]["boundary_lower_tail_probability"] == {
        "numerator": 19563240828196568093877794807059879732777007399191,
        "denominator": 4186075415650136232375709449693104595298645120822791,
        "decimal": (
            "0.0046734085953294312155525420285184051591494689164470968721905591515408140698284927"
        ),
    }
    assert (
        selector.hypergeometric_lower_tail(
            population_size=1_622,
            success_count=selector.NULL_SUCCESS_COUNT,
            sample_size=16,
            cutoff=0,
        )
        > selector.PER_LOOK_ALPHA
    )


def test_full_stage_zero_producer_schema_interop(stage_fixture: StageFixture) -> None:
    snapshot = stage_fixture.binding.receipt["source_snapshot"]
    assert (
        gate._stage_zero_git_identity(
            selector.REPOSITORY_ROOT, snapshot["snapshotted_repo_paths"]
        )
        == stage_fixture.binding.receipt["git_identity"]
    )
    binding = selector.validate_stage_zero_root(
        stage_fixture.root, stage_fixture.audit, synthetic_test_only=True
    )
    assert len(binding.possible_image_ids) == 1_106
    assert len(binding.impossible_image_ids) == 516
    assert binding.possible_image_ids[:3] == ("1", "2", "3")
    assert binding.audit["full_certificate_replay"] is True


def test_operational_read_errors_are_not_reclassified_as_semantic(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        selector._read_json(tmp_path / "missing.json", "missing JSON")
    with pytest.raises(FileNotFoundError):
        selector._read_jsonl(tmp_path / "missing.jsonl", "missing JSONL")


def test_stage_zero_operational_replay_error_propagates(
    monkeypatch: pytest.MonkeyPatch, stage_fixture: StageFixture
) -> None:
    def fail_replay(**_: object) -> None:
        raise OSError("injected Stage-Zero replay read failure")

    monkeypatch.setattr(gate, "_validate_success_receipt_readback", fail_replay)
    with pytest.raises(OSError, match="Stage-Zero replay read failure"):
        selector.validate_stage_zero_root(
            stage_fixture.root, stage_fixture.audit, synthetic_test_only=True
        )


def test_stage_zero_rejects_minimal_root_extra_inventory_and_minimal_audit(
    tmp_path: Path, stage_fixture: StageFixture
) -> None:
    minimal = tmp_path / "minimal"
    minimal.mkdir()
    shutil.copyfile(stage_fixture.root / "receipt.json", minimal / "receipt.json")
    _freeze_tree(minimal)
    with pytest.raises(selector.SelectionError):
        selector.validate_stage_zero_root(
            minimal, stage_fixture.audit, synthetic_test_only=True
        )

    extra = tmp_path / "extra"
    shutil.copytree(stage_fixture.root, extra)
    _thaw_tree(extra)
    (extra / "undeclared.txt").write_text("extra\n", encoding="utf-8")
    _freeze_tree(extra)
    with pytest.raises(selector.SelectionError):
        selector.validate_stage_zero_root(
            extra, stage_fixture.audit, synthetic_test_only=True
        )

    minimal_audit = tmp_path / "minimal-audit.json"
    _write_json(minimal_audit, {"verdict": "approved"})
    minimal_audit.chmod(0o444)
    with pytest.raises(selector.SelectionError, match="key inventory differs"):
        selector.validate_stage_zero_root(
            stage_fixture.root, minimal_audit, synthetic_test_only=True
        )

    extended_audit = tmp_path / "extended-audit.json"
    extended = dict(stage_fixture.binding.audit)
    extended["unbound_reviewer_note"] = "not in the closed receipt schema"
    _write_json(extended_audit, extended)
    extended_audit.chmod(0o444)
    with pytest.raises(selector.SelectionError, match="key inventory differs"):
        selector.validate_stage_zero_root(
            stage_fixture.root, extended_audit, synthetic_test_only=True
        )


def test_frozen_source_manifest_is_exact_and_detects_drift(tmp_path: Path) -> None:
    sources, labels = _named_sources(tmp_path)
    manifest = selector.build_frozen_source_manifest(
        sources, expected_named_labels=labels
    )
    selector.revalidate_frozen_source_manifest(manifest)
    ordered_labels = manifest["ordered_labels"]
    assert isinstance(ordered_labels, list)
    assert set(ordered_labels) == set(labels) | set(selector.AUTO_SOURCE_LABELS)

    with pytest.raises(selector.SelectionError, match="missing"):
        selector.build_frozen_source_manifest(
            sources[:-1], expected_named_labels=labels
        )
    extra_path = tmp_path / "extra.json"
    extra_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="extra"):
        selector.build_frozen_source_manifest(
            [
                *sources,
                selector.NamedSource(
                    "extra", extra_path, selector.sha256_file(extra_path)
                ),
            ],
            expected_named_labels=labels,
        )
    with pytest.raises(selector.SelectionError, match="duplicate a label"):
        selector.build_frozen_source_manifest(
            [sources[0], sources[0]], expected_named_labels=labels
        )
    duplicate_path = sources[0].path
    duplicate_sources = [
        sources[0],
        selector.NamedSource(
            sources[1].label,
            duplicate_path,
            selector.sha256_file(duplicate_path),
        ),
    ]
    with pytest.raises(selector.SelectionError, match="canonical path"):
        selector.build_frozen_source_manifest(
            duplicate_sources, expected_named_labels=labels
        )
    sources[0].path.write_text("drift\n", encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="drifted"):
        selector.revalidate_frozen_source_manifest(manifest)


def test_moving_upstream_sources_are_explicit_runtime_bindings_not_constants() -> None:
    assert {
        "b16_drop_chronology_helper",
        "b16_drop_chronology_impact_review",
    } <= selector.PRODUCTION_NAMED_SOURCE_LABELS
    selector_source = Path(selector.__file__).read_text(encoding="utf-8")
    for runtime_hash in (
        "a8279d745bb923d58eb33f1911611978897e9da94da00edd174453133178ff3d",
        "cf7b498b1aa643d14b54d144eb6d1444b5a5ef9b9b92e852d000efe685c470d5",
        "d7eb0ddb96a1feea3867c795deb392581cbb62f9d7e37add2e86dae9cfd65385",
    ):
        assert runtime_hash not in selector_source


def test_complete_member_manifest_hashes_bytes_before_decode_and_detects_drift(
    tmp_path: Path, stage_fixture: StageFixture
) -> None:
    image_path = _image_path(tmp_path)
    manifest = selector.build_synthetic_member_manifest(
        stage_fixture.binding.possible_image_ids, image_path=image_path
    )
    validated = selector.validate_member_manifest(
        manifest, stage_fixture.binding.possible_image_ids
    )
    assert validated["count"] == 1_106
    assert len(validated["members"]) == 1_106
    image_path.write_bytes(b"not an image")
    with pytest.raises(selector.SelectionError, match="source hash differs"):
        selector.validate_member_manifest(
            manifest, stage_fixture.binding.possible_image_ids
        )


def test_official_owner_hash_semantics_are_canonical_and_strict() -> None:
    owners = [
        {
            "owner_id": "7:2",
            "category": "Traffic Light",
            "bbox": [1, 2, 3, 4],
            "category_id": selector.COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME[
                "traffic light"
            ],
            "annotation_index": 1,
            "image_id": "7",
        },
        {
            "owner_id": "7:1",
            "category": "person",
            "bbox": [0, 1, 2, 3],
            "category_id": selector.COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME["person"],
            "annotation_index": 0,
            "image_id": "7",
        },
    ]
    canonical = selector._canonical_official_owner_semantics(
        owners,
        image_id="7",
    )
    assert [row["owner_id"] for row in canonical] == ["7:1", "7:2"]
    assert canonical[1]["normalized_category_name"] == "traffic light"
    assert canonical[1]["source_category_name"] == "Traffic Light"
    assert canonical[1]["annotation_index"] == 1
    assert canonical[1]["image_id"] == "7"
    assert canonical[1]["source_canvas_box_xyxy"] == [1.0, 2.0, 3.0, 4.0]

    changed = copy.deepcopy(owners)
    changed[0]["annotation_index"] = 2
    changed_canonical = selector._canonical_official_owner_semantics(
        changed, image_id="7"
    )
    assert selector.sha256_json(changed_canonical) != selector.sha256_json(canonical)

    extra = copy.deepcopy(owners)
    extra[0]["unbound_field"] = "must be rejected"
    with pytest.raises(selector.SelectionError, match="key inventory differs"):
        selector._canonical_official_owner_semantics(extra, image_id="7")

    wrong_category_id = copy.deepcopy(owners)
    wrong_category_id[0]["category_id"] = 1
    with pytest.raises(selector.SelectionError, match="source category ID drifted"):
        selector._canonical_official_owner_semantics(wrong_category_id, image_id="7")

    duplicate = copy.deepcopy(owners)
    duplicate[0]["owner_id"] = "7:1"
    with pytest.raises(selector.SelectionError, match="IDs drifted"):
        selector._canonical_official_owner_semantics(
            duplicate,
            image_id="7",
        )


@pytest.mark.parametrize(
    ("fault_point", "expected_entropy_calls"),
    [
        ("claim_root_created", 0),
        ("claim_file_fsynced", 0),
        ("claim_readback_validated", 0),
        ("entropy_acquired", 1),
        ("entropy_file_created", 1),
        ("entropy_file_fsynced", 1),
    ],
)
def test_incomplete_claim_or_entropy_is_terminal_unknown_without_redraw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
    fault_point: str,
    expected_entropy_calls: int,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(ZERO_ENTROPY)
    output = tmp_path / "selection-v1"

    def interrupt(stage: str) -> None:
        if stage == fault_point:
            raise KeyboardInterrupt(stage)

    with pytest.raises(KeyboardInterrupt):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=provider,
            entropy_factory=entropy,
            fault_injector=interrupt,
            synthetic_test_only=True,
        )
    assert entropy.calls == expected_entropy_calls
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=_forbidden_entropy,
        synthetic_test_only=True,
    )
    assert receipt["terminal_status"] == "entropy_persistence_unknown"
    assert receipt["selection_published"] is False
    assert set(path.name for path in output.iterdir()) == {"receipt.json"}
    assert not (output / "selection.json").exists()
    _assert_immutable_tree(output)
    _assert_immutable_tree(selector.journal_root_for(output))


def test_complete_entropy_journal_resumes_without_second_factory_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(ZERO_ENTROPY)
    output = tmp_path / "selection-v1"

    def interrupt(stage: str) -> None:
        if stage == "entropy_readback_validated":
            raise KeyboardInterrupt(stage)

    with pytest.raises(KeyboardInterrupt):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=provider,
            entropy_factory=entropy,
            fault_injector=interrupt,
            synthetic_test_only=True,
        )
    assert entropy.calls == 1
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=_forbidden_entropy,
        synthetic_test_only=True,
    )
    assert receipt["terminal_status"] == "completed"
    assert receipt["entropy"]["byte_count"] == 64


def test_transient_postentropy_input_read_failure_is_mechanical_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(ZERO_ENTROPY)
    output = tmp_path / "selection-v1"

    def interrupt(stage: str) -> None:
        if stage == "entropy_readback_validated":
            raise KeyboardInterrupt(stage)

    with pytest.raises(KeyboardInterrupt, match="entropy_readback_validated"):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=provider,
            entropy_factory=entropy,
            fault_injector=interrupt,
            synthetic_test_only=True,
        )
    assert entropy.calls == 1

    original_revalidate = selector.revalidate_frozen_source_manifest

    def fail_operational_read(_: Mapping[str, Any]) -> None:
        raise OSError("injected transient source read failure")

    monkeypatch.setattr(
        selector, "revalidate_frozen_source_manifest", fail_operational_read
    )
    with pytest.raises(selector.MechanicalSelectionError, match="mechanical"):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=provider,
            entropy_factory=_forbidden_entropy,
            synthetic_test_only=True,
        )
    journal = selector.journal_root_for(output)
    assert not (journal / "terminal.json").exists()
    assert not output.exists()

    monkeypatch.setattr(
        selector, "revalidate_frozen_source_manifest", original_revalidate
    )
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=_forbidden_entropy,
        synthetic_test_only=True,
    )
    assert receipt["terminal_status"] == "completed"
    assert entropy.calls == 1


def test_mechanical_materialization_failure_resumes_same_entropy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(ZERO_ENTROPY)
    output = tmp_path / "selection-v1"

    def fail(stage: str) -> None:
        if stage == "materialization_started":
            raise OSError("injected materialization failure")

    with pytest.raises(selector.MechanicalSelectionError):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=provider,
            entropy_factory=entropy,
            fault_injector=fail,
            synthetic_test_only=True,
        )
    assert entropy.calls == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".selection-v1.staging-*"))
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=_forbidden_entropy,
        synthetic_test_only=True,
    )
    assert receipt["terminal_status"] == "completed"
    assert not list(tmp_path.glob("*failed*"))


def test_invalid_postrename_root_is_quarantined_then_resumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(ZERO_ENTROPY)
    output = tmp_path / "selection-v1"

    def fail(stage: str) -> None:
        if stage == "post_rename":
            raise OSError("injected post-rename failure")

    with pytest.raises(selector.MechanicalSelectionError):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=provider,
            entropy_factory=entropy,
            fault_injector=fail,
            synthetic_test_only=True,
        )
    assert entropy.calls == 1
    assert not output.exists()
    quarantines = list(tmp_path.glob("selection-v1.invalid-*"))
    assert len(quarantines) == 1
    _assert_immutable_tree(quarantines[0])
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=_forbidden_entropy,
        synthetic_test_only=True,
    )
    assert receipt["terminal_status"] == "completed"


def test_entropy_rejection_is_terminal_and_exposes_no_selected_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(_rejected_entropy_bytes(stage_fixture.binding))
    output = tmp_path / "selection-v1"
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=entropy,
        synthetic_test_only=True,
    )
    assert entropy.calls == 1
    assert receipt["terminal_status"] == "entropy_rejected"
    assert receipt["randomization"]["accepted"] is False
    assert receipt["randomization"]["accepted_initial_rank"] is None
    assert receipt["randomization"]["unranking_trace"] == []
    assert receipt["randomization"]["null_success_count_K"] == 248
    assert receipt["randomization"]["per_look_alpha"] == {
        "numerator": 1,
        "denominator": 40,
        "decimal": "0.025",
    }
    assert receipt["randomization"]["cumulative_sample_sizes"] == [16, 32]
    assert receipt["randomization"]["hypergeometric_design"] == (
        selector.hypergeometric_design(REALIZED_POSSIBLE_COUNT)
    )
    assert not any("image_ids" in key for key in receipt)
    assert not (output / "selection.json").exists()
    replay = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=_forbidden_entropy,
        synthetic_test_only=True,
    )
    assert replay == receipt


@pytest.mark.parametrize(
    "fault_point", ["terminal_pending_created", "terminal_renamed"]
)
def test_terminal_marker_interruption_recovers_without_entropy_redraw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
    fault_point: str,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(_rejected_entropy_bytes(stage_fixture.binding))
    output = tmp_path / "selection-v1"

    def interrupt(stage: str) -> None:
        if stage == fault_point:
            raise KeyboardInterrupt(stage)

    with pytest.raises(KeyboardInterrupt, match=fault_point):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=provider,
            entropy_factory=entropy,
            fault_injector=interrupt,
            synthetic_test_only=True,
        )
    assert entropy.calls == 1
    assert not output.exists()

    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=_forbidden_entropy,
        synthetic_test_only=True,
    )
    assert receipt["terminal_status"] == "entropy_rejected"
    journal = selector.journal_root_for(output)
    assert {path.name for path in journal.iterdir()} == {
        "claim.json",
        "entropy.bin",
        "entropy-receipt.json",
        "terminal.json",
    }
    _assert_immutable_tree(journal)
    _assert_immutable_tree(output)


def test_terminal_output_read_os_error_is_mechanical_without_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(_rejected_entropy_bytes(stage_fixture.binding))
    output = tmp_path / "selection-v1"
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=entropy,
        synthetic_test_only=True,
    )
    assert entropy.calls == 1
    original_validate = selector._validate_terminal_output

    def fail_read(_: Path, __: Mapping[str, Any]) -> dict[str, Any]:
        raise OSError("injected terminal-output read failure")

    monkeypatch.setattr(selector, "_validate_terminal_output", fail_read)
    with pytest.raises(selector.MechanicalSelectionError, match="terminal-output"):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=provider,
            entropy_factory=_forbidden_entropy,
            synthetic_test_only=True,
        )
    assert output.exists()
    assert not list(tmp_path.glob("selection-v1.invalid-*"))

    monkeypatch.setattr(selector, "_validate_terminal_output", original_validate)
    replay = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=_forbidden_entropy,
        synthetic_test_only=True,
    )
    assert replay == receipt


def test_member_manifest_drift_after_entropy_voids_without_redraw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    base = selector.build_synthetic_member_manifest(
        stage_fixture.binding.possible_image_ids,
        image_path=_image_path(tmp_path),
    )
    changed = copy.deepcopy(base)
    changed_members = changed["members"]
    assert isinstance(changed_members, list)
    first_member = changed_members[0]
    assert isinstance(first_member, dict)
    first_member["official_owner_record_sha256"] = "f" * 64
    changed["members_sha256"] = selector.sha256_json(changed_members)
    calls = 0

    def provider() -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        return base if calls == 1 else changed

    entropy = CountingEntropy(ZERO_ENTROPY)
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=tmp_path / "selection-v1",
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=entropy,
        synthetic_test_only=True,
    )
    assert entropy.calls == 1
    assert calls == 2
    assert receipt["terminal_status"] == "void"
    assert "member manifest drifted" in receipt["reason"]


def test_source_drift_after_entropy_voids_without_redraw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(ZERO_ENTROPY)

    def drift(stage: str) -> None:
        if stage == "entropy_readback_validated":
            sources[0].path.write_text("semantic drift\n", encoding="utf-8")

    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=tmp_path / "selection-v1",
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=entropy,
        fault_injector=drift,
        synthetic_test_only=True,
    )
    assert entropy.calls == 1
    assert receipt["terminal_status"] == "void"
    assert "source" in receipt["reason"]


def test_stage_binding_drift_after_entropy_voids_without_redraw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    calls = 0

    def changing(
        stage_zero_root: Path,
        audit_path: Path,
        *,
        synthetic_test_only: bool = False,
    ) -> selector.StageZeroBinding:
        nonlocal calls
        del stage_zero_root, audit_path, synthetic_test_only
        calls += 1
        if calls == 1:
            return stage_fixture.binding
        return replace(stage_fixture.binding, audit_sha256="f" * 64)

    monkeypatch.setattr(selector, "validate_stage_zero_root", changing)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(ZERO_ENTROPY)
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=tmp_path / "selection-v1",
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=entropy,
        synthetic_test_only=True,
    )
    assert entropy.calls == 1
    assert calls == 2
    assert receipt["terminal_status"] == "void"
    assert "Stage-Zero binding drifted" in receipt["reason"]


def test_success_publication_has_exact_receipts_views_bindings_and_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(ZERO_ENTROPY)
    output = tmp_path / "selection-v1"
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=entropy,
        synthetic_test_only=True,
    )
    assert entropy.calls == 1
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert selection["terminal_status"] == "completed"
    assert len(selection["look_one_image_ids"]) == 16
    assert len(selection["look_two_additional_image_ids"]) == 16
    assert (
        selection["look_one_image_ids"] + selection["look_two_additional_image_ids"]
        == selection["look_two_cumulative_image_ids"]
    )
    assert len(set(selection["look_two_cumulative_image_ids"])) == 32
    assert len(selection["entropy"]["unranking_trace"]) == 32
    assert receipt["frozen_source_manifest"] == selection["frozen_source_manifest"]
    assert receipt["stage_zero_binding"] == selection["stage_zero_binding"]
    assert (
        receipt["possible_pool_member_manifest"]
        == selection["possible_pool_member_manifest"]
    )
    inventory = selector._root_inventory(output)
    inventory.pop("receipt.json")
    assert receipt["outputs"] == inventory
    claim = json.loads(
        (selector.journal_root_for(output) / "claim.json").read_text(encoding="utf-8")
    )
    assert (
        claim["frozen_source_manifest_sha256"]
        == receipt["bindings"]["frozen_source_manifest_sha256"]
    )
    assert (
        claim["stage_zero_binding_sha256"]
        == receipt["bindings"]["stage_zero_binding_sha256"]
    )
    assert (
        claim["member_manifest_sha256"]
        == receipt["bindings"]["possible_pool_member_manifest_sha256"]
    )
    assert claim["null_success_count_K"] == selection["null_success_count_K"] == 248
    assert (
        claim["per_look_alpha"]
        == selection["per_look_alpha"]
        == receipt["randomization"]["per_look_alpha"]
    )
    assert (
        claim["hypergeometric_design"]
        == selection["hypergeometric_design"]
        == receipt["randomization"]["hypergeometric_design"]
    )
    assert not list(tmp_path.glob(".selection-v1.staging-*"))
    _assert_immutable_tree(output)
    _assert_immutable_tree(selector.journal_root_for(output))
    replay = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=_forbidden_entropy,
        synthetic_test_only=True,
    )
    assert replay == receipt


def test_completed_output_cannot_be_rebound_away_from_terminal_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    output = tmp_path / "selection-v1"
    selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=CountingEntropy(ZERO_ENTROPY),
        synthetic_test_only=True,
    )
    _thaw_tree(output)
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    selection["look_one_image_ids"][0] = "999999"
    receipt["randomization"]["look_one_image_ids"][0] = "999999"
    _write_json(output / "selection.json", selection)
    inventory = selector._root_inventory(output)
    inventory.pop("receipt.json")
    receipt["outputs"] = inventory
    _write_json(output / "receipt.json", receipt)
    _freeze_tree(output)
    with pytest.raises(selector.SelectionError, match="canonical journal"):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=provider,
            entropy_factory=_forbidden_entropy,
            synthetic_test_only=True,
        )


def test_missing_named_source_and_minimal_audit_fail_before_claim_or_entropy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    output = tmp_path / "missing-source-selection"
    with pytest.raises(selector.SelectionError, match="missing"):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=sources[:-1],
            expected_named_labels=labels,
            member_manifest_provider=_member_provider(
                stage_fixture.binding, _image_path(tmp_path)
            ),
            entropy_factory=_forbidden_entropy,
            synthetic_test_only=True,
        )
    assert not output.exists()
    assert not selector.journal_root_for(output).exists()

    monkeypatch.undo()
    minimal_audit = tmp_path / "minimal-audit.json"
    _write_json(minimal_audit, {"verdict": "approved"})
    minimal_audit.chmod(0o444)
    output = tmp_path / "minimal-audit-selection"
    with pytest.raises(selector.SelectionError):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=minimal_audit,
            output_root=output,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=_member_provider(
                stage_fixture.binding, _image_path(tmp_path)
            ),
            entropy_factory=_forbidden_entropy,
            synthetic_test_only=True,
        )
    assert not selector.journal_root_for(output).exists()


def test_production_mode_rejects_injected_declarative_member_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    output = tmp_path / "selection-v1"
    with pytest.raises(selector.SelectionError, match="output root is not canonical"):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=_member_provider(
                stage_fixture.binding, _image_path(tmp_path)
            ),
            entropy_factory=_forbidden_entropy,
            synthetic_test_only=False,
        )
    assert not selector.journal_root_for(output).exists()
    with pytest.raises(selector.SelectionError, match="reconstructed"):
        selector._current_bound_inputs(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            named_sources=sources,
            expected_named_labels=labels,
            member_manifest_provider=_member_provider(
                stage_fixture.binding, _image_path(tmp_path)
            ),
            synthetic_test_only=False,
        )


def test_synthetic_mode_cannot_target_canonical_production_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    output = tmp_path / "canonical-selection-v1"
    monkeypatch.setattr(selector, "FROZEN_OUTPUT_ROOT", output)
    with pytest.raises(selector.SelectionError, match="synthetic test mode"):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=[],
            member_manifest_provider=None,
            entropy_factory=_forbidden_entropy,
            synthetic_test_only=True,
        )
    assert not selector.journal_root_for(output).exists()


def test_production_mode_rejects_non_os_entropy_before_journal_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    output = tmp_path / "canonical-selection-v1"
    _, labels = _named_sources(tmp_path)
    monkeypatch.setattr(selector, "FROZEN_OUTPUT_ROOT", output)
    monkeypatch.setattr(selector, "FROZEN_STAGE_ZERO_ROOT", stage_fixture.root)
    monkeypatch.setattr(selector, "PRODUCTION_NAMED_SOURCE_LABELS", labels)
    with pytest.raises(selector.SelectionError, match="secrets.token_bytes"):
        selector.execute_selection(
            stage_zero_root=stage_fixture.root,
            stage_zero_audit=stage_fixture.audit,
            output_root=output,
            named_sources=[],
            expected_named_labels=labels,
            member_manifest_provider=None,
            entropy_factory=_forbidden_entropy,
            synthetic_test_only=False,
        )
    assert not selector.journal_root_for(output).exists()


def test_invalid_entropy_factory_result_terminalizes_unknown_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    entropy = CountingEntropy(bytes(63))
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=tmp_path / "selection-v1",
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=entropy,
        synthetic_test_only=True,
    )
    assert entropy.calls == 1
    assert receipt["terminal_status"] == "entropy_persistence_unknown"


def test_entropy_source_os_failure_terminalizes_unknown_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_fixture: StageFixture,
) -> None:
    _patch_stage(monkeypatch, stage_fixture.binding)
    sources, labels = _named_sources(tmp_path)
    provider = _member_provider(stage_fixture.binding, _image_path(tmp_path))
    calls = 0

    def failed_entropy(byte_count: int) -> bytes:
        nonlocal calls
        assert byte_count == selector.ENTROPY_BYTE_COUNT
        calls += 1
        raise OSError("injected entropy source failure")

    output = tmp_path / "selection-v1"
    receipt = selector.execute_selection(
        stage_zero_root=stage_fixture.root,
        stage_zero_audit=stage_fixture.audit,
        output_root=output,
        named_sources=sources,
        expected_named_labels=labels,
        member_manifest_provider=provider,
        entropy_factory=failed_entropy,
        synthetic_test_only=True,
    )
    assert calls == 1
    assert receipt["terminal_status"] == "entropy_persistence_unknown"
    assert "persistence proof failed" in receipt["reason"]
    assert set(path.name for path in output.iterdir()) == {"receipt.json"}
