from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "research" / "run_prefix_row_factorial_likelihood.py"
_SPEC = importlib.util.spec_from_file_location("prefix_row_factorial_likelihood", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _row(description: list[int], start: int = 1) -> list[int]:
    return [
        _MODULE.OBJECT_REF_START,
        *description,
        _MODULE.OBJECT_REF_END,
        _MODULE.BOX_START,
        *[_MODULE.COORDINATE_TOKEN_START + start + index for index in range(4)],
        _MODULE.BOX_END,
    ]


def _write_bundle(path: Path, *, prompt: list[int], generated: list[int], image_id: str = "img") -> None:
    path.write_text(
        json.dumps(
            {
                "decode_result": {
                    "prompt_token_ids": prompt,
                    "generated_token_ids": generated,
                },
                "execution_evidence": {"image_id": image_id},
            }
        ),
        encoding="utf-8",
    )


def _fixture(
    *, fixture_id: str, path: Path, prompt: list[int], row: list[int], kind: str = "native_contiguous", support_state_id: str = "P0", owner: str | None = None
) -> dict[str, object]:
    _write_bundle(path, prompt=prompt, generated=row)
    return {
        "fixture_id": fixture_id,
        "source_bundle": str(path),
        "source_bundle_sha256": _MODULE.sha256_file(path),
        "source_token_span": [0, len(row)],
        "token_ids_sha256": _MODULE.sha256_json(row),
        "owner": owner or fixture_id,
        "category": "person",
        "description": "person",
        "geometry": [1, 2, 3, 4],
        "natural_support": {"state_kind": kind, "support_state_id": support_state_id, "source": "unit-test", "count": 1, "policy": "native"},
    }


def _candidate(*, candidate_id: str, path: Path, prompt: list[int], row: list[int], support: str, owner: str = "candidate") -> dict[str, object]:
    _write_bundle(path, prompt=prompt, generated=row)
    return {
        "candidate_id": candidate_id,
        "support_state_id": support,
        "source_bundle": str(path),
        "source_bundle_sha256": _MODULE.sha256_file(path),
        "source_token_span": [0, len(row)],
        "token_ids_sha256": _MODULE.sha256_json(row),
        "owner": owner,
        "category": "person",
        "role": "remaining",
        "description": "person",
        "geometry": [5, 6, 7, 8],
        "natural_support": {"state_kind": "native_contiguous", "support_state_id": support, "source": "unit-test", "count": 1, "policy": "native"},
    }


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = [11, 12]
    parent_path = tmp_path / "parent.json"
    _write_bundle(parent_path, prompt=base, generated=[])
    a = _row([21], start=1)
    b = _row([31], start=10)
    c = _row([41], start=20)
    fixture_a = _fixture(fixture_id="A", path=tmp_path / "a.json", prompt=base, row=a, support_state_id="P0")
    fixture_b = _fixture(fixture_id="B", path=tmp_path / "b.json", prompt=base + a, row=b, support_state_id="PA")
    candidates = [
        _candidate(candidate_id="c-P0", path=tmp_path / "c-p0.json", prompt=base, row=c, support="P0", owner="candidate"),
        _candidate(candidate_id="c-PA", path=tmp_path / "c-pa.json", prompt=base + a, row=c, support="PA", owner="candidate"),
        _candidate(candidate_id="c-PB", path=tmp_path / "c-pb.json", prompt=base + b, row=c, support="PB", owner="candidate"),
        _candidate(candidate_id="c-PAB", path=tmp_path / "c-pab.json", prompt=base + a + b, row=c, support="PAB", owner="candidate"),
    ]
    manifest: dict[str, object] = {
        "schema_version": _MODULE.MANIFEST_SCHEMA_VERSION,
        "image_id": "img",
        "parent": {
            "source_bundle": str(parent_path),
            "source_bundle_sha256": _MODULE.sha256_file(parent_path),
            "prefix_token_count": 0,
            "prompt_token_ids_sha256": _MODULE.sha256_json(base),
            "prefix_token_ids_sha256": _MODULE.sha256_json([]),
            "reconstructed_prompt_token_ids_sha256": _MODULE.sha256_json(base),
        },
        "row_fixtures": [fixture_a, fixture_b],
        "states": [
            {"state_id": "P0", "state_kind": "native_contiguous", "row_fixture_ids": []},
            {"state_id": "PA", "state_kind": "native_contiguous", "row_fixture_ids": ["A"]},
            {"state_id": "PB", "state_kind": "forced_replay", "row_fixture_ids": ["B"]},
            {"state_id": "PAB", "state_kind": "native_contiguous", "row_fixture_ids": ["A", "B"]},
        ],
        "candidate_variants": candidates,
        "pairwise_contrasts": [{"contrast_id": "A_to_AB", "before_state_id": "PA", "after_state_id": "PAB", "candidate_ids": ["c-PAB"]}],
        "two_by_two_interaction": {"state_ids": {"P0": "P0", "PA": "PA", "PB": "PB", "PAB": "PAB"}, "candidate_ids": ["c-PAB"]},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_native_pa_and_pab_are_accepted(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    normalized = _MODULE.validate_factorial_manifest(manifest, manifest_path=path)
    assert normalized["states"]["PA"]["state_kind"] == "native_contiguous"
    assert normalized["states"]["PAB"]["row_fixture_ids"] == ["A", "B"]
    assert normalized["states"]["PAB"]["reconstructed_prompt_token_ids"][-len(_row([31], 10)) :] == _row([31], 10)


def test_forced_pb_is_accepted_and_labeled(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["states"][2]["state_kind"] = "forced_replay"  # type: ignore[index]
    normalized = _MODULE.validate_factorial_manifest(manifest, manifest_path=path)
    assert normalized["states"]["PB"]["state_kind"] == "forced_replay"
    assert normalized["states"]["PB"]["row_fixture_ids"] == ["B"]
    forced_evidence = _MODULE.comparison_evidence(normalized, ["PB", "PAB"])
    assert forced_evidence["evidence_scope"] == "descriptive_only"
    assert forced_evidence["contains_forced_replay"] is True
    assert "forced_replay state PB" in forced_evidence["warning"]
    native_evidence = _MODULE.comparison_evidence(normalized, ["PA", "PAB"])
    assert native_evidence["evidence_scope"] == "conclusion_eligible"
    assert native_evidence["contains_forced_replay"] is False


def test_falsely_declared_native_pb_is_rejected(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["states"][2]["state_kind"] = "native_contiguous"  # type: ignore[index]
    with pytest.raises(ValueError, match="PB must be forced_replay"):
        _MODULE.validate_factorial_manifest(manifest, manifest_path=path)


def test_factorial_rejects_wrong_pa_or_pab_fixture_order(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["states"][1]["row_fixture_ids"] = ["A", "B"]  # type: ignore[index]
    with pytest.raises(ValueError, match="PA must contain exactly one row fixture"):
        _MODULE.validate_factorial_manifest(manifest, manifest_path=path)
    path, manifest = _manifest(tmp_path / "reversed")
    manifest["states"][3]["row_fixture_ids"] = ["B", "A"]  # type: ignore[index]
    with pytest.raises(ValueError, match="PAB must contain the same A fixture then B fixture"):
        _MODULE.validate_factorial_manifest(manifest, manifest_path=path)


def test_candidate_support_prompt_mismatch_is_rejected(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    candidate = next(item for item in manifest["candidate_variants"] if item["candidate_id"] == "c-PA")  # type: ignore[index]
    source = Path(candidate["source_bundle"])
    row = _row([41], start=20)
    _write_bundle(source, prompt=[11, 12], generated=row)
    candidate["source_bundle_sha256"] = _MODULE.sha256_file(source)  # type: ignore[index]
    with pytest.raises(ValueError, match="c-PA source prompt does not equal support state PA"):
        _MODULE.validate_factorial_manifest(manifest, manifest_path=path)


def _logits(row: list[int], boundary: int, shift: float) -> torch.Tensor:
    vocab = 152670 + 20
    value = torch.zeros((boundary + len(row), vocab), dtype=torch.float32)
    for index, token in enumerate(row):
        value[boundary + index - 1, token] = shift
    return value


def test_exact_candidate_can_be_scored_across_states_without_losing_support_labels(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    normalized = _MODULE.validate_factorial_manifest(manifest, manifest_path=path)
    candidate = next(item for item in normalized["candidate_variants"] if item["candidate_id"] == "c-PAB")
    emitted = {}
    for state_id, state in normalized["states"].items():
        raw = _MODULE.score_token_logits(_logits(candidate["token_ids"], len(state["reconstructed_prompt_token_ids"]), 1.0), boundary_length=len(state["reconstructed_prompt_token_ids"]), row_tokens=candidate["token_ids"])
        emitted[state_id] = _MODULE.emit_factorial_score(raw, candidate, state)
    assert set(emitted) == set(_MODULE.STATE_IDS)
    assert all(item["candidate_support_state_id"] == "PAB" for item in emitted.values())
    assert emitted["P0"]["state_id"] == "P0"
    assert emitted["PAB"]["state_kind"] == "native_contiguous"


def test_pairwise_delta_and_two_by_two_interaction_arithmetic() -> None:
    def score(value: float) -> dict[str, object]:
        return {phase: {"sum": value, "mean": value / 2, "count": 2} for phase in _MODULE.PHASES} | {"token_ids_sha256": "x", "token_count": 2}

    before = score(1.0)
    after = score(3.0)
    delta = _MODULE.paired_state_delta(before, after)
    assert delta["full_row"]["sum_delta"] == 2.0
    interaction = _MODULE.factorial_interaction({"P0": score(0.0), "PA": score(1.0), "PB": score(2.0), "PAB": score(6.0)})
    assert interaction["full_row"]["sum_interaction"] == 3.0


def test_hash_and_span_errors_are_rejected(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    candidate = manifest["candidate_variants"][0]  # type: ignore[index]
    candidate["token_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="token_ids_sha256 mismatch"):
        _MODULE.validate_factorial_manifest(manifest, manifest_path=path)
    path, manifest = _manifest(tmp_path / "span")
    candidate = manifest["candidate_variants"][0]  # type: ignore[index]
    candidate["source_token_span"] = [1, 2]
    with pytest.raises(ValueError, match="source span is not one complete natural row"):
        _MODULE.validate_factorial_manifest(manifest, manifest_path=path)
