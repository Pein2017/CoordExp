from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.research.build_human13_k_union_manifest import load_manifest
from scripts.research.build_human13_row_contrast_successor import (
    COORD_TOKEN_END_EXCLUSIVE,
    COORD_TOKEN_START,
    PriorOutputSpec,
    build_successor_ledger,
    canonical_write,
    load_ledger,
)


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-12-human13-k-union-to-greedy-overfit-screen"
)
MANIFEST = ROOT / "manifest/human13-k-union-manifest.json"
SUCCESSOR = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-13-human13-missing-arms-successor/matrix/v3/"
    "human13-missing-arms-v3/a4/eval"
)


def test_direct_script_entry_can_resolve_repo_imports() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/build_human13_row_contrast_successor.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd="/tmp",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _prior(milestone: int) -> PriorOutputSpec:
    directory = SUCCESSOR / f"milestone-{milestone}"
    return PriorOutputSpec(
        arm_id="A4",
        milestone=milestone,
        outputs_path=directory / f"A4.milestone-{milestone}.jsonl",
        receipt_path=directory / "A4.receipt.json",
    )


def test_manifest_events_recover_complete_rows_and_owner_state() -> None:
    manifest = load_manifest(MANIFEST)
    ledger = build_successor_ledger(manifest, manifest_path=MANIFEST)

    assert ledger.schema_version == "human13_row_contrast_successor.v1"
    assert ledger.manifest_sha256 == (
        "a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb"
    )
    assert len(ledger.events) == 12
    assert len(ledger.g_watch_rows) == 173
    assert {row.stratum for row in ledger.positive_rows} == {"G", "H"}
    assert (
        len({row.owner_id for row in ledger.positive_rows if row.stratum == "H"}) == 73
    )

    for event in ledger.events:
        assert event.source_kind == "manifest"
        assert len(event.duplicate_row.coordinate_offsets) == 4
        assert (
            event.duplicate_row.coordinate_offsets[-1]
            == len(event.duplicate_row.token_ids) - 2
        )
        assert all(
            COORD_TOKEN_START
            <= event.duplicate_row.token_ids[index]
            < COORD_TOKEN_END_EXCLUSIVE
            for index in event.duplicate_row.coordinate_offsets
        )
        assert not set(event.covered_owner_ids) & set(event.uncovered_owner_ids)
        assert set(event.covered_owner_ids) | set(event.uncovered_owner_ids) == set(
            event.target_owner_ids
        )
        assert tuple(group.owner_id for group in event.candidate_groups) == tuple(
            sorted(group.owner_id for group in event.candidate_groups)
        )
        assert all(group.rows for group in event.candidate_groups)


def test_optional_a4_outputs_are_exactly_bound_and_add_static_events() -> None:
    manifest = load_manifest(MANIFEST)
    ledger = build_successor_ledger(
        manifest,
        manifest_path=MANIFEST,
        prior_outputs=(_prior(1), _prior(2)),
    )

    assert not ledger.exclusions
    assert {source.milestone for source in ledger.prior_sources} == {1, 2}
    assert len(ledger.events) > 12
    assert any(event.source_kind == "prior_output" for event in ledger.events)
    assert len(
        {
            (event.image_id, event.prefix_token_ids, event.duplicate_row.token_ids)
            for event in ledger.events
        }
    ) == len(ledger.events)


def test_misaligned_optional_output_is_excluded_without_changing_manifest_events(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(MANIFEST)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"image_id":14038,"generated_token_ids":[1]}\n', encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")

    ledger = build_successor_ledger(
        manifest,
        manifest_path=MANIFEST,
        prior_outputs=(
            PriorOutputSpec(
                arm_id="A4", milestone=1, outputs_path=bad, receipt_path=receipt
            ),
        ),
    )

    assert len(ledger.events) == 12
    assert len(ledger.exclusions) == 1
    assert ledger.exclusions[0].code.startswith("prior_output.")
    assert not ledger.prior_sources


def test_canonical_round_trip_and_digest_are_deterministic(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    first = build_successor_ledger(manifest, manifest_path=MANIFEST)
    second = build_successor_ledger(manifest, manifest_path=MANIFEST)
    assert first == second

    output = tmp_path / "ledger.json"
    digest = canonical_write(first, output)
    assert len(digest) == 64
    assert load_ledger(output) == first
    with pytest.raises(FileExistsError):
        canonical_write(first, output)

    document = json.loads(output.read_text(encoding="utf-8"))
    document["manifest_sha256"] = "0" * 64
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical|digest|manifest"):
        load_ledger(output)


def test_builder_rejects_manifest_digest_mismatch(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    copied = tmp_path / "manifest.json"
    copied.write_bytes(MANIFEST.read_bytes())
    Path(f"{copied}.sha256").write_text(
        f"{'0' * 64}  {copied.name}\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="manifest digest"):
        build_successor_ledger(manifest, manifest_path=copied)


def test_internal_event_identity_tamper_is_rejected(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    ledger = build_successor_ledger(manifest, manifest_path=MANIFEST)
    tampered = replace(
        ledger,
        events=(replace(ledger.events[0], target_owner_ids=()), *ledger.events[1:]),
    )
    with pytest.raises(ValueError, match="owner partition"):
        canonical_write(tampered, tmp_path / "tampered.json")
