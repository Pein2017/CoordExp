from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts.research import build_human13_k_union_manifest as manifest_builder
from scripts.research.build_human13_on_policy_frontier import (
    CheckpointIdentity,
    CurrentDecode,
    CurrentPrediction,
    build_frontier_iteration,
    candidate_aliases_for_owners,
    canonical_write,
    load_frontier_iteration,
    natural_pre_stop_prefix,
)


def _request(mode: str, *, seed: int | None = None) -> manifest_builder.RequestIdentity:
    if mode == "source_greedy":
        return manifest_builder.RequestIdentity(
            backend="hf",
            backend_version="test-hf",
            mode=mode,
            n=1,
            seed=None,
            physical_batch_index=0,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=3084,
        )
    assert seed is not None
    return manifest_builder.RequestIdentity(
        backend="vllm",
        backend_version="test-vllm",
        mode=mode,
        n=1,
        seed=seed,
        physical_batch_index=(seed - 21001) // 4,
        temperature=0.4,
        top_p=0.95,
        repetition_penalty=1.10,
        max_new_tokens=512,
    )


def _manifest(
    tmp_path: Path, *, h_tokens: tuple[int, int, int, int] = (21, 22, 23, 24)
) -> tuple[manifest_builder.Human13KUnionManifest, Path]:
    source = manifest_builder.TrajectoryInput(
        trajectory_id="source",
        request=_request("source_greedy"),
        token_ids=(100, 11, 12, 13, 14, 999),
        terminal_token_index=5,
        stop_reason="im_end",
        parser_status="accepted",
        rows=(
            manifest_builder.PredictionRowInput(
                "source-g", 0, "person", (0.0, 0.0, 10.0, 10.0), 1, 5, 3
            ),
        ),
    )
    sampled = []
    for seed in range(21001, 21017):
        rows = ()
        tokens = (200, 999)
        if seed in {21001, 21002}:
            candidate_tokens = h_tokens if seed == 21001 else (31, 32, 33, 34)
            tokens = (200, *candidate_tokens, 999)
            rows = (
                manifest_builder.PredictionRowInput(
                    f"k-h-{seed}",
                    0,
                    "person",
                    (20.0, 0.0, 30.0, 10.0),
                    1,
                    5,
                    3,
                ),
            )
        sampled.append(
            manifest_builder.TrajectoryInput(
                trajectory_id=f"k-{seed}",
                request=_request("k_sampled", seed=seed),
                token_ids=tokens,
                terminal_token_index=len(tokens) - 1,
                stop_reason="im_end",
                parser_status="accepted",
                rows=rows,
            )
        )
    built = manifest_builder.build_manifest(
        binding=manifest_builder.default_binding(),
        images=(
            manifest_builder.ImageInput(
                image_id=2299,
                owners=(
                    manifest_builder.OwnerInput(
                        "gt:2299:g", "person", (0.0, 0.0, 10.0, 10.0), 0
                    ),
                    manifest_builder.OwnerInput(
                        "gt:2299:h", "person", (20.0, 0.0, 30.0, 10.0), 1
                    ),
                    manifest_builder.OwnerInput(
                        "gt:2299:m", "person", (40.0, 0.0, 50.0, 10.0), 2
                    ),
                ),
                source=source,
                sampled=tuple(sampled),
            ),
        ),
        require_full_panel=False,
    )
    path = tmp_path / "manifest.json"
    manifest_builder.canonical_write(built, path)
    return built, path


def _decode(
    *, checkpoint: CheckpointIdentity, include_h: bool = False
) -> CurrentDecode:
    tokens = (700, 11, 12, 13, 14, 15, 16, 17, 18, 999)
    rows = [
        CurrentPrediction(0, "person", (0.0, 0.0, 10.0, 10.0), 1, 5),
        CurrentPrediction(1, "person", (0.1, 0.1, 10.1, 10.1), 5, 9),
    ]
    if include_h:
        tokens = (*tokens[:-1], 21, 22, 23, 24, 999)
        rows.append(CurrentPrediction(2, "person", (20.0, 0.0, 30.0, 10.0), 9, 13))
    return CurrentDecode(
        image_id=2299,
        trajectory_id="current-2299",
        generated_token_ids=tokens,
        predictions=tuple(rows),
        parser="compact_object_box_closed_only",
        parser_status="accepted",
        stop_reason="im_end",
        checkpoint=checkpoint,
        terminal_token_index=len(tokens) - 1,
        malformed_row_count=0,
    )


def test_frontier_binds_current_natural_tokens_matching_and_k_neutrality(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    checkpoint = CheckpointIdentity("/accepted/step-0", "a" * 64)

    frontier = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=0,
        checkpoint=checkpoint,
        decodes=(_decode(checkpoint=checkpoint),),
    )

    image = frontier.images[0]
    assert image.generated_token_ids == (700, 11, 12, 13, 14, 15, 16, 17, 18, 999)
    assert image.rows[0].token_ids == (11, 12, 13, 14)
    assert image.canonical_owner_ids == ("gt:2299:g",)
    assert image.constrained_protected_owner_ids == ("gt:2299:g",)
    assert image.covered_h_owner_ids == ()
    assert image.uncovered_h_owner_ids == ("gt:2299:h",)
    assert tuple(alias.owner_id for alias in image.candidate_aliases) == (
        "gt:2299:h",
        "gt:2299:h",
    )
    assert {alias.row_id for alias in image.candidate_aliases} == {
        "k-h-21001",
        "k-h-21002",
    }
    assert natural_pre_stop_prefix(image) == image.generated_token_ids[:-1]
    assert image.terminal_token_index == len(image.generated_token_ids) - 1
    assert image.malformed_row_count == 0
    assert "gt:2299:m" not in image.candidate_owner_ids
    assert len(image.duplicate_events) == 1
    assert image.duplicate_events[0].duplicate_generated_order == 1
    assert frontier.protected_owner_ages == (("gt:2299:g", 1),)


def test_live_sealed_manifest_exposes_all_309_native_candidate_aliases() -> None:
    manifest_path = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-12-human13-k-union-to-greedy-overfit-screen/manifest/"
        "human13-k-union-manifest.json"
    )
    if not manifest_path.is_file():
        pytest.skip("live sealed Human-13 manifest is unavailable")
    manifest = manifest_builder.load_manifest(manifest_path, require_full_panel=True)
    aliases = tuple(
        alias
        for image in manifest.images
        for alias in candidate_aliases_for_owners(
            image, owner_ids=set(image.h_owner_ids)
        )
    )

    assert len(aliases) == 309
    assert {alias.row_id for alias in aliases} == {
        row_id for image in manifest.images for row_id in image.candidate_row_ids
    }


def test_frontier_accepts_real_canonical_parser_status(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    checkpoint = CheckpointIdentity("/accepted/step-0", "a" * 64)
    decode = replace(_decode(checkpoint=checkpoint), parser_status="accepted")

    frontier = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=0,
        checkpoint=checkpoint,
        decodes=(decode,),
    )

    assert frontier.images[0].parser_status == "accepted"


def test_frontier_promotes_h_only_after_a_second_accepted_iteration(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    first_checkpoint = CheckpointIdentity("/accepted/step-0", "a" * 64)
    first = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=0,
        checkpoint=first_checkpoint,
        decodes=(_decode(checkpoint=first_checkpoint, include_h=True),),
    )
    first_path = tmp_path / "frontier-0.json"
    canonical_write(first, first_path)
    second_checkpoint = CheckpointIdentity("/accepted/step-1", "b" * 64)
    second = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=1,
        checkpoint=second_checkpoint,
        previous=first,
        previous_path=first_path,
        decodes=(_decode(checkpoint=second_checkpoint, include_h=True),),
    )

    assert first.protected_owner_ids == ("gt:2299:g",)
    assert second.protected_owner_ids == ("gt:2299:g", "gt:2299:h")
    assert second.protected_owner_ages == (("gt:2299:g", 2), ("gt:2299:h", 2))


def test_frontier_is_canonical_content_addressed_and_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    checkpoint = CheckpointIdentity("/accepted/step-0", "a" * 64)
    first = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=0,
        checkpoint=checkpoint,
        decodes=(_decode(checkpoint=checkpoint),),
    )
    second = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=0,
        checkpoint=checkpoint,
        decodes=(_decode(checkpoint=checkpoint),),
    )
    assert first == second
    output = tmp_path / "frontier.json"
    digest = canonical_write(first, output)
    assert len(digest) == 64
    assert load_frontier_iteration(output) == first
    with pytest.raises(FileExistsError):
        canonical_write(first, output)

    wrong_checkpoint = replace(
        _decode(checkpoint=checkpoint),
        checkpoint=CheckpointIdentity("/accepted/step-0", "b" * 64),
    )
    with pytest.raises(ValueError, match="checkpoint"):
        build_frontier_iteration(
            manifest,
            manifest_path=manifest_path,
            iteration=0,
            checkpoint=checkpoint,
            decodes=(wrong_checkpoint,),
        )
    with pytest.raises(ValueError, match="span"):
        build_frontier_iteration(
            manifest,
            manifest_path=manifest_path,
            iteration=0,
            checkpoint=checkpoint,
            decodes=(
                replace(
                    _decode(checkpoint=checkpoint),
                    predictions=(
                        replace(
                            _decode(checkpoint=checkpoint).predictions[0], token_end=99
                        ),
                    ),
                ),
            ),
        )
    with pytest.raises(ValueError, match="parser"):
        build_frontier_iteration(
            manifest,
            manifest_path=manifest_path,
            iteration=0,
            checkpoint=checkpoint,
            decodes=(replace(_decode(checkpoint=checkpoint), parser_status="partial"),),
        )
    with pytest.raises(ValueError, match="parser"):
        build_frontier_iteration(
            manifest,
            manifest_path=manifest_path,
            iteration=0,
            checkpoint=checkpoint,
            decodes=(replace(_decode(checkpoint=checkpoint), parser="wrong"),),
        )

    document = json.loads(output.read_text(encoding="utf-8"))
    document["manifest_sha256"] = "0" * 64
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    output.write_bytes(payload)
    (tmp_path / "frontier.json.sha256").write_text(
        f"{__import__('hashlib').sha256(payload).hexdigest()}  frontier.json\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="manifest|digest"):
        load_frontier_iteration(output)


@pytest.mark.parametrize(
    ("field", "value"),
    (("terminal_token_index", 0), ("malformed_row_count", -1)),
)
def test_load_rejects_terminal_or_malformed_identity_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    checkpoint = CheckpointIdentity("/accepted/step-0", "a" * 64)
    frontier = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=0,
        checkpoint=checkpoint,
        decodes=(_decode(checkpoint=checkpoint),),
    )
    output = tmp_path / "frontier.json"
    canonical_write(frontier, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["images"][0][field] = value
    _rewrite_canonical(output, document)
    with pytest.raises(ValueError, match="terminal|malformed|semantic|derive"):
        load_frontier_iteration(output)


def _rewrite_canonical(path: Path, document: dict[str, object]) -> None:
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    path.write_bytes(payload)
    Path(f"{path}.sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n", encoding="ascii"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("canonical_owner_ids", []),
        ("covered_h_owner_ids", ["gt:2299:h"]),
        ("uncovered_h_owner_ids", []),
        ("duplicate_events", []),
    ),
)
def test_load_rederives_semantic_image_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    checkpoint = CheckpointIdentity("/accepted/step-0", "a" * 64)
    frontier = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=0,
        checkpoint=checkpoint,
        decodes=(_decode(checkpoint=checkpoint),),
    )
    output = tmp_path / "frontier.json"
    canonical_write(frontier, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["images"][0][field] = value
    _rewrite_canonical(output, document)
    with pytest.raises(ValueError, match="semantic|derive|match|duplicate|coverage"):
        load_frontier_iteration(output)


def test_write_rejects_inconsistent_row_slice(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    checkpoint = CheckpointIdentity("/accepted/step-0", "a" * 64)
    frontier = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=0,
        checkpoint=checkpoint,
        decodes=(_decode(checkpoint=checkpoint),),
    )
    image = frontier.images[0]
    bad_row = replace(image.rows[0], token_ids=(999,))
    bad = replace(frontier, images=(replace(image, rows=(bad_row, *image.rows[1:])),))
    with pytest.raises(ValueError, match="slice|semantic"):
        canonical_write(bad, tmp_path / "bad-frontier.json")
    wrong_manifest_digest = replace(frontier, manifest_sha256="0" * 64)
    wrong_output = tmp_path / "wrong-manifest-frontier.json"
    with pytest.raises(ValueError, match="manifest.*digest"):
        canonical_write(wrong_manifest_digest, wrong_output)
    assert not wrong_output.exists()
    assert not Path(f"{wrong_output}.sha256").exists()


def test_load_rederives_protected_owner_ages(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    checkpoint = CheckpointIdentity("/accepted/step-0", "a" * 64)
    frontier = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=0,
        checkpoint=checkpoint,
        decodes=(_decode(checkpoint=checkpoint),),
    )
    output = tmp_path / "frontier.json"
    canonical_write(frontier, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["protected_owner_ages"] = [["gt:2299:g", 99]]
    _rewrite_canonical(output, document)
    with pytest.raises(ValueError, match="protection ages"):
        load_frontier_iteration(output)


def test_write_rejects_predecessor_from_different_manifest(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest(tmp_path / "current")
    source_checkpoint = CheckpointIdentity("/accepted/step-0", "a" * 64)
    previous = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=0,
        checkpoint=source_checkpoint,
        decodes=(_decode(checkpoint=source_checkpoint, include_h=True),),
    )
    previous_path = tmp_path / "current-previous.json"
    canonical_write(previous, previous_path)
    current_checkpoint = CheckpointIdentity("/accepted/step-1", "b" * 64)
    current = build_frontier_iteration(
        manifest,
        manifest_path=manifest_path,
        iteration=1,
        checkpoint=current_checkpoint,
        previous=previous,
        previous_path=previous_path,
        decodes=(_decode(checkpoint=current_checkpoint, include_h=True),),
    )

    foreign, foreign_path = _manifest(tmp_path / "foreign", h_tokens=(31, 32, 33, 34))
    foreign_previous = build_frontier_iteration(
        foreign,
        manifest_path=foreign_path,
        iteration=0,
        checkpoint=source_checkpoint,
        decodes=(_decode(checkpoint=source_checkpoint, include_h=True),),
    )
    foreign_previous_path = tmp_path / "foreign-previous.json"
    foreign_digest = canonical_write(foreign_previous, foreign_previous_path)
    poisoned = replace(
        current,
        previous_frontier_path=str(foreign_previous_path.resolve()),
        previous_frontier_sha256=foreign_digest,
    )
    bad_output = tmp_path / "bad-current.json"
    with pytest.raises(ValueError, match="previous.*manifest|manifest.*previous"):
        canonical_write(poisoned, bad_output)
    assert not bad_output.exists()

    valid_output = tmp_path / "valid-current.json"
    canonical_write(current, valid_output)
    document = json.loads(valid_output.read_text(encoding="utf-8"))
    document["previous_frontier_path"] = str(foreign_previous_path.resolve())
    document["previous_frontier_sha256"] = foreign_digest
    _rewrite_canonical(valid_output, document)
    with pytest.raises(ValueError, match="previous.*manifest|manifest.*previous"):
        load_frontier_iteration(valid_output)


def test_build_rejects_manifest_object_that_differs_from_bound_path(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    owner = manifest.images[0].owners[1]
    different_image = replace(
        manifest.images[0],
        owners=(
            manifest.images[0].owners[0],
            replace(owner, category="cat"),
            *manifest.images[0].owners[2:],
        ),
    )
    different = replace(manifest, images=(different_image,))
    checkpoint = CheckpointIdentity("/accepted/step-0", "a" * 64)
    with pytest.raises(ValueError, match="manifest.*bound|bound.*manifest"):
        build_frontier_iteration(
            different,
            manifest_path=manifest_path,
            iteration=0,
            checkpoint=checkpoint,
            decodes=(_decode(checkpoint=checkpoint),),
        )


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_build_rejects_nonfinite_current_boxes(tmp_path: Path, value: float) -> None:
    manifest, manifest_path = _manifest(tmp_path)
    checkpoint = CheckpointIdentity("/accepted/step-0", "a" * 64)
    decode = _decode(checkpoint=checkpoint)
    bad_prediction = replace(decode.predictions[0], bbox=(value, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="finite|bbox"):
        build_frontier_iteration(
            manifest,
            manifest_path=manifest_path,
            iteration=0,
            checkpoint=checkpoint,
            decodes=(
                replace(
                    decode,
                    predictions=(bad_prediction, *decode.predictions[1:]),
                ),
            ),
        )
