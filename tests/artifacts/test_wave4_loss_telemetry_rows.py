"""Wave-4 artifact-first row-schema tests for canonical loss telemetry.

These tests are the RED evidence for tasks 4.1/4.4 of
``standardize-coordexp-swift-supervised-losses``. They drive the REAL
streaming loss protocol (``prepare_planned_step`` /``compute_micro_step``
/``finalize_planned_step``), project the finalized bundle through the real
rank-zero train projection (``src.training.reporting.CompletedStepReporter``)
and the real forward-eval projection (``src.eval.forward._metric_scalars`` ->
``ForwardEvalObservation.to_logging_row``), and assert the EXACT persisted
``logging.jsonl`` row key set for the five bounded shapes named by task 4.4:

1. enabled baseline (base CE + enabled token-type gate),
2. named zero-weight gate ablation,
3. enabled coordinate Gaussian/RPS auxiliary,
4. omitted coordinate Gaussian/RPS auxiliary,
5. unsafe-step serialization (non-finite computed values).

Field contract under test (design decision 5, artifacts delta
``Wide-Step Logging Stream``, supervision delta ``Loss Bundle Metrics``):

- every computed term has ``loss/<term>/raw`` and ``loss/<term>/weighted``;
- the ambiguous bare ``loss/<term>`` alias does NOT exist (no dual-write);
- ``loss/total`` equals the sum of WEIGHTED OBJECTIVE terms;
- counts/denominators/finite diagnostics share the term namespace
  (``loss/<term>/selected_count``, ``loss/<term>/segment_count``,
  ``finite/<term>``);
- top-level ``acc_top1``/``acc_top5``;
- the gate ablation keeps raw/count/finite and has weighted exactly ``0.0``;
- an omitted auxiliary has NO field in its family at all;
- non-finite computed values serialize as JSON ``null`` and are named in
  ``non_finite_fields``;
- Wave-3's additive backward-side keys (``backward_loss``,
  ``backward_contribution``, ``backend_gradient_scale``) are SEMANTICALLY
  backend state, never persisted row fields (pre-DDP audit I-5).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch

from src.artifacts.run_writer import RunWriter
from src.coordinate_targets import CoordinateLossTarget
from src.eval.forward import EVAL_FORWARD_SPLIT, ForwardEvalObservation, _metric_scalars
from src.losses import (
    CoordGaussianRPSLoss,
    LossContext,
    LossRunner,
    TokenVocabularyGroups,
)
from src.packing.planner import PackedSegment
from src.runtime.metrics import reduce_rank_payloads
from src.supervision import TokenAtom, TokenSequence
from src.training import reporting
from src.training.supervised_trainer import CompletedStepObservation

CANONICAL_GROUPS = ("desc_text", "schema", "coordinate", "eos")
REPO_ROOT = Path(__file__).resolve().parents[2]

# Keys every row carries regardless of the composed term inventory.
_ROW_ENVELOPE_TRAIN = frozenset(
    {
        "step",
        "split",
        "micro_step_count",
        "optimizer_update_status",
        "finite_status",
        "non_finite_fields",
        "accuracy_stats",
        "lr/group_0",
    }
)
_ROW_ENVELOPE_EVAL = frozenset(
    {
        "step",
        "split",
        "trigger_reasons",
        "example_count",
        "pack_count",
        "non_finite_fields",
        "accuracy_stats",
    }
)
_SHARED_METRIC_KEYS = frozenset(
    {
        "loss/total",
        "acc_top1",
        "acc_top5",
        "finite/total_loss",
        "count/supervised_atoms",
        "count/eligible_segments",
        "count/skipped_segments",
        "count/packs",
        "count/examples",
    }
)
# The complete per-term family. `raw`/`weighted` replace the ambiguous bare
# `loss/<term>`; the rest are the matching count, denominator, and finite
# diagnostics in the same term namespace.
_TERM_FIELD_SUFFIXES = (
    "/raw",
    "/weighted",
    "/selected_count",
    "/segment_count",
    "/token_weighted_diag",
)
# Never a persisted row field (audit I-5): these are backend/backward state.
FORBIDDEN_ROW_SUBSTRINGS = (
    "backward_loss",
    "backward_contribution",
    "backend_gradient_scale",
)


# DECLARED FLIP (add-coordexp-swift-training-observability, Wave 3, tasks
# 3.3/3.4/3.7).
#
# Old assertion: train and forward-eval rows had the SAME complete key set for
# every computed term.
#
# New assertion: the loss projection is still one shared projection, but TRAIN
# rows additionally expose the configured weight and the denominator inputs
# needed to interpret raw/weighted ("Additive to that prerequisite schema,
# train rows MUST also expose, for every actually computed loss term, its
# configured weight, denominator scope, eligible-segment count, selected-atom
# count, and skipped-segment count"). `segment_count` above IS the
# eligible-segment count, so it is not duplicated. Eval rows are unchanged.
_WAVE3_TRAIN_ONLY_TERM_SUFFIXES = (
    "/weight",
    "/denominator_scope",
    "/selected_atom_count",
    "/skipped_segment_count",
)
# A train observation carrying finalized loss telemetry but no measured step
# duration, work count, or allocator sample names those fields unavailable
# rather than publishing a fabricated zero for them.
_WAVE3_TRAIN_ONLY_ENVELOPE = frozenset({"unavailable_fields"})


def _term_keys(term_names: tuple[str, ...]) -> set[str]:
    keys: set[str] = set()
    for name in term_names:
        keys.update(f"loss/{name}{suffix}" for suffix in _TERM_FIELD_SUFFIXES)
        keys.add(f"finite/{name}")
    return keys


def wave3_train_only_keys(term_names: tuple[str, ...]) -> set[str]:
    keys = set(_WAVE3_TRAIN_ONLY_ENVELOPE)
    for name in term_names:
        keys.update(
            f"loss/{name}{suffix}" for suffix in _WAVE3_TRAIN_ONLY_TERM_SUFFIXES
        )
    return keys


def expected_train_row_keys(term_names: tuple[str, ...]) -> set[str]:
    return (
        set(_ROW_ENVELOPE_TRAIN)
        | set(_SHARED_METRIC_KEYS)
        | _term_keys(term_names)
        | wave3_train_only_keys(term_names)
    )


def expected_eval_row_keys(term_names: tuple[str, ...]) -> set[str]:
    return set(_ROW_ENVELOPE_EVAL) | set(_SHARED_METRIC_KEYS) | _term_keys(term_names)


# ---------------------------------------------------------------------------
# Real streaming harness (world size one; no distributed collective involved)
# ---------------------------------------------------------------------------


def _groups() -> TokenVocabularyGroups:
    return TokenVocabularyGroups(
        vocab_size=8,
        desc_text=(7,),
        schema=(1, 2),
        coordinate=(3, 4),
        eos=(5,),
        blocked=(0, 6),
    )


def _segment(segment_index: int, start: int, end: int) -> PackedSegment:
    return PackedSegment(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        start=start,
        end=end,
    )


def _atom(
    *,
    segment_index: int,
    target_position: int,
    token_id: int,
    token_type: str = "desc_text",
    coordinate_target: CoordinateLossTarget | None = None,
) -> TokenAtom:
    return TokenAtom(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        target_position=target_position,
        token_id=token_id,
        token_type=token_type,
        text="x",
        logical_target_position=target_position,
        object_id="obj-1" if coordinate_target is not None else None,
        field="bbox[0]" if coordinate_target is not None else None,
        source="unit",
        coordinate_target=coordinate_target,
    )


def _context(
    logits: torch.Tensor,
    segments: tuple[PackedSegment, ...],
    atoms: tuple[TokenAtom, ...],
) -> LossContext:
    return LossContext(
        logits=logits,
        token_sequence=TokenSequence(
            pack_index=0,
            input_ids=tuple(0 for _ in range(int(logits.shape[1]))),
            segments=segments,
            atoms=atoms,
            spans=(),
        ),
        vocab_groups=_groups(),
        logits_position_ids=None,
    )


def _text_context(*, non_finite: bool = False) -> LossContext:
    # The supervised atom at `target_position=1` is predicted from the logits
    # row at index 0, so that is the row the non-finite injection must poison.
    first_row = (
        (0.0, 1.0, 0.0, float("inf"), 0.0, 0.0, 0.0, 8.0)
        if non_finite
        else (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0)
    )
    second_row = (0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0)
    return _context(
        torch.tensor(((first_row, second_row),), dtype=torch.float32),
        (_segment(0, 0, 2),),
        (_atom(segment_index=0, target_position=1, token_id=7),),
    )


def _coordinate_context() -> LossContext:
    return _context(
        torch.tensor(
            (
                (
                    (0.0, 0.0, 8.0, 2.0, 0.0, 0.0, 0.0, 1.0),
                    (0.0, 0.0, 0.0, 7.0, 1.0, 0.0, 0.0, 1.0),
                ),
            ),
            dtype=torch.float32,
        ),
        (_segment(0, 0, 2),),
        (
            _atom(
                segment_index=0,
                target_position=1,
                token_id=3,
                token_type="coordinate",
                coordinate_target=CoordinateLossTarget(bbox=(2, 3, 8, 13), slot_index=0),
            ),
        ),
    )


def _coord_term() -> CoordGaussianRPSLoss:
    return CoordGaussianRPSLoss(
        gaussian_weight=0.5,
        rps_weight=0.2,
        temperature=1.0,
        gaussian_r95_axis_fraction=0.5,
        gaussian_r95_cap_bins=4,
        gaussian_r95_min_bins=1,
        gaussian_r95_fallback_bins=4,
    )


def _finalized(
    *,
    gate_weight: float = 0.1,
    coord_weight: float = 0.0,
    non_finite: bool = False,
) -> dict[str, Any]:
    """One REAL finalized planned-step loss artifact at world size one."""

    coord_enabled = coord_weight > 0.0
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=gate_weight,
        token_type_gate_groups=CANONICAL_GROUPS,
        coord_gaussian_rps_weight=coord_weight,
        coord_gaussian_rps=_coord_term() if coord_enabled else None,
    )
    context = (
        _coordinate_context() if coord_enabled else _text_context(non_finite=non_finite)
    )
    plan = runner.prepare_planned_step((context.token_sequence,))
    micro = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    return runner.finalize_planned_step((micro.to_artifact_dict(),), plan)


# ---------------------------------------------------------------------------
# Rank-zero projections (train and forward eval use the SAME metric mapping)
# ---------------------------------------------------------------------------


class _Accelerator:
    is_main_process = True
    num_processes = 1


class _Runtime:
    """World-size-one gather: the projection under test is the seam.

    DECLARED FLIP (add-coordexp-swift-training-observability, Wave 2, task
    2.3): the reporter now declares a typed `MetricBatch`, so this double runs
    the real world-size-one reduction rather than echoing a mapping. Row keys
    and values are unchanged, which the exact key-set assertions below keep
    proving.
    """

    is_main_process = True
    world_size = 1
    accelerator = _Accelerator()

    def gather_metrics(self, batch: Any) -> object:
        reduced = reduce_rank_payloads(
            [batch.to_rank_payload(rank=0, world_size=1)], world_size=1
        )
        result: dict[str, object] = {"metrics": dict(reduced.metrics)}
        if reduced.accuracy_stats is not None:
            result["accuracy_stats"] = dict(reduced.accuracy_stats)
        return result


def _writer(tmp_path: Path) -> RunWriter:
    return RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="run",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="now",
        config_fingerprint="fp",
        resolved_config={},
        world_size=1,
        resolved_max_steps=5,
    )


def _train_row(
    tmp_path: Path,
    artifact: dict[str, Any],
    *,
    optimizer_update_status: str = "applied",
    finite_status: str = "finite",
) -> dict[str, Any]:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(
        CompletedStepObservation(
            planned_step_id=1,
            micro_step_count=1,
            loss_bundle_artifact=artifact,
            optimizer_update_status=optimizer_update_status,
            finite_status=finite_status,
            scheduler_artifact={"learning_rates": [{"group_index": 0, "lr": 1e-5}]},
        )
    )
    return json.loads(writer.logging_path.read_text(encoding="utf-8"))


def _eval_row(tmp_path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    writer = _writer(tmp_path)
    observation = ForwardEvalObservation(
        planned_step_id=1,
        split=EVAL_FORWARD_SPLIT,
        trigger_reasons=("scheduled",),
        example_count=1,
        pack_count=1,
        scalars=_metric_scalars(None, artifact),
        accuracy_stats=dict(artifact["accuracy_stats"]),
    )
    writer.append_logging_row(observation.to_logging_row())
    return json.loads(writer.logging_path.read_text(encoding="utf-8"))


def _term_names(artifact: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(term["name"]) for term in artifact["terms"])


def _assert_no_bare_aliases(row: dict[str, Any], term_names: tuple[str, ...]) -> None:
    for name in term_names:
        assert f"loss/{name}" not in row, (
            f"ambiguous bare alias loss/{name} must not be dual-written; "
            "raw and weighted are the only per-term values"
        )


def _assert_no_backward_state(row: dict[str, Any]) -> None:
    for key in row:
        for forbidden in FORBIDDEN_ROW_SUBSTRINGS:
            assert forbidden not in key, (
                f"row field {key!r} leaks backward/backend state; persisted rows "
                "carry SEMANTIC values only (pre-DDP audit I-5)"
            )


# ---------------------------------------------------------------------------
# 4.4 shape 1 - enabled baseline
# ---------------------------------------------------------------------------


def test_enabled_baseline_train_row_has_exact_raw_weighted_schema(
    tmp_path: Path,
) -> None:
    artifact = _finalized(gate_weight=0.1)
    names = _term_names(artifact)
    assert names == ("base_ce", "token_type_gate")

    row = _train_row(tmp_path, artifact)

    assert set(row) == expected_train_row_keys(names)
    _assert_no_bare_aliases(row, names)
    _assert_no_backward_state(row)
    assert row["split"] == "train"
    assert row["non_finite_fields"] == []
    # Weighted == raw * configured weight for every computed term.
    assert row["loss/base_ce/weighted"] == pytest.approx(row["loss/base_ce/raw"] * 1.0)
    assert row["loss/token_type_gate/weighted"] == pytest.approx(
        row["loss/token_type_gate/raw"] * 0.1
    )
    # Total is the sum of the WEIGHTED OBJECTIVE terms, and is not any raw sum.
    assert row["loss/total"] == pytest.approx(
        row["loss/base_ce/weighted"] + row["loss/token_type_gate/weighted"]
    )
    assert row["loss/total"] != pytest.approx(
        row["loss/base_ce/raw"] + row["loss/token_type_gate/raw"]
    )
    assert row["acc_top1"] == pytest.approx(1.0)
    assert row["acc_top5"] == pytest.approx(1.0)
    assert row["finite/base_ce"] == 1.0
    assert row["finite/token_type_gate"] == 1.0


@pytest.mark.parametrize(
    "shape,kwargs",
    [
        ("enabled_baseline", {"gate_weight": 0.1}),
        ("gate_ablation", {"gate_weight": 0.0}),
        ("coord_auxiliary_enabled", {"gate_weight": 0.1, "coord_weight": 1.0}),
        ("coord_auxiliary_omitted", {"gate_weight": 0.1, "coord_weight": 0.0}),
        ("unsafe_step", {"gate_weight": 0.1, "non_finite": True}),
    ],
)
def test_eval_row_uses_the_same_projection_as_train_for_every_shape(
    tmp_path: Path, shape: str, kwargs: dict[str, Any]
) -> None:
    """4.2: ONE projection for train and forward eval, on all five shapes."""

    artifact = _finalized(**kwargs)
    names = _term_names(artifact)
    unsafe = shape == "unsafe_step"

    train = _train_row(
        tmp_path / "t",
        artifact,
        optimizer_update_status="skipped" if unsafe else "applied",
        finite_status="non_finite" if unsafe else "finite",
    )
    evaluated = _eval_row(tmp_path / "e", artifact)

    assert set(train) == expected_train_row_keys(names)
    assert set(evaluated) == expected_eval_row_keys(names)
    # Identical computed-term projection on both splits: same field names,
    # same values, same non-finite normalization, from one artifact.
    selector = ("loss/", "finite/", "count/", "acc_top")
    train_only = wave3_train_only_keys(names)
    shared = {
        key
        for key in train
        if key.startswith(selector) and key not in train_only
    }
    assert shared == {key for key in evaluated if key.startswith(selector)}
    for key in shared:
        if train[key] is None:
            assert evaluated[key] is None
        else:
            assert evaluated[key] == pytest.approx(train[key])
    assert set(train["non_finite_fields"]) == set(evaluated["non_finite_fields"])


def test_enabled_baseline_eval_row_uses_the_same_projection(tmp_path: Path) -> None:
    artifact = _finalized(gate_weight=0.1)
    names = _term_names(artifact)

    train = _train_row(tmp_path / "t", artifact)
    evaluated = _eval_row(tmp_path / "e", artifact)

    assert set(evaluated) == expected_eval_row_keys(names)
    _assert_no_bare_aliases(evaluated, names)
    _assert_no_backward_state(evaluated)
    assert evaluated["split"] == "eval"
    # Same computed-term projection on both splits: identical loss/finite/count
    # field names and identical values from one artifact.
    train_only = wave3_train_only_keys(names)
    shared = {
        key
        for key in train
        if key.startswith(("loss/", "finite/", "count/", "acc_top"))
        and key not in train_only
    }
    assert shared == {
        key
        for key in evaluated
        if key.startswith(("loss/", "finite/", "count/", "acc_top"))
    }
    for key in shared:
        assert evaluated[key] == pytest.approx(train[key])


# ---------------------------------------------------------------------------
# 4.4 shape 2 - named zero-weight gate ablation
# ---------------------------------------------------------------------------


def test_gate_ablation_row_keeps_raw_count_finite_and_zero_weighted(
    tmp_path: Path,
) -> None:
    artifact = _finalized(gate_weight=0.0)
    names = _term_names(artifact)
    assert names == ("base_ce", "token_type_gate")

    row = _train_row(tmp_path, artifact)

    assert set(row) == expected_train_row_keys(names)
    _assert_no_bare_aliases(row, names)
    # The diagnostic stays fully visible.
    assert math.isfinite(row["loss/token_type_gate/raw"])
    assert row["loss/token_type_gate/raw"] != 0.0
    assert row["loss/token_type_gate/selected_count"] >= 1.0
    assert row["finite/token_type_gate"] == 1.0
    # ... and is exactly zero-weighted, hence distinguishable from an
    # optimized objective term whose weighted value tracks its raw value.
    assert row["loss/token_type_gate/weighted"] == 0.0
    assert row["loss/total"] == pytest.approx(row["loss/base_ce/weighted"])
    assert row["loss/base_ce/weighted"] != 0.0


# ---------------------------------------------------------------------------
# 4.4 shape 3 - enabled coordinate auxiliary
# ---------------------------------------------------------------------------


def test_enabled_coordinate_auxiliary_row_adds_the_full_term_family(
    tmp_path: Path,
) -> None:
    artifact = _finalized(gate_weight=0.1, coord_weight=1.0)
    names = _term_names(artifact)
    assert names == ("base_ce", "token_type_gate", "coord_gaussian_rps")

    row = _train_row(tmp_path, artifact)

    assert set(row) == expected_train_row_keys(names)
    _assert_no_bare_aliases(row, names)
    _assert_no_backward_state(row)
    assert row["loss/coord_gaussian_rps/weighted"] == pytest.approx(
        row["loss/coord_gaussian_rps/raw"] * 1.0
    )
    assert row["loss/total"] == pytest.approx(
        row["loss/base_ce/weighted"]
        + row["loss/token_type_gate/weighted"]
        + row["loss/coord_gaussian_rps/weighted"]
    )


# ---------------------------------------------------------------------------
# 4.4 shape 4 - omitted coordinate auxiliary
# ---------------------------------------------------------------------------


def test_omitted_coordinate_auxiliary_family_is_entirely_absent(
    tmp_path: Path,
) -> None:
    artifact = _finalized(gate_weight=0.1, coord_weight=0.0)
    names = _term_names(artifact)
    assert "coord_gaussian_rps" not in names

    train = _train_row(tmp_path / "t", artifact)
    evaluated = _eval_row(tmp_path / "e", artifact)

    for row in (train, evaluated):
        assert not any("coord_gaussian_rps" in key for key in row), sorted(
            key for key in row if "coord_gaussian_rps" in key
        )
    assert set(train) == expected_train_row_keys(names)
    assert set(evaluated) == expected_eval_row_keys(names)


# ---------------------------------------------------------------------------
# 4.4 shape 5 - unsafe-step serialization
# ---------------------------------------------------------------------------


def test_unsafe_step_row_nulls_non_finite_values_and_names_them(
    tmp_path: Path,
) -> None:
    artifact = _finalized(gate_weight=0.1, non_finite=True)
    names = _term_names(artifact)

    row = _train_row(
        tmp_path,
        artifact,
        optimizer_update_status="skipped",
        finite_status="non_finite",
    )

    assert set(row) == expected_train_row_keys(names)
    _assert_no_bare_aliases(row, names)
    assert row["optimizer_update_status"] == "skipped"
    assert row["finite_status"] == "non_finite"
    non_finite_fields = set(row["non_finite_fields"])
    # The non-finite computed values keep their names with JSON null.
    assert {
        "loss/total",
        "loss/base_ce/raw",
        "loss/base_ce/weighted",
    } <= non_finite_fields
    for field_name in non_finite_fields:
        assert row[field_name] is None
    # Explicit statuses survive normalization; the finite flags are 0.0.
    assert row["finite/total_loss"] == 0.0
    assert row["finite/base_ce"] == 0.0
    # The serialized bytes carry real JSON null, never NaN/Infinity literals.
    raw_bytes = (tmp_path / "run" / "logging.jsonl").read_text(encoding="utf-8")
    assert "NaN" not in raw_bytes and "Infinity" not in raw_bytes
    assert '"loss/total":null' in raw_bytes.replace(" ", "")


# ---------------------------------------------------------------------------
# Audit I-5 - Wave-3 additive keys are not row fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gate_weight": 0.1},
        {"gate_weight": 0.0},
        {"gate_weight": 0.1, "coord_weight": 1.0},
    ],
)
def test_backward_and_backend_state_never_reaches_a_persisted_row(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    """Rows carry SEMANTIC values only (pre-DDP audit I-5).

    `backward_loss`, `backward_contribution`, and `backend_gradient_scale`
    exist on the bundle/term artifacts and in `diagnostics`, but are backend
    mean-gradient compensation state whose value is rank- and world-size-
    dependent. They MUST NOT be projected into `logging.jsonl`.
    """

    artifact = _finalized(**kwargs)
    # Precondition: the artifact really does carry them (otherwise this test
    # is vacuous).
    assert "backward_loss" in artifact
    assert "backend_gradient_scale" in artifact["diagnostics"]
    assert all("backward_contribution" in term for term in artifact["terms"])

    _assert_no_backward_state(_train_row(tmp_path / "t", artifact))
    _assert_no_backward_state(_eval_row(tmp_path / "e", artifact))


# ---------------------------------------------------------------------------
# Frozen decompose fixtures stay byte-frozen (entry-audit F-5/F-6)
# ---------------------------------------------------------------------------


def test_frozen_training_orchestration_fixtures_are_untouched() -> None:
    subtree = subprocess.run(
        ["git", "rev-parse", "HEAD:tests/fixtures/training_orchestration"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert subtree == "2fe137ccffc9b51e4eb227a91e2fbe6607dfc528"
    worktree_digest = hashlib.sha256()
    fixture_root = REPO_ROOT / "tests" / "fixtures" / "training_orchestration"
    for path in sorted(fixture_root.rglob("*")):
        if path.is_file():
            worktree_digest.update(
                str(path.relative_to(fixture_root)).encode("utf-8") + b"\0"
            )
            worktree_digest.update(path.read_bytes())
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "tests/fixtures/training_orchestration"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert status == "", f"frozen fixtures modified in the worktree: {status}"
    # The frozen fixtures carry only `loss/total`, which this change preserves.
    frozen_rows = json.loads(
        (fixture_root / "completed_step_rows.json").read_text(encoding="utf-8")
    )
    for row in frozen_rows["rows"]:
        assert not any(
            key.startswith("loss/") and key != "loss/total" for key in row
        )
