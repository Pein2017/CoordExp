from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

import pytest

from scripts.research import resumable_natural_boundary_support_completion as adapter
from src.artifacts.evidence_journal import ExecutionEvidenceJournal
from src.artifacts.json_values import json_sha256
from src.common.errors import ArtifactContractError


def _plan_path(tmp_path: Path) -> Path:
    contexts = []
    for index, (cost, shard) in enumerate(((7, 0), (6, 1), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6), (1, 7))):
        contexts.append(
            {
                "context_id": f"context-{index}",
                "context_plan_position": index,
                "scalar_equivalent_forward_count": cost,
                "shard_index": shard,
                "candidate_ids": [f"candidate-{index}"],
            }
        )
    raw: dict[str, Any] = {
        "schema_version": "fixture.plan.v1",
        "unit_id": "fixture-unit",
        "contexts": contexts,
        "calibration_reuse": {"calibration_sha256": "a" * 64},
        "support_lineage": {"support_rule": {"fixture": True}},
        "scope": {"native_tp_other_not_scored": True},
        "work": {
            "candidate_batch_size": 1,
            "scalar_equivalent_forward_count": sum(item[0] for item in ((7, 0), (6, 1), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6), (1, 7))),
            "per_shard": [
                {"shard_index": shard, "scalar_equivalent_forward_count": cost}
                for shard, cost in enumerate((7, 6, 5, 4, 3, 2, 1, 1))
            ],
        },
    }
    raw["plan_content_sha256"] = json_sha256(raw)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def _observation(context: dict[str, Any]) -> dict[str, Any]:
    scores = {context["candidate_ids"][0]: 0.0}
    return {
        "context_id": context["context_id"],
        "status": "measured",
        "support_features": {"assessed": True},
        "candidate_scores": scores,
        "candidate_score_count": context["scalar_equivalent_forward_count"],
        "candidate_scores_sha256": json_sha256(scores),
    }


def _fixture_census() -> dict[str, str]:
    return {
        "file_sha256": "b" * 64,
        "self_sha256": "c" * 64,
        "s_owner_ids_sha256": "d" * 64,
    }


def _accept_receipt_set(receipts: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    return {"accepted_shards": sorted(receipts)}


def test_lpt_schedule_uses_cost_then_context_and_lowest_load_count_slot(tmp_path: Path) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    first = adapter.plan_physical_slots(plan, slot_count=2)
    second = adapter.plan_physical_slots(plan, slot_count=2)

    assert first == second
    assert [slot["scalar_equivalent_forward_count"] for slot in first["slots"]] == [15, 14]
    assert first["slots"][0]["context_ids"] == ["context-0", "context-3", "context-4", "context-7"]
    assert first["slots"][1]["context_ids"] == ["context-1", "context-2", "context-5", "context-6"]
    assert adapter.validate_schedule(first, plan) == first
    drifted = dict(first)
    drifted["slot_count"] = 3
    with pytest.raises(adapter.SupportShardAdapterError, match="identity"):
        adapter.validate_schedule(drifted, plan)


def test_adapter_refuses_symlink_inputs_and_uses_canonical_exclusive_output(tmp_path: Path) -> None:
    target = tmp_path / "plan.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "plan-link.json"
    link.symlink_to(target)
    with pytest.raises(adapter.SupportShardAdapterError, match="non-symlink"):
        adapter.file_sha256(link)
    output = tmp_path / "receipt.json"
    assert adapter.write_once_json(output, {"value": 1})
    assert output.read_bytes() == b'{"value":1}'
    assert adapter.write_once_json(output, {"value": 1})
    with pytest.raises(adapter.SupportShardAdapterError, match="overwrite"):
        adapter.write_once_json(output, {"value": 2})


def test_nonmutation_recheck_consumes_source_binding_read_only(tmp_path: Path) -> None:
    receipt = adapter.verify_active_nonmutation(
        source_bindings_path=Path(
            "openspec/changes/archive/2026-08-07-add-resumable-costed-research-probe-shards/verification/source-bindings-v3.json"
        ),
        output_path=tmp_path / "nonmutation.json",
    )
    assert receipt["result"] == "unchanged_at_recheck"
    assert set(receipt["observed"]) == {"consumer", "merger", "sealed_plan", "census"}


def test_sealed_plan_schedule_has_the_accepted_exact_costs() -> None:
    plan = adapter.load_logical_plan(Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-06-natural-boundary-routing-history-replication/"
        "support-completion-plan-v1/plan.json"
    ))
    schedule = adapter.plan_physical_slots(plan)
    assert [slot["context_count"] for slot in schedule["slots"]] == [25] * 8
    assert [slot["scalar_equivalent_forward_count"] for slot in schedule["slots"]] == [
        9675, 9675, 9675, 9683, 9678, 9681, 9672, 9689,
    ]
    assert sum(slot["scalar_equivalent_forward_count"] for slot in schedule["slots"]) == 77428


def test_slot_journal_continuation_is_exact_and_never_reexecutes_accepted_payload(tmp_path: Path) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    schedule = adapter.plan_physical_slots(plan, slot_count=2)
    observed: list[str] = []

    def observe(context: dict[str, Any]) -> dict[str, Any]:
        observed.append(context["context_id"])
        return _observation(context)

    root = tmp_path / "slot-0"
    adapter.execute_slot(
        root=root, execution_id="fixture", execution_identity={"model": "fake"},
        plan=plan, schedule=schedule, slot_index=0, observe_context=observe, stop_after_records=1,
    )
    before = tuple(observed)
    with pytest.raises(ArtifactContractError) as exc_info:
        adapter.open_slot_journal(
            root=root, execution_id="fixture", execution_identity={"model": "changed"},
            plan=plan, schedule=schedule, slot_index=0, continuation=True,
        )
    assert exc_info.value.code == "journal.continuation_identity_mismatch"
    adapter.execute_slot(
        root=root, execution_id="fixture", execution_identity={"model": "fake"},
        plan=plan, schedule=schedule, slot_index=0, observe_context=observe, continuation=True,
    )
    assert observed[:1] == list(before)
    assert len(observed) == len(set(observed))


def test_observer_exception_is_best_effort_attempt_diagnostic_without_retry(tmp_path: Path) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    schedule = adapter.plan_physical_slots(plan, slot_count=2)
    root = tmp_path / "slot"

    def fail(_context: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("synthetic observer fault")

    with pytest.raises(RuntimeError, match="synthetic"):
        adapter.execute_slot(
            root=root, execution_id="fault", execution_identity={"model": "fake"},
            plan=plan, schedule=schedule, slot_index=0, observe_context=fail,
        )
    diagnostics = ExecutionEvidenceJournal.inspect_diagnostics(root)
    assert diagnostics.snapshot.terminal is None
    assert len(diagnostics.records) == 0
    assert [view.status for view in diagnostics.attempts] == ["failed"]


def test_cleanup_runs_after_observer_exception_without_erasing_attempt(tmp_path: Path) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    schedule = adapter.plan_physical_slots(plan, slot_count=2)
    cleaned: list[bool] = []
    with pytest.raises(RuntimeError, match="fault"):
        adapter.execute_slot(
            root=tmp_path / "slot", execution_id="cleanup", execution_identity={"model": "fake"},
            plan=plan, schedule=schedule, slot_index=0,
            observe_context=lambda _context: (_ for _ in ()).throw(RuntimeError("fault")),
            cleanup=lambda: cleaned.append(True),
        )
    assert cleaned == [True]
    assert ExecutionEvidenceJournal.inspect_diagnostics(tmp_path / "slot").attempts[0].status == "failed"


def test_consumer_validator_is_required_after_raw_digest_admission(tmp_path: Path) -> None:
    path = _plan_path(tmp_path)
    plan = adapter.load_logical_plan(path)
    calls: list[tuple[Path, str, Path]] = []

    def validator(*_args: Any, expected_plan_sha256: str, census_path: Path, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append((path, expected_plan_sha256, census_path))
        return dict(plan.raw), {"census": "validated"}

    assert adapter.validate_sealed_consumer_plan(
        plan_path=path, expected_plan_file_sha256=plan.file_sha256,
        census_path=tmp_path / "census.json", runner_validate_execution_plan=validator,
    ).file_sha256 == plan.file_sha256
    assert calls == [(path, plan.file_sha256, tmp_path / "census.json")]
    with pytest.raises(adapter.SupportShardAdapterError, match="raw-byte"):
        adapter.validate_sealed_consumer_plan(
            plan_path=path, expected_plan_file_sha256="0" * 64,
            census_path=tmp_path / "census.json", runner_validate_execution_plan=validator,
        )


def test_parent_launcher_publishes_exit_diagnostics_without_retry(tmp_path: Path) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    schedule = adapter.plan_physical_slots(plan, slot_count=2)
    root = tmp_path / "slot"
    journal, _ = adapter.open_slot_journal(
        root=root, execution_id="launcher", execution_identity={"model": "fake"},
        plan=plan, schedule=schedule, slot_index=0, continuation=False,
    )
    journal.close()
    receipt = adapter.launch_slot_worker(
        command=[sys.executable, "-c", "raise SystemExit(3)"],
        journal_root=root, mechanics_receipt_path=tmp_path / "exit.json", physical_slot_index=0,
        logical_plan_file_sha256=plan.file_sha256, schedule_sha256=schedule["content_sha256"],
        expected_context_count=4, cwd=tmp_path,
    )
    assert receipt["return_code"] == 3
    assert receipt["accepted_context_count"] == 0
    assert receipt["missing_context_count"] == 4
    assert receipt["attempt_id"] is None
    assert receipt["cwd"] == str(tmp_path.resolve())
    assert receipt["executable"]["path"] == str(Path(sys.executable).resolve())
    assert receipt["executable"]["raw_sha256"] == adapter.file_sha256(sys.executable)
    assert (tmp_path / "exit.json").is_file()


def test_parent_launcher_records_failure_before_journal_creation(tmp_path: Path) -> None:
    receipt = adapter.launch_slot_worker(
        command=[sys.executable, "-c", "raise SystemExit(4)"],
        journal_root=tmp_path / "absent-slot",
        mechanics_receipt_path=tmp_path / "pre-journal-exit.json",
        physical_slot_index=0,
        logical_plan_file_sha256="1" * 64,
        schedule_sha256="2" * 64,
        expected_context_count=3,
    )

    assert receipt["return_code"] == 4
    assert receipt["attempt_id"] is None
    assert receipt["accepted_context_count"] == 0
    assert receipt["missing_context_count"] == 3
    assert receipt["journal_terminal"] is None
    assert receipt["journal_inspection"]["status"] == "absent"


def test_parent_launcher_records_invalid_post_exit_journal_without_fabricated_counts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "invalid-slot"
    child_code = "from pathlib import Path; Path(r'%s').mkdir()" % root
    receipt = adapter.launch_slot_worker(
        command=[sys.executable, "-c", child_code],
        journal_root=root,
        mechanics_receipt_path=tmp_path / "invalid-journal-exit.json",
        physical_slot_index=0,
        logical_plan_file_sha256="1" * 64,
        schedule_sha256="2" * 64,
        expected_context_count=3,
    )

    assert receipt["return_code"] == 0
    assert receipt["accepted_context_count"] is None
    assert receipt["missing_context_count"] is None
    assert receipt["journal_terminal"] is None
    assert receipt["temporary_paths"] is None
    assert receipt["journal_inspection"]["status"] == "invalid"


def test_parent_launcher_records_signal_as_mechanical_exit(tmp_path: Path) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    schedule = adapter.plan_physical_slots(plan, slot_count=2)
    root = tmp_path / "slot"
    journal, _ = adapter.open_slot_journal(
        root=root, execution_id="signal", execution_identity={"model": "fake"},
        plan=plan, schedule=schedule, slot_index=0, continuation=False,
    )
    journal.close()
    receipt = adapter.launch_slot_worker(
        command=["/bin/sh", "-c", "kill -TERM $$"],
        journal_root=root, mechanics_receipt_path=tmp_path / "signal.json", physical_slot_index=0,
        logical_plan_file_sha256=plan.file_sha256, schedule_sha256=schedule["content_sha256"],
        expected_context_count=4,
    )
    assert receipt["return_code"] == -15
    assert receipt["terminating_signal"] == 15
    assert receipt["claim_boundary"].startswith("mechanics_only")


def test_parent_launcher_enforces_bounded_process_group_timeout(tmp_path: Path) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    schedule = adapter.plan_physical_slots(plan, slot_count=2)
    root = tmp_path / "slot"
    journal, _ = adapter.open_slot_journal(
        root=root, execution_id="timeout", execution_identity={"model": "fake"},
        plan=plan, schedule=schedule, slot_index=0, continuation=False,
    )
    journal.close()
    receipt = adapter.launch_slot_worker(
        command=["/bin/sh", "-c", "sleep 30"], journal_root=root,
        mechanics_receipt_path=tmp_path / "timeout.json", physical_slot_index=0,
        logical_plan_file_sha256=plan.file_sha256, schedule_sha256=schedule["content_sha256"],
        expected_context_count=4, timeout_seconds=0.05, termination_grace_seconds=0.5,
    )
    assert receipt["external_signal_sent"] == 15
    assert receipt["return_code"] == -15


def test_parent_launcher_observes_real_sigterm_after_durable_record_without_retry(
    tmp_path: Path,
) -> None:
    plan_path = _plan_path(tmp_path)
    plan = adapter.load_logical_plan(plan_path)
    schedule = adapter.plan_physical_slots(plan, slot_count=2)
    schedule_path = tmp_path / "schedule.json"
    adapter.write_once_json(schedule_path, schedule)
    root = tmp_path / "slot"
    child_code = r"""
import json, os, signal, sys, time
from scripts.research import resumable_natural_boundary_support_completion as a
plan = a.load_logical_plan(sys.argv[1])
schedule = json.loads(open(sys.argv[2], encoding='utf-8').read())
calls = 0
def observe(context):
    global calls
    calls += 1
    if calls > 1:
        time.sleep(30)
    candidate = context['candidate_ids'][0]
    return {'context_id': context['context_id'], 'status': 'measured', 'support_features': {'assessed': True}, 'candidate_scores': {candidate: 0.0}}
try:
    a.execute_slot(root=sys.argv[3], execution_id='signal-child', execution_identity={'fixture': 'signal-child'}, plan=plan, schedule=schedule, slot_index=0, observe_context=observe, install_sigterm_handler=True)
except a.WorkerSIGTERM:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)
"""
    receipt = adapter.launch_slot_worker(
        command=[sys.executable, "-c", child_code, str(plan_path), str(schedule_path), str(root)],
        journal_root=root,
        mechanics_receipt_path=tmp_path / "sigterm-exit.json",
        physical_slot_index=0,
        logical_plan_file_sha256=plan.file_sha256,
        schedule_sha256=schedule["content_sha256"],
        expected_context_count=4,
        terminate_after_first_durable_record=True,
        timeout_seconds=10,
    )
    diagnostics = ExecutionEvidenceJournal.inspect_diagnostics(root)
    assert receipt["return_code"] == -15
    assert receipt["terminating_signal"] == 15
    assert receipt["external_signal_sent"] == 15
    assert receipt["accepted_context_count"] == 1
    assert receipt["missing_context_count"] == 3
    assert len(diagnostics.records) == 1
    assert [attempt.status for attempt in diagnostics.attempts] == ["failed"]


def test_launcher_preserves_unfinished_attempt_and_diagnostic_publication_failure(
    tmp_path: Path,
) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    schedule = adapter.plan_physical_slots(plan, slot_count=2)
    root = tmp_path / "slot"
    journal, _ = adapter.open_slot_journal(
        root=root,
        execution_id="unfinished",
        execution_identity={"fixture": "unfinished"},
        plan=plan,
        schedule=schedule,
        slot_index=0,
        continuation=False,
    )
    journal.close()
    child_code = r"""
import sys
from pathlib import Path
from src.artifacts.evidence_journal import ExecutionEvidenceJournal
j = ExecutionEvidenceJournal.open(root=Path(sys.argv[1]), execution_id='unfinished', execution_identity={'adapter_schema_version': 'natural_boundary_support_completion_adapter.v1', 'execution_identity': {'fixture': 'unfinished'}, 'schedule_sha256': sys.argv[2], 'physical_slot_index': 0}, expected_work_item_ids=['context-0','context-3','context-4','context-7'], context={'logical_plan_file_sha256': sys.argv[3], 'logical_plan_content_sha256': sys.argv[4], 'schedule_sha256': sys.argv[2], 'slot_index': 0})
j.start_attempt()
j.close()
raise SystemExit(7)
"""
    output = tmp_path / "occupied-exit.json"
    adapter.write_once_json(output, {"occupied": True})
    with pytest.raises(adapter.SupportShardAdapterError, match="overwrite"):
        adapter.launch_slot_worker(
            command=[sys.executable, "-c", child_code, str(root), schedule["content_sha256"], plan.file_sha256, plan.content_sha256],
            journal_root=root,
            mechanics_receipt_path=output,
            physical_slot_index=0,
            logical_plan_file_sha256=plan.file_sha256,
            schedule_sha256=schedule["content_sha256"],
            expected_context_count=4,
        )
    diagnostics = ExecutionEvidenceJournal.inspect_diagnostics(root)
    assert [attempt.status for attempt in diagnostics.attempts] == ["unfinished"]
    assert diagnostics.records == ()


def test_materializer_requires_complete_eligible_records_and_is_history_independent(tmp_path: Path) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    schedule = adapter.plan_physical_slots(plan, slot_count=8)

    def complete(root: Path, *, interrupt: bool) -> dict[int, dict[str, Any]]:
        def observe(context: dict[str, Any]) -> dict[str, Any]:
            return _observation(context)

        roots = []
        for slot in range(8):
            slot_root = root / f"slot-{slot}"
            roots.append(slot_root)
            adapter.execute_slot(
                root=slot_root, execution_id=f"fixture-{slot}", execution_identity={"model": "fake"},
                plan=plan, schedule=schedule, slot_index=slot, observe_context=observe,
                stop_after_records=0 if interrupt and slot == 0 else None,
            )
            if interrupt and slot == 0:
                adapter.execute_slot(
                    root=slot_root, execution_id=f"fixture-{slot}", execution_identity={"model": "fake"},
                    plan=plan, schedule=schedule, slot_index=slot, observe_context=observe, continuation=True,
                )
        return dict(adapter.materialize_legacy_shard_receipts(
            plan=plan, schedule=schedule, slot_roots=roots,
            execution_identity={"model": "fake"},
            census_binding=_fixture_census(),
            receipt_set_validator=_accept_receipt_set,
        ))

    uninterrupted = complete(tmp_path / "uninterrupted", interrupt=False)
    resumed = complete(tmp_path / "resumed", interrupt=True)
    assert [json.dumps(uninterrupted[index], sort_keys=True) for index in range(8)] == [
        json.dumps(resumed[index], sort_keys=True) for index in range(8)
    ]

    incomplete_root = tmp_path / "incomplete"
    adapter.execute_slot(
        root=incomplete_root, execution_id="incomplete", execution_identity={"model": "fake"},
        plan=plan, schedule=schedule, slot_index=0, observe_context=_observation,
    )
    with pytest.raises(adapter.SupportShardAdapterError, match="every physical slot"):
        adapter.materialize_legacy_shard_receipts(
            plan=plan, schedule=schedule, slot_roots=[incomplete_root],
            execution_identity={"model": "fake"},
            census_binding=_fixture_census(),
            receipt_set_validator=_accept_receipt_set,
        )


def test_materializer_refuses_mechanically_accepted_failure_payload(tmp_path: Path) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    schedule = adapter.plan_physical_slots(plan, slot_count=8)
    roots = []
    for slot in range(8):
        root = tmp_path / f"slot-{slot}"
        roots.append(root)
        def observation(context: dict[str, Any], *, target: int = slot) -> dict[str, Any]:
            value = _observation(context)
            if target == 0:
                value["status"] = "failed"
                value["support_features"] = None
            return value
        adapter.execute_slot(
            root=root, execution_id=f"failure-{slot}", execution_identity={"model": "fake"},
            plan=plan, schedule=schedule, slot_index=slot, observe_context=observation,
        )
    with pytest.raises(adapter.SupportShardAdapterError, match="ineligible"):
        adapter.materialize_legacy_shard_receipts(
            plan=plan, schedule=schedule, slot_roots=roots,
            execution_identity={"model": "fake"},
            census_binding=_fixture_census(),
            receipt_set_validator=_accept_receipt_set,
        )


def test_bounded_terminal_and_legacy_materializer_reject_foreign_slot_identity(
    tmp_path: Path,
) -> None:
    full = adapter.load_logical_plan(_plan_path(tmp_path))
    plan = adapter.project_logical_contexts(full, context_ids=["context-0", "context-1"])
    schedule = adapter.plan_physical_slots(plan, slot_count=1)
    identity = {"fixture": "bound"}
    root = tmp_path / "slot"
    adapter.execute_slot(
        root=root,
        execution_id="bounded",
        execution_identity=identity,
        plan=plan,
        schedule=schedule,
        slot_index=0,
        observe_context=_observation,
    )
    receipt = adapter.materialize_bounded_mechanics_terminal(
        plan=plan,
        schedule=schedule,
        slot_roots=[root],
        execution_identity=identity,
        output_path=tmp_path / "bounded.json",
    )
    assert receipt["context_ids"] == ["context-0", "context-1"]
    assert receipt["claim_boundary"].startswith("bounded_live_mechanics_only")
    with pytest.raises(adapter.SupportShardAdapterError, match="identity is foreign"):
        adapter.materialize_bounded_mechanics_terminal(
            plan=plan,
            schedule=schedule,
            slot_roots=[root],
            execution_identity={"fixture": "foreign"},
            output_path=tmp_path / "foreign.json",
        )


def test_legacy_materializer_rejects_swapped_slots_and_foreign_denominator(
    tmp_path: Path,
) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    schedule = adapter.plan_physical_slots(plan, slot_count=2)
    identity = {"model": "fake"}
    roots = [tmp_path / f"valid-slot-{slot}" for slot in range(2)]
    for slot, root in enumerate(roots):
        adapter.execute_slot(
            root=root,
            execution_id=f"valid-{slot}",
            execution_identity=identity,
            plan=plan,
            schedule=schedule,
            slot_index=slot,
            observe_context=_observation,
        )
    with pytest.raises(adapter.SupportShardAdapterError, match="identity is foreign"):
        adapter.materialize_legacy_shard_receipts(
            plan=plan,
            schedule=schedule,
            slot_roots=list(reversed(roots)),
            execution_identity=identity,
            census_binding=_fixture_census(),
            receipt_set_validator=_accept_receipt_set,
        )

    foreign_root = tmp_path / "foreign-slot-0"
    expected_ids = tuple(schedule["slots"][0]["context_ids"])
    slot_identity = {
        "adapter_schema_version": adapter.ADAPTER_SCHEMA_VERSION,
        "execution_identity": identity,
        "schedule_sha256": schedule["content_sha256"],
        "physical_slot_index": 0,
    }
    slot_context = {
        "logical_plan_file_sha256": plan.file_sha256,
        "logical_plan_content_sha256": plan.content_sha256,
        "schedule_sha256": schedule["content_sha256"],
        "slot_index": 0,
    }
    journal = ExecutionEvidenceJournal.create(
        root=foreign_root,
        execution_id="foreign-denominator",
        execution_identity=slot_identity,
        expected_work_item_ids=list(reversed(expected_ids)),
        context=slot_context,
    )
    try:
        attempt = journal.start_attempt()
        contexts = {item.context_id: item.source for item in plan.contexts}
        for context_id in reversed(expected_ids):
            journal.append_record(
                work_item_id=context_id,
                payload=_observation(dict(contexts[context_id])),
                attempt_id=attempt,
            )
        journal.finalize()
    finally:
        journal.close()
    with pytest.raises(adapter.SupportShardAdapterError, match="denominator is foreign"):
        adapter.materialize_legacy_shard_receipts(
            plan=plan,
            schedule=schedule,
            slot_roots=[foreign_root, roots[1]],
            execution_identity=identity,
            census_binding=_fixture_census(),
            receipt_set_validator=_accept_receipt_set,
        )


def test_legacy_materializer_validates_all_receipts_before_publication(
    tmp_path: Path,
) -> None:
    plan = adapter.load_logical_plan(_plan_path(tmp_path))
    schedule = adapter.plan_physical_slots(plan, slot_count=2)
    identity = {"model": "fake"}
    roots = [tmp_path / f"slot-{slot}" for slot in range(2)]
    for slot, root in enumerate(roots):
        adapter.execute_slot(
            root=root,
            execution_id=f"reject-{slot}",
            execution_identity=identity,
            plan=plan,
            schedule=schedule,
            slot_index=slot,
            observe_context=_observation,
        )

    def reject(_receipts: Mapping[int, Mapping[str, Any]]) -> Mapping[str, Any]:
        raise RuntimeError("fixture merger rejection")

    output_root = tmp_path / "receipts"
    with pytest.raises(adapter.SupportShardAdapterError, match="merger validator rejected"):
        adapter.materialize_legacy_shard_receipts(
            plan=plan,
            schedule=schedule,
            slot_roots=roots,
            execution_identity=identity,
            census_binding=_fixture_census(),
            receipt_set_validator=reject,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_unchanged_active_merger_accepts_cpu_materialized_receipt_shape() -> None:
    """Load the sibling only for read-only compatibility validation."""

    sibling_root = Path("../research-probes").resolve()
    import scripts.research

    sibling_research = str(sibling_root / "scripts" / "research")
    if sibling_research not in scripts.research.__path__:
        scripts.research.__path__.append(sibling_research)
    sys.modules.pop("scripts.research.run_natural_boundary_support_completion", None)
    sys.modules.pop("scripts.research.merge_natural_boundary_support_completion", None)
    from scripts.research import merge_natural_boundary_support_completion as merger
    from scripts.research import run_natural_boundary_support_completion as runner

    plan_path = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-06-natural-boundary-routing-history-replication/"
        "support-completion-plan-v1/plan.json"
    )
    plan = adapter.load_logical_plan(plan_path)
    census = runner.validate_admission_census(runner.DEFAULT_CENSUS_PATH, plan=plan.raw)
    receipts: dict[int, dict[str, Any]] = {}
    for shard in range(8):
        observations = []
        for context in sorted((item for item in plan.contexts if item.shard_index == shard), key=lambda item: item.plan_position):
            scores = {candidate_id: 0.0 for candidate_id in context.source["candidate_ids"]}
            observations.append({
                "context_id": context.context_id,
                "status": "measured",
                "candidate_ids": list(context.source["candidate_ids"]),
                "candidate_scores": scores,
                "candidate_score_count": context.scalar_equivalent_forward_count,
                "candidate_scores_sha256": json_sha256(dict(sorted(scores.items()))),
                "support_features": {"assessed": True, "peak_lift": 0.0, "local_concentration": 0.0},
                "failure_count": 0,
            })
        receipts[shard] = adapter.build_legacy_receipt(
            plan, shard_index=shard, observations=observations, census_binding=census
        )
    result = merger.merge_support_receipts(
        plan.raw,
        receipts,
        prior_support=Path(
            "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
            "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/"
            "s-step2444-final-support.json"
        ),
        h0_source=plan.raw["h0_lineage"]["source"]["path"],
        input_plan_sha256=plan.file_sha256,
        input_census_binding=census,
    )
    assert result["receipt"]["record_count"] == 220


def test_full_plan_fresh_process_interruption_resume_has_identical_eight_receipts(
    tmp_path: Path,
) -> None:
    """CPU-only equivalence over the sealed 200-context logical plan."""

    sibling_root = Path("../research-probes").resolve()
    import scripts.research

    sibling_research = str(sibling_root / "scripts" / "research")
    if sibling_research not in scripts.research.__path__:
        scripts.research.__path__.append(sibling_research)
    sys.modules.pop("scripts.research.run_natural_boundary_support_completion", None)
    sys.modules.pop("scripts.research.merge_natural_boundary_support_completion", None)
    from scripts.research import merge_natural_boundary_support_completion as merger
    from scripts.research import run_natural_boundary_support_completion as runner

    plan_path = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-06-natural-boundary-routing-history-replication/"
        "support-completion-plan-v1/plan.json"
    )
    plan = adapter.load_logical_plan(plan_path)
    schedule = adapter.plan_physical_slots(plan)
    schedule_path = tmp_path / "schedule.json"
    adapter.write_once_json(schedule_path, schedule)
    census = runner.validate_admission_census(runner.DEFAULT_CENSUS_PATH, plan=plan.raw)

    def observe(context: dict[str, Any]) -> dict[str, Any]:
        scores = {candidate_id: 0.0 for candidate_id in context["candidate_ids"]}
        return {
            "context_id": context["context_id"], "status": "measured",
            "candidate_ids": list(context["candidate_ids"]), "candidate_scores": scores,
            "candidate_score_count": context["scalar_equivalent_forward_count"],
            "candidate_scores_sha256": json_sha256(dict(sorted(scores.items()))),
            "support_features": {"assessed": True, "peak_lift": 0.0, "local_concentration": 0.0}, "failure_count": 0,
        }

    def materialize(root: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
        roots = [root / f"slot-{slot}" for slot in range(8)]
        def validate_with_current_merger(
            receipts: Mapping[int, Mapping[str, Any]],
        ) -> Mapping[str, Any]:
            return merger.merge_support_receipts(
                plan.raw,
                receipts,
                prior_support=Path(
                    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
                    "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/"
                    "s-step2444-final-support.json"
                ),
                h0_source=plan.raw["h0_lineage"]["source"]["path"],
                input_plan_sha256=plan.file_sha256,
                input_census_binding=census,
            )

        receipts = adapter.materialize_legacy_shard_receipts(
            plan=plan, schedule=schedule, slot_roots=roots,
            execution_identity={"fixture": "full"},
            census_binding=census,
            receipt_set_validator=validate_with_current_merger,
        )
        diagnostics = [ExecutionEvidenceJournal.inspect_diagnostics(path) for path in roots]
        return dict(receipts), [
            {
                "record_digests": [record.record_digest for record in item.records],
                "payload_fingerprints": [
                    record.payload_fingerprint for record in item.records
                ],
                "attempts": [attempt.status for attempt in item.attempts],
            }
            for item in diagnostics
        ]

    uninterrupted_root = tmp_path / "uninterrupted"
    for slot in range(8):
        adapter.execute_slot(
            root=uninterrupted_root / f"slot-{slot}", execution_id=f"full-{slot}",
            execution_identity={"fixture": "full"}, plan=plan, schedule=schedule,
            slot_index=slot, observe_context=observe,
        )
    uninterrupted, uninterrupted_diagnostics = materialize(uninterrupted_root)

    resumed_root = tmp_path / "resumed"
    adapter.execute_slot(
        root=resumed_root / "slot-0", execution_id="full-0", execution_identity={"fixture": "full"},
        plan=plan, schedule=schedule, slot_index=0, observe_context=observe, stop_after_records=1,
    )
    child_code = """
import json, sys
from scripts.research import resumable_natural_boundary_support_completion as a
plan = a.load_logical_plan(sys.argv[1])
schedule = json.loads(open(sys.argv[2], encoding='utf-8').read())
def observe(context):
    scores = {candidate_id: 0.0 for candidate_id in context['candidate_ids']}
    return {'context_id': context['context_id'], 'status': 'measured', 'candidate_ids': list(context['candidate_ids']), 'candidate_scores': scores, 'candidate_score_count': context['scalar_equivalent_forward_count'], 'candidate_scores_sha256': a.json_sha256(dict(sorted(scores.items()))), 'support_features': {'assessed': True, 'peak_lift': 0.0, 'local_concentration': 0.0}, 'failure_count': 0}
a.execute_slot(root=sys.argv[3], execution_id='full-0', execution_identity={'fixture': 'full'}, plan=plan, schedule=schedule, slot_index=0, observe_context=observe, continuation=True)
"""
    subprocess.run(
        [sys.executable, "-c", child_code, str(plan_path), str(schedule_path), str(resumed_root / "slot-0")],
        check=True, cwd=Path.cwd(), env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    for slot in range(1, 8):
        adapter.execute_slot(
            root=resumed_root / f"slot-{slot}", execution_id=f"full-{slot}",
            execution_identity={"fixture": "full"}, plan=plan, schedule=schedule,
            slot_index=slot, observe_context=observe,
        )
    resumed, resumed_diagnostics = materialize(resumed_root)
    uninterrupted_bytes = [json.dumps(uninterrupted[index], sort_keys=True, separators=(",", ":")).encode() + b"\n" for index in range(8)]
    resumed_bytes = [json.dumps(resumed[index], sort_keys=True, separators=(",", ":")).encode() + b"\n" for index in range(8)]
    assert uninterrupted_bytes == resumed_bytes
    assert [hashlib.sha256(item).hexdigest() for item in uninterrupted_bytes] == [hashlib.sha256(item).hexdigest() for item in resumed_bytes]
    assert [item["payload_fingerprints"] for item in uninterrupted_diagnostics] == [
        item["payload_fingerprints"] for item in resumed_diagnostics
    ]
    assert [item["record_digests"] for item in uninterrupted_diagnostics] != [
        item["record_digests"] for item in resumed_diagnostics
    ]
    assert uninterrupted_diagnostics[0]["attempts"] == ["unfinished"]
    assert resumed_diagnostics[0]["attempts"] == ["unfinished", "unfinished"]
    result = merger.merge_support_receipts(
        plan.raw, resumed,
        prior_support=Path(
            "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
            "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/s-step2444-final-support.json"
        ),
        h0_source=plan.raw["h0_lineage"]["source"]["path"],
        input_plan_sha256=plan.file_sha256, input_census_binding=census,
    )
    assert result["receipt"]["record_count"] == 220
    receipt_path = os.environ.get("COORDEXP_CPU_EQUIVALENCE_RECEIPT")
    if receipt_path:
        proof = {
            "schema_version": adapter.MECHANICS_SCHEMA_VERSION,
            "kind": "cpu_interruption_equivalence",
            "logical_plan_file_sha256": plan.file_sha256,
            "schedule_sha256": schedule["content_sha256"],
            "interruption": {"slot_index": 0, "after_durable_records": 1, "continuation": "fresh_process_exact_identity"},
            "missing_set_transition": {
                "before_interruption": len(schedule["slots"][0]["context_ids"]),
                "after_first_durable_record": len(schedule["slots"][0]["context_ids"]) - 1,
                "after_continuation": 0,
            },
            "record_digests": {
                "uninterrupted": [item["record_digests"] for item in uninterrupted_diagnostics],
                "resumed": [item["record_digests"] for item in resumed_diagnostics],
                "equal": [item["record_digests"] for item in uninterrupted_diagnostics]
                == [item["record_digests"] for item in resumed_diagnostics],
                "expected_equal": False,
                "reason": "record envelopes bind distinct attempt identifiers",
            },
            "payload_fingerprints": {
                "uninterrupted": [
                    item["payload_fingerprints"] for item in uninterrupted_diagnostics
                ],
                "resumed": [
                    item["payload_fingerprints"] for item in resumed_diagnostics
                ],
                "equal": [
                    item["payload_fingerprints"] for item in uninterrupted_diagnostics
                ]
                == [item["payload_fingerprints"] for item in resumed_diagnostics],
            },
            "terminal_receipt_bytes_equal": uninterrupted_bytes == resumed_bytes,
            "terminal_receipt_sha256": [hashlib.sha256(item).hexdigest() for item in resumed_bytes],
            "merger": {"verdict": "accepted", "record_count": result["receipt"]["record_count"]},
            "attempt_history": {"uninterrupted_slot_0": uninterrupted_diagnostics[0]["attempts"], "resumed_slot_0": resumed_diagnostics[0]["attempts"]},
            "claim_boundary": "cpu_deterministic_fixture_mechanics_only_no_model_or_support_prevalence_claim",
        }
        proof["content_sha256"] = json_sha256(proof)
        adapter.write_once_json(receipt_path, proof)
