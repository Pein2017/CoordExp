from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from copy import deepcopy

import pytest
import torch

from scripts.research import run_s_natural_boundary_k_n_h_cohort as cohort
from scripts.research import s_natural_boundary_k_n_h_live_executor as live


def test_direct_file_import_smoke() -> None:
    script = Path(live.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _event() -> dict[str, object]:
    return {
        "event_id": "gt:5001:15",
        "gt_owner_id": "gt:5001:15",
        "image_id": 5001,
        "owner_refs": {"gt_owner_id": "gt:5001:15", "source_panel_object_index": 15, "derived_panel_object_index": 15},
        "natural_boundary": {
            "prefix_token_ids": [7, 8],
            "prefix_sha256": cohort.sha256_json([7, 8]),
        },
    }


def _row() -> dict[str, object]:
    return {
        "gt_owner_id": "gt:5001:15",
        "image_id": 5001,
        "source_panel_object_index": 15,
        "derived_panel_object_index": 15,
        "exact_prefix_token_ids": [7, 8],
        "exact_prefix_sha256": cohort.sha256_json([7, 8]),
        "geometry": {
            "launch_eligible": True,
            "image_plan_identity": {"cell_count": 4},
            "same_class_competitor_owner_id": None,
            "image_cell_regions": {
                "a_exclusive": [0], "b_exclusive": [1], "background": [2], "shared_core": [3]
            },
        },
        "same_class_competitor_owner_id": None,
    }


def _pre_gpu_identity() -> dict[str, object]:
    fixture = runpy.run_path(str(Path(__file__).with_name("test_run_s_natural_boundary_k_n_h_cohort.py")))
    return fixture["_executor_identity"]()["pre_gpu"]


def _runtime_versions() -> dict[str, str]:
    return {
        "python_version": "3.11",
        "torch_version": "2.7.0",
        "transformers_version": "4.57.1",
    }


def test_executor_identity_retains_and_hashes_full_runtime_cohort_preflight() -> None:
    fixture = runpy.run_path(str(Path(__file__).with_name("test_run_s_natural_boundary_k_n_h_cohort.py")))
    source = fixture["_executor_identity"]()
    observed = source["observed"]
    identity = live._executor_identity(
        source["pre_gpu"],
        model=observed["model"],
        backend=observed["backend"],
        device=observed["device"],
        cuda=observed["cuda"],
        config_sha256=observed["config_sha256"],
        runtime_versions=observed["runtime_versions"],
        full_runtime_cohort_preflight=source["full_runtime_cohort_preflight"],
    )
    assert identity["full_runtime_cohort_preflight"] == source["full_runtime_cohort_preflight"]
    body = dict(identity)
    identity_sha256 = body.pop("identity_sha256")
    assert identity_sha256 == cohort.sha256_json(body)
    tampered = dict(identity)
    tampered_preflight = deepcopy(tampered["full_runtime_cohort_preflight"])
    tampered_preflight["event_count"] = 10
    tampered["full_runtime_cohort_preflight"] = tampered_preflight
    tampered_body = dict(tampered)
    tampered_body.pop("identity_sha256")
    assert identity_sha256 != cohort.sha256_json(tampered_body)


def _ledger_history_fixture() -> tuple[SimpleNamespace, dict[str, object], tuple[int, ...], tuple[int, ...]]:
    wrapper = live.legacy.static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
    )
    a_row = (100, 41, 101, 102, 10, 11, 12, 13, 103)
    b_row = (100, 42, 101, 102, 20, 21, 22, 23, 103)
    a_history = a_row
    b_history = a_row + b_row
    boundaries = [
        {
            "generated_order": 0,
            "row_start_step": 0,
            "span_end_step": len(a_row) - 1,
            "closure_step": len(a_row) - 1,
            "closure_token_id": 103,
            "match_status": "unmatched",
            "gt_owner_id": None,
        },
        {
            "generated_order": 1,
            "row_start_step": len(a_row),
            "span_end_step": len(b_history) - 1,
            "closure_step": len(b_history) - 1,
            "closure_token_id": 103,
            "match_status": "tp",
            "gt_owner_id": "gt:1:a",
        },
    ]
    trace = {
        "generated_token_ids": list(b_history),
        "generated_token_ids_sha256": live.legacy.sha256_token_ids(b_history),
        "rows": [list(a_row), list(b_row)],
        "row_token_ids_sha256": [
            live.legacy.sha256_token_ids(a_row),
            live.legacy.sha256_token_ids(b_row),
        ],
    }
    ledger = {
        "gt:1:a": {
            "natural_boundary": 0,
            "covered_owner_ids": (),
            "latest_covered_owner_id": None,
            "exact_prefix_token_ids": a_history,
            "exact_prefix_sha256": live.legacy.sha256_token_ids(a_history),
            "strict_complete_row": True,
        },
        "gt:1:b": {
            "natural_boundary": 1,
            "covered_owner_ids": ("gt:1:a",),
            "latest_covered_owner_id": "gt:1:a",
            "exact_prefix_token_ids": b_history,
            "exact_prefix_sha256": live.legacy.sha256_token_ids(b_history),
            "strict_complete_row": False,
            "generated_row_boundaries": boundaries,
            "due_boundary_evidence": {
                "covered_owner_ids": ["gt:1:a"],
                "latest_covered_owner_id": "gt:1:a",
                "latest_covered_generated_order": 1,
            },
        },
    }
    runtime = SimpleNamespace(
        h0=trace,
        prompt_ids=torch.tensor([[5, 6]], dtype=torch.long),
        owners=[{"owner_id": "gt:1:a"}, {"owner_id": "gt:1:b"}],
    )
    adapter = SimpleNamespace(
        checkpoint="S",
        panel_rows={"1": runtime},
        h0_rows={"1": trace},
        h0_ledger_records={"1": ledger},
        wrapper_contract=wrapper,
        model_device=torch.device("cpu"),
        tokenizer=None,
    )
    event = {
        "image_id": 1,
        "gt_owner_id": "gt:1:b",
        "A_B": {
            "S": {
                "pair_status": "verified_pair",
                "A_latest_covered": {
                    "gt_owner_id": "gt:1:a",
                    "source_panel_object_index": 0,
                    "natural_boundary": 0,
                    "strict_complete_row": True,
                },
                "B_verified_uncovered": {
                    "gt_owner_id": "gt:1:b",
                    "verified_support": True,
                    "strict_complete_row": False,
                    "natural_boundary": 1,
                    "exact_prefix_sha256": live.legacy.sha256_token_ids(b_history),
                },
            }
        },
        "target_row_token_ids": list(b_row),
    }
    return adapter, event, a_history, b_history


def test_ledger_exact_history_resolver_binds_full_physical_prefix_and_prompt() -> None:
    adapter, event, a_history, b_history = _ledger_history_fixture()
    resolved = live._ledger_exact_history_resolver(adapter, event)
    assert tuple(resolved["exact_A_history_token_ids"]) == a_history
    assert tuple(resolved["exact_history_token_ids"]) == b_history
    assert tuple(resolved["completed_row_ids"]) == (a_history, b_history[len(a_history) :])
    assert tuple(resolved["latest_row_ids"]) == b_history[len(a_history) :]
    context = live.legacy._make_event_context(
        adapter,
        event,
        history_resolver=live._ledger_exact_history_resolver,
    )
    assert context.prefix_ids.detach().cpu().tolist() == [[5, 6, *b_history, 100]]
    assert context.prefix_receipt["h0"]["exact_generated_history_prefix_sha256"] == live.legacy.sha256_token_ids(b_history)
    assert context.latest_row_ids == list(b_history[len(a_history) :])
    assert context.completed_row_ids == (a_history, b_history[len(a_history) :])


def test_ledger_exact_history_resolver_rejects_altered_exact_history() -> None:
    adapter, event, _a_history, b_history = _ledger_history_fixture()
    adapter.h0_ledger_records["1"]["gt:1:b"]["exact_prefix_token_ids"] = (999, *b_history[1:])
    adapter.h0_ledger_records["1"]["gt:1:b"]["exact_prefix_sha256"] = live.legacy.sha256_token_ids((999, *b_history[1:]))
    with pytest.raises(live.LiveExecutorError, match="literal H0 generated prefix"):
        live._ledger_exact_history_resolver(adapter, event)


@pytest.mark.parametrize("tamper", ("mid_row", "order", "boundary", "owner"))
def test_ledger_exact_history_resolver_rejects_physical_boundary_drift(tamper: str) -> None:
    adapter, event, a_history, b_history = _ledger_history_fixture()
    b_record = adapter.h0_ledger_records["1"]["gt:1:b"]
    if tamper == "mid_row":
        partial = b_history[:-1]
        b_record["exact_prefix_token_ids"] = partial
        b_record["exact_prefix_sha256"] = live.legacy.sha256_token_ids(partial)
    elif tamper == "order":
        b_record["due_boundary_evidence"]["latest_covered_generated_order"] = 0
    elif tamper == "boundary":
        b_record["generated_row_boundaries"][1]["row_start_step"] = len(a_history) - 1
    else:
        b_record["generated_row_boundaries"][1]["gt_owner_id"] = "gt:1:wrong"
    with pytest.raises(live.LiveExecutorError):
        live._ledger_exact_history_resolver(adapter, event)


def test_prefix_and_geometry_are_bound_to_census() -> None:
    event = _event()
    row = _row()
    event2 = _event()
    event2.update({"event_id": "gt:5002:16", "gt_owner_id": "gt:5002:16", "image_id": 5002, "owner_refs": {"gt_owner_id": "gt:5002:16", "source_panel_object_index": 16, "derived_panel_object_index": 16}})
    row2 = _row()
    row2.update({"gt_owner_id": "gt:5002:16", "image_id": 5002, "source_panel_object_index": 16, "derived_panel_object_index": 16})
    bound = live._runtime_event(event, row)
    assert bound["gt_owner_id"] == event["event_id"]
    assert bound["image_cell_regions"]["b_exclusive"] == [1]
    row["exact_prefix_token_ids"] = [99]
    with pytest.raises(live.LiveExecutorError, match="prefix"):
        live._runtime_event(event, row)


def test_source_index_and_geometry_fail_closed() -> None:
    event = _event()
    row = _row()
    row["source_panel_object_index"] = 16
    with pytest.raises(live.LiveExecutorError, match="source-panel"):
        live._runtime_event(event, row)
    row = _row()
    row["geometry"] = {"launch_eligible": False}
    with pytest.raises(live.LiveExecutorError, match="geometry"):
        live._runtime_event(event, row)


def _raw_owner_panel(*, image_id: int = 5001, count: int = 1) -> dict[str, object]:
    objects = []
    for index in range(count):
        x1 = 101 + index * 37
        y1 = 103 + index * 29
        objects.append(
            {
                "bbox": [
                    f"<|coord_{x1}|>",
                    f"<|coord_{y1}|>",
                    f"<|coord_{x1 + 211}|>",
                    f"<|coord_{y1 + 173}|>",
                ],
                "description": f"class-{index}",
                "coco_ann_id": f"ann-{index}",
            }
        )
    return {"image_id": image_id, "width": 1001, "height": 1001, "objects": objects}


def _raw_mapping_and_plan(*, count: int = 1) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    source = _raw_owner_panel(count=count)
    derived = deepcopy(source)
    _owners, mapping, _receipt = live.legacy._build_source_derived_owner_mapping(source, derived)
    return live._ephemeral_exact_raw_owner_mapping(source, derived, mapping), {
        "image_width": 1001.0,
        "image_height": 1001.0,
        "grid_rows": 7,
        "grid_cols": 7,
        "cell_count": 49,
    }


def test_fractional_weights_use_exact_raw_bbox_not_historical_rounding() -> None:
    mapping, plan = _raw_mapping_and_plan()
    owner = mapping["gt:5001:0"]
    raw_weights = live._mapping_cell_weights(owner, plan)
    rounded_weights = live._mapping_cell_weights(
        {**owner, "source_pixel_bbox_exact_raw": list(owner["source_pixel_bbox"])},
        plan,
    )
    assert raw_weights
    assert raw_weights != rounded_weights


@pytest.mark.parametrize(
    "mutate",
    (
        lambda mapping: mapping.pop("source_pixel_bbox_exact_raw"),
        lambda mapping: mapping.update({"source_pixel_bbox_exact_raw": [1.0, 2.0, "bad", 4.0]}),
    ),
)
def test_exact_raw_mapping_malformed_or_mismatched_fails_closed(mutate: object) -> None:
    mapping, plan = _raw_mapping_and_plan()
    owner = dict(mapping["gt:5001:0"])
    if callable(mutate):
        mutate(owner)
    with pytest.raises(live.LiveExecutorError, match="exact raw"):
        live._mapping_cell_weights(owner, plan)


def test_source_derived_raw_mapping_identity_mismatch_fails_closed() -> None:
    source = _raw_owner_panel()
    derived = deepcopy(source)
    _owners, mapping, _receipt = live.legacy._build_source_derived_owner_mapping(source, derived)
    mapping["gt:5001:0"]["source_pixel_bbox"] = [999, 999, 999, 999]
    with pytest.raises(live.LiveExecutorError, match="mapping identity"):
        live._ephemeral_exact_raw_owner_mapping(source, derived, mapping)


def test_source_derived_exact_raw_divergence_fails_closed_even_when_rounded_matches() -> None:
    source = _raw_owner_panel()
    source["width"] = 100
    source["height"] = 100
    derived = deepcopy(source)
    derived["objects"][0]["bbox"][0] = "<|coord_102|>"
    _owners, mapping, _receipt = live.legacy._build_source_derived_owner_mapping(source, derived)
    assert mapping["gt:5001:0"]["source_pixel_bbox"] == mapping["gt:5001:0"]["derived_pixel_bbox"]
    with pytest.raises(live.LiveExecutorError, match="source/derived raw identity"):
        live._ephemeral_exact_raw_owner_mapping(source, derived, mapping)


def test_fresh_eleven_owner_exact_raw_projection_passes_cpu_only() -> None:
    mapping, plan = _raw_mapping_and_plan(count=11)
    projected = {
        owner_id: live._mapping_cell_weights(owner, plan)
        for owner_id, owner in mapping.items()
    }
    assert len(projected) == 11
    assert all(weights for weights in projected.values())


def test_fresh_production_eleven_event_cpu_preflight_passes_before_model_load() -> None:
    root = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-06-natural-boundary-routing-history-replication/"
        "cpu-census-v3-native-fn-supersession-v1"
    )
    cohort_path = root / "s-context-cohort.json"
    cohort_manifest_path = root / "s-context-cohort.manifest.json"
    manifest_path = root / "admitted-event-manifest.json"
    census_path = root / "admission-census.json"
    if not all(path.is_file() for path in (cohort_path, cohort_manifest_path, manifest_path, census_path)):
        pytest.skip("fresh 11-event production CPU artifacts are unavailable")
    cohort_doc = json.loads(cohort_path.read_text(encoding="utf-8"))
    admitted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_panel_path = Path(cohort_doc["sources"]["source_panel"]["path"])
    derived_panel_path = Path(cohort_doc["sources"]["derived_panel"]["path"])
    h0_dir = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-05-static-dynamic-owner-interface-crossover/h0/"
        "qwen3-vl-2b-static-dynamic-owner-interface-s-step2444-h0-repair1"
    )
    if not all(path.is_file() for path in (source_panel_path, derived_panel_path, h0_dir / "image_plan.jsonl")):
        pytest.skip("fresh 11-event source/derived/H0 artifacts are unavailable")
    orchestrator = live.legacy.OwnerInterfaceOrchestrator(
        checkpoint="S",
        stage="all",
        output_dir=root / "cpu-preflight-test-output",
        config_path=live.legacy.CHECKPOINTS["S"]["config"],
        panel_path=derived_panel_path,
        cohort_path=cohort_path,
        h0_root=h0_dir.parent,
        h0_dir=h0_dir,
    )
    orchestrator._load_cpu_contract()
    paths = {
        "manifest": manifest_path,
        "census": census_path,
        "cohort": cohort_path,
        "cohort_manifest": cohort_manifest_path,
        "panel": derived_panel_path,
        "h0_dir": h0_dir,
    }
    receipt = live._preflight_full_runtime_cohort(orchestrator, paths)
    assert receipt["status"] == "passed"
    assert receipt["event_count"] == 11
    context_bindings = receipt["processor_context_bindings"]
    assert len(context_bindings) == 11
    binding_by_id = {item["event_id"]: item for item in context_bindings}
    assert {"gt:4134:34", "gt:16228:15", "gt:16228:43", "gt:16228:48"} <= set(binding_by_id)
    assert all(item["latest_terminal_key_position_count"] == 1 for item in context_bindings)
    assert all(item["exact_history_sha256"] == item["manifest_history_sha256"] for item in context_bindings)
    admitted_by_id = {event["event_id"]: event for event in admitted_manifest["events"]}
    assert len(admitted_by_id) == 11
    for context in cohort_doc["events"]:
        event_id = context["gt_owner_id"]
        admitted = admitted_by_id[event_id]
        frozen_geometry = context["geometry_by_checkpoint"]["S"]
        assert frozen_geometry["geometry_sha256"] == admitted["geometry"]["geometry_sha256"]
        assert frozen_geometry["image_cell_regions"] == admitted["geometry"]["image_cell_regions"]
    from src.data import load_raw_examples

    source_values = load_raw_examples(source_panel_path)
    derived_values = load_raw_examples(derived_panel_path)
    source_by_image = {live.legacy._raw_image_id(value): value for value in source_values}
    derived_by_image = {live.legacy._raw_image_id(value): value for value in derived_values}
    _owners, rounded_mapping, _mapping_receipt = live.legacy._build_source_derived_owner_mapping(
        source_by_image["5001"], derived_by_image["5001"]
    )
    exact_mapping = live._ephemeral_exact_raw_owner_mapping(
        source_by_image["5001"], derived_by_image["5001"], rounded_mapping
    )
    image_plans = live._load_h0_image_plans(h0_dir / "image_plan.jsonl", derived_values)
    exact_weights = {
        owner_id: live._mapping_cell_weights(mapping, image_plans["5001"])
        for owner_id, mapping in exact_mapping.items()
    }
    rounded_weights = {
        owner_id: live._mapping_cell_weights(
            {**mapping, "source_pixel_bbox_exact_raw": mapping["source_pixel_bbox"]},
            image_plans["5001"],
        )
        for owner_id, mapping in exact_mapping.items()
    }
    raw_occupied = {index for weights in exact_weights.values() for index in weights}
    rounded_occupied = {index for weights in rounded_weights.values() for index in weights}
    assert sorted(raw_occupied - rounded_occupied) == [93, 94, 104]
    gt5001_geometry = next(
        event["geometry"] for event in admitted_manifest["events"] if event["event_id"] == "gt:5001:15"
    )
    expected_background = [
        index for index in range(image_plans["5001"]["cell_count"])
        if index not in raw_occupied
    ][:140]
    assert len(gt5001_geometry["image_cell_regions"]["background"]) == 140
    assert gt5001_geometry["image_cell_regions"]["background"] == expected_background


def test_live_executor_fails_if_guarded_runner_did_not_preconfigure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = live.SNaturalBoundaryKNHLiveExecutor()
    monkeypatch.setattr(
        executor,
        "_validate_inputs",
        lambda event, arm_order: (_event(), _row(), {}),
    )
    with pytest.raises(live.LiveExecutorError, match="not preconfigured"):
        executor.execute_event(_event())


def test_backend_identity_is_derived_from_loaded_adapter() -> None:
    hf_session_type = type("HFBackendSession", (), {})
    hf_session_type.__module__ = "src.inference.hf_backend"
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            backend=SimpleNamespace(
                type="hf", hf=SimpleNamespace(attn_implementation="sdpa")
            ),
            model=SimpleNamespace(dtype="fp32"),
            generation=SimpleNamespace(temperature=0.0, top_p=1.0),
        ),
        session=hf_session_type(),
        model=SimpleNamespace(training=False),
    )
    observed = live._observed_backend_identity(adapter)
    assert observed["generation"] == {
        "mode": "greedy", "temperature": 0.0, "top_p": 1.0
    }
    adapter.config.model.dtype = "fp16"
    with pytest.raises(live.LiveExecutorError, match="fp32"):
        live._observed_backend_identity(adapter)


@pytest.mark.parametrize(
    ("field", "wrong"),
    (
        ("source_panel_object_index", 99),
        ("derived_panel_object_index", 99),
        ("coco_ann_id", "wrong-ann"),
    ),
)
def test_covered_a_authoritative_panel_mapping_fails_closed(
    field: str,
    wrong: object,
) -> None:
    event = _event()
    event["owner_refs"] = {
        **event["owner_refs"],
        "covered_owner_ids": ["gt:5001:14"],
        "covered_A_owner_id": "gt:5001:14",
    }
    event["natural_boundary"] = {
        "prefix_token_ids": [7, 8],
        "prefix_sha256": cohort.sha256_json([7, 8]),
    }
    row = {
        "natural_boundary": 2,
        "covered_A_natural_boundary": 1,
        "covered_owner_ids": ["gt:5001:14"],
        "h0_record_index": 4,
    }
    a_source = {
        "source_panel_object_index": 14,
        "derived_panel_object_index": 13,
        "coco_ann_id": "ann-a",
    }
    a_source[field] = wrong
    legacy_event = {
        "panel_identity": {
            "source_panel_object_index": 15,
            "derived_panel_object_index": 15,
            "coco_ann_id": "ann-b",
        },
        "A_B": {"S": {"A_latest_covered": {"source_panel_object_index": 14}}},
        "geometry_by_checkpoint": {"S": {"owner_sources": {"gt:5001:14": a_source}}},
    }
    h0_records = {
        "gt:5001:15": {
            "exact_prefix_token_ids": (7, 8),
            "exact_prefix_sha256": cohort.sha256_json([7, 8]),
            "natural_boundary": 2,
            "covered_owner_ids": ("gt:5001:14",),
            "latest_covered_owner_id": "gt:5001:14",
            "record_index": 4,
        },
        "gt:5001:14": {
            "natural_boundary": 1,
            "exact_prefix_token_ids": (7,),
            "exact_prefix_sha256": cohort.sha256_json([7]),
            "strict_complete_row": True,
        },
    }
    mapping = {
        "gt:5001:15": {
            "source_index": 15, "derived_index": 15, "coco_ann_id": "ann-b"
        },
        "gt:5001:14": {
            "source_index": 14, "derived_index": 13, "coco_ann_id": "ann-a"
        },
    }
    with pytest.raises(live.LiveExecutorError, match="covered-A panel mapping"):
        live._validate_authoritative_event_binding(
            event,
            row,
            legacy_event,
            h0_records=h0_records,
            owner_mapping=mapping,
        )


def test_executor_loads_model_once_and_attests_each_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    event = _event()
    row = _row()
    event2 = _event()
    event2.update({"event_id": "gt:5002:16", "gt_owner_id": "gt:5002:16", "image_id": 5002, "owner_refs": {"gt_owner_id": "gt:5002:16", "source_panel_object_index": 16, "derived_panel_object_index": 16}})
    row2 = _row()
    row2.update({"gt_owner_id": "gt:5002:16", "image_id": 5002, "source_panel_object_index": 16, "derived_panel_object_index": 16})
    executor = live.SNaturalBoundaryKNHLiveExecutor()
    paths = {"config": Path("config"), "panel": Path("panel"), "cohort": Path("cohort"), "h0_root": Path("h0")}
    targets = iter(((event, row, paths), (event2, row2, paths), (event2, row2, paths)))
    monkeypatch.setattr(executor, "_validate_inputs", lambda value, arm_order: next(targets))
    monkeypatch.setattr(
        live.legacy,
        "_bind_materialization_event_h0_context",
        lambda *_args, **_kwargs: pytest.fail(
            "live executor must use the ledger-exact resolver, not synthetic materialization binding"
        ),
    )
    load_count = 0

    load_binding = _pre_gpu_identity()
    executor.configure_pre_gpu_identity(load_binding, runtime_versions=_runtime_versions())

    def legacy_event(value: dict[str, object]) -> dict[str, object]:
        owner_refs = value["owner_refs"]
        assert isinstance(owner_refs, dict)
        return {
            **value,
            "source_panel_object_index": owner_refs["source_panel_object_index"],
            "panel_identity": {"derived_panel_object_index": owner_refs["derived_panel_object_index"]},
            "geometry_by_checkpoint": {"S": {"checkpoint": "S"}},
            "checkpoint_status": {"S": {"disposition": "established"}},
        }

    def load_once(_paths: object, current_binding: object) -> None:
        nonlocal load_count
        if executor._adapter is not None:
            return
        assert current_binding == load_binding
        load_count += 1
        executor._adapter = SimpleNamespace(
            wrapper_contract=SimpleNamespace(object_ref_start_token_id=999)
        )
        executor._legacy_event = event
        executor._orchestrator = SimpleNamespace(events=[legacy_event(event), legacy_event(event2)])
        expected_observed = runpy.run_path(
            str(Path(__file__).with_name("test_run_s_natural_boundary_k_n_h_cohort.py"))
        )["_executor_identity"]()["observed"]
        executor._identity = {
            "config_sha256": load_binding["input_hashes"]["config_sha256"],
            "full_runtime_cohort_preflight": runpy.run_path(
                str(Path(__file__).with_name("test_run_s_natural_boundary_k_n_h_cohort.py"))
            )["_full_runtime_cohort_preflight"](),
            "model_identity": expected_observed["model"],
            "backend_identity": expected_observed["backend"],
            "model_device_attestation": {"passed": True, "logical_model_device": "cuda:0"},
            "cuda_visible_devices": {"raw": "0", "tokens": ["0"], "selected_physical_device": "0"},
        }

    monkeypatch.setattr(executor, "_load_once", load_once)
    seen_events: list[str] = []
    monkeypatch.setattr(
        live.legacy,
        "_make_event_context",
        lambda adapter, event, **_kwargs: (
            seen_events.append(str(event.get("gt_owner_id")))
            or SimpleNamespace(
                runtime=object(),
                prefix_ids=torch.tensor([[7, 8, 999]], dtype=torch.long),
            )
        ),
    )
    monkeypatch.setattr(
        live.gate,
        "build_natural_event_context",
        lambda binding, event_id: (
            SimpleNamespace(
                prefix_token_ids=(7, 8),
                prompt_token_ids=(),
                exact_history_token_ids=(7, 8),
            ),
            SimpleNamespace(natural_prefix_token_ids=(7, 8)),
            {"event_id": event_id},
        ),
    )
    monkeypatch.setattr(live.gate, "build_live_attention_mask_actuators", lambda binding, context, frozen_event=False: {})
    monkeypatch.setattr(live.gate, "build_live_residual_actuator", lambda: None)
    monkeypatch.setattr(live.gate, "_resolve_k14_reference_positions", lambda binding: {})
    monkeypatch.setattr(live.gate, "LiveRuntimeBinding", lambda **kwargs: SimpleNamespace(**kwargs, device="cuda:0", model=object()))
    arm_fixture = runpy.run_path(str(Path(__file__).with_name("test_run_s_natural_boundary_k_n_h_cohort.py"))) ["_arm_result"]

    class FakeGate:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.scalar = SimpleNamespace(k14_reference_positions={})

        def run_matrix(self, *, arms: object, **kwargs: object) -> dict[str, object]:
            return {
                "runtime_identity": {"event_id": "event-scoped-gate-identity"},
                "arms": {arm: arm_fixture(arm, 0) for arm in arms},
            }

    monkeypatch.setattr(live.gate, "SPrimaryNaturalBoundaryGate", FakeGate)
    first = executor.execute_event(event)
    for arm, result in first.items():
        assert result["executor_identity"]["full_runtime_cohort_preflight"] == executor._identity[
            "full_runtime_cohort_preflight"
        ]
        cohort.validate_arm_result(result, arm)
    executor._identity.pop("full_runtime_cohort_preflight")
    with pytest.raises(live.LiveExecutorError, match="full runtime cohort preflight"):
        executor.execute_event(event2)
    fixture = runpy.run_path(str(Path(__file__).with_name("test_run_s_natural_boundary_k_n_h_cohort.py")))
    invalid_preflight = deepcopy(fixture["_full_runtime_cohort_preflight"]())
    invalid_preflight["event_count"] = 10
    executor._identity["full_runtime_cohort_preflight"] = invalid_preflight
    with pytest.raises(live.LiveExecutorError, match="full runtime cohort preflight"):
        executor.execute_event(event2)
    assert tuple(first) == cohort.ARM_ORDER
    assert load_count == 1
    assert seen_events == ["gt:5001:15", "gt:5002:16", "gt:5002:16"]


def test_arm_order_mismatch_is_rejected_before_load(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = live.SNaturalBoundaryKNHLiveExecutor()
    with pytest.raises(live.LiveExecutorError, match="arm order"):
        executor._validate_inputs(_event(), cohort.ARM_ORDER[:-1])


def test_wrong_legacy_indices_are_not_an_exact_s_event() -> None:
    runtime_event = live._runtime_event(_event(), _row())
    candidate = {
        "gt_owner_id": "gt:5001:15",
        "image_id": 5001,
        "source_panel_object_index": 99,
        "panel_identity": {"derived_panel_object_index": 15},
        "geometry_by_checkpoint": {"S": {"checkpoint": "S"}},
        "checkpoint_status": {"S": {"disposition": "established"}},
    }
    assert live._is_exact_legacy_s_event(candidate, runtime_event) is False


def test_loaded_singleton_rejects_path_or_receipt_drift() -> None:
    executor = live.SNaturalBoundaryKNHLiveExecutor()
    executor._adapter = object()
    executor._load_binding = {"binding_sha256": "a" * 64}
    with pytest.raises(live.LiveExecutorError, match="loaded singleton identity"):
        executor._load_once({}, {"binding_sha256": "b" * 64})


def test_full_runtime_cohort_preflight_rejects_missing_context_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _event()
    event["natural_boundary"] = {
        "prefix_token_ids": [7, 8],
        "prefix_sha256": cohort.sha256_json([7, 8]),
    }
    row = _row()
    row.update(
        {
            "checkpoint": "S",
            "natural_boundary": 1,
            "h0_record_index": 4,
            "covered_owner_ids": ["gt:5001:2"],
        }
    )
    monkeypatch.setattr(live.cohort, "validate_manifest", lambda path: {"events": [event]})
    monkeypatch.setattr(live, "_json_file", lambda path, label: ({"rows": [row]}, b"{}"))
    with pytest.raises(live.LiveExecutorError, match="0 contexts"):
        live._preflight_full_runtime_cohort(
            SimpleNamespace(events=[]),
            {
                "manifest": Path("manifest"),
                "census": Path("census"),
                "cohort": Path("cohort.json"),
                "cohort_manifest": Path("cohort.manifest.json"),
            },
        )


def test_failed_full_cohort_preflight_never_calls_model_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = 0

    def loader(orchestrator: object, identity: object) -> object:
        nonlocal loads
        loads += 1
        return object()

    class FakeOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            self.events: list[object] = []

        def _load_cpu_contract(self) -> dict[str, object]:
            return {"config_sha256": "c" * 64}

    executor = live.SNaturalBoundaryKNHLiveExecutor(loader=loader)
    monkeypatch.setattr(live.legacy, "OwnerInterfaceOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(live.gate, "_require_preload_cuda_visibility", lambda: {"raw": "0"})
    monkeypatch.setattr(
        live,
        "_preflight_full_runtime_cohort",
        lambda orchestrator, paths: (_ for _ in ()).throw(live.LiveExecutorError("missing context")),
    )
    paths = {
        "config": Path("config"),
        "panel": Path("panel"),
        "cohort": Path("cohort"),
        "h0_root": Path("h0"),
        "h0_dir": Path("h0/selected"),
        "manifest": Path("manifest"),
        "census": Path("census"),
    }
    with pytest.raises(live.LiveExecutorError, match="missing context"):
        executor._load_once(paths, {"binding_sha256": "b" * 64})
    assert loads == 0
