import copy
import json
import math

import pytest
import torch

from probes.owner_successor_scale import training as t


def test_first_fork_uses_literal_repeat_and_credible_tokens():
    spec = t.first_divergence([10, 20, 31, 40], [10, 20, 30, 99])
    assert spec == {
        "prefix_token_ids": [10, 20],
        "divergence_index": 2,
        "repeat_token_id": 31,
        "credible_token_id": 30,
    }
    with pytest.raises(ValueError, match="no first divergence"):
        t.first_divergence([1, 2], [1, 2])


def test_active_fork_gradient_pushes_repeat_down_and_credible_up_and_shifts_at_gamma():
    logits = torch.tensor([0.0, 1.25, 0.5], dtype=torch.float32, requires_grad=True)
    loss = t.fork_hinge(logits, repeat_token_id=1, credible_token_id=2)
    assert loss.item() == pytest.approx(1.75)
    loss.backward()
    assert logits.grad.tolist() == [0.0, 1.0, -1.0]

    boundary = torch.tensor([0.0, -0.5, 0.5], dtype=torch.float32)
    assert t.fork_hinge(boundary, repeat_token_id=1, credible_token_id=2).item() == 0.0
    shifted = boundary.clone()
    shifted[1] += 0.01
    assert t.fork_hinge(shifted, repeat_token_id=1, credible_token_id=2).item() == pytest.approx(0.01)


def test_fork_bank_keeps_fixed_denominator_and_nonrepeat_is_zero():
    anchor = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
    active = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, requires_grad=True)
    specs = [None] * 32
    logits = [None] * 32
    specs[7] = {"repeat_token_id": 1, "credible_token_id": 2}
    logits[7] = active
    loss, stats = t.normalized_fork_loss(logits, specs, package_count=32, zero=anchor)
    assert loss.item() == pytest.approx(2.0 / 32)
    assert stats["active_events"] == 1 and stats["denominator"] == 32
    loss.backward()
    assert anchor.grad.item() == 0.0
    assert active.grad.tolist() == [0.0, 1 / 32, -1 / 32]

    loss, stats = t.normalized_fork_loss([None] * 32, [None] * 32,
                                         package_count=32, zero=anchor)
    assert loss.item() == 0.0 and stats["active_events"] == 0


def test_calibration_is_one_closed_form_ratio_and_holds_zero_or_no_event():
    assert t.calibrate_fork_lambda(common_gradient_norm=8, raw_fork_gradient_norm=2,
                                   active_events=3) == 1.0
    with pytest.raises(ValueError, match="no strict-repeat"):
        t.calibrate_fork_lambda(common_gradient_norm=8, raw_fork_gradient_norm=2,
                                active_events=0)
    with pytest.raises(ValueError, match="fork objective has zero"):
        t.calibrate_fork_lambda(common_gradient_norm=8, raw_fork_gradient_norm=0,
                                active_events=1)


@pytest.mark.parametrize("new_count", [32, 33, 127, 128])
def test_balanced_schedule_has_exact_exposures_and_only_block_refreshes(new_count):
    old = [f"old-{i}" for i in range(16)]
    new = [f"new-{i}" for i in range(new_count)]
    schedule = t.balanced_schedule(old, new)
    t.validate_schedule(schedule, old, new)
    assert [row["update"] for row in schedule] == list(range(1, 257))
    assert [row["update"] for row in schedule if row["refresh_before_update"]] == list(range(1, 257, 8))
    assert {row["new_positive_per_record_scale"] for row in schedule} == {8 / new_count}


def _constant_scalars(count, value, *, grad):
    return [grad * 0 + value for _ in range(count)]


def test_common_banks_are_independently_normalized_with_frozen_new_scaling():
    parameter = torch.tensor(0.0, requires_grad=True)
    # Zero logits give per-token CE log(2).  Old has 2 records; N=33 gives
    # either 4 or 5 records/update, but every record retains scale 8/33.
    old_logits = [torch.zeros(1, 2, requires_grad=True) for _ in range(2)]
    new_logits = [torch.zeros(1, 2, requires_grad=True) for _ in range(5)]
    loss, components = t.independent_bank_objective(
        old_positive=old_logits,
        old_positive_targets=[[0], [1]],
        new_positive=new_logits,
        new_positive_targets=[[0]] * 5,
        old_witness_kl=_constant_scalars(16, 2.0, grad=parameter),
        new_witness_kl=_constant_scalars(33, 3.0, grad=parameter),
        normal_kl=_constant_scalars(54, 4.0, grad=parameter),
        normal_margin=_constant_scalars(54, 5.0, grad=parameter),
        new_package_count=33,
    )
    assert components == pytest.approx({
        "old_ce": math.log(2),
        "new_ce": 5 * 8 / 33 * math.log(2),
        "old_witness_kl": 2,
        "new_witness_kl": 3,
        "normal_kl": 4,
        "normal_margin": 5,
    })
    assert loss.item() == pytest.approx(
        components["old_ce"] + components["new_ce"]
        + 10 * 2 + 10 * 3 + 100 * 4 + 10 * 5,
        rel=1e-6,
    )


def test_reference_bank_uses_literal_positions_and_detaches_frozen_teacher():
    current = torch.zeros(3, 4, dtype=torch.float32, requires_grad=True)
    teacher_source = torch.tensor([[3.0, 0.0, 0.0, 0.0]], requires_grad=True)
    teacher = torch.log_softmax(teacher_source, -1)
    entry = {"record": {"record_id": "w", "target_token_ids": [0, 1, 2], "kl_positions": [1]}}
    kl, margin = t.reference_bank_items([current], [entry], {"w": teacher})
    assert margin == [] and len(kl) == 1
    kl[0].backward()
    assert teacher_source.grad is None
    assert current.grad[0].abs().sum().item() == 0.0
    assert current.grad[1].abs().sum().item() > 0.0
    assert current.grad[2].abs().sum().item() == 0.0


def test_runtime_backwards_each_bounded_replay_batch_before_building_next(monkeypatch):
    events = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.25))

    model = Model()

    def fake_replay(observed_model, entries):
        events.append(("forward", len(entries)))
        rows = []
        for _ in entries:
            logits = torch.stack((observed_model.value, observed_model.value * 0))
            logits.register_hook(lambda grad: events.append(("backward", 1)))
            rows.append(logits.unsqueeze(0))
        return rows

    monkeypatch.setattr(t, "batched_aligned_logits", fake_replay)
    items = [{"component": "old_ce", "scale": 0.5,
              "entry": {"prompt_ids": [1], "record": {"target_token_ids": [0]}}}
             for _ in range(5)]
    stats = t.backward_microbatches(t.BatchedObjective(model), items,
                                    microbatch_size=2, synchronize=False)
    assert [event for event in events if event[0] == "forward"] == [
        ("forward", 2), ("forward", 2), ("forward", 1)
    ]
    first_second_forward = events.index(("forward", 2), 1)
    assert any(event[0] == "backward" for event in events[1:first_second_forward])
    assert model.value.grad is not None and torch.isfinite(model.value.grad)
    assert stats["old_ce"] > 0 and all(stats[key] == 0 for key in stats if key != "old_ce")


def packet(status="calibration_pending"):
    old_ids = [f"old-{i}" for i in range(16)]
    packages = [{
        "package_id": f"new-{i}",
        "image_id": i // 2,
        "h_plus_token_ids": [1, 2],
        "credible_c_token_ids": [151646, 3, 151647, 151648, 4, 5, 6, 7, 151649],
        "c_trust": {"status": "physically_trusted", "supported_not_yet_covered": True,
                    "evidence": {"path": "/x", "sha256": "a" * 64}},
        "w_trust": {"status": "physically_trusted", "immediate_successor": True,
                    "evidence": {"path": "/y", "sha256": "b" * 64}},
        "c_record": {
            "record_id": f"p-{i}", "example_id": str(i // 2),
            "prompt_token_ids": [9], "prompt_token_ids_sha256": t.old_training.old.digest_ids([9]),
            "prefix_token_ids": [1, 2],
            "target_token_ids": [151646, 3, 151647, 151648, 4, 5, 6, 7, 151649],
            "image": {"row_id": str(i // 2), "row_index": i, "image_id": i // 2,
                      "image_path": "/image", "image_sha256": "c" * 64,
                      "observed_image_grid_thw": [1, 2, 3], "executed_media_sha256": "d" * 64},
        },
        "witness_record": {
            "record_id": f"w-{i}", "example_id": str(i // 2),
            "prompt_token_ids": [9], "prompt_token_ids_sha256": t.old_training.old.digest_ids([9]),
            "prefix_token_ids": [1, 2, 151646, 3, 151647, 151648, 4, 5, 6, 7, 151649],
            "target_token_ids": [151646, 8, 151647, 151648, 10, 11, 12, 13, 151649],
            "kl_positions": list(range(9)),
            "unknown_mask_policy": "literal_positions_only",
            "image": {"row_id": str(i // 2), "row_index": i, "image_id": i // 2,
                      "image_path": "/image", "image_sha256": "c" * 64,
                      "observed_image_grid_thw": [1, 2, 3], "executed_media_sha256": "d" * 64},
        },
        "case": {"row_id": str(i // 2)}, "golden": {"gt": []}, "h_text": "",
    } for i in range(32)]
    value = {
        "schema": t.SCHEMA,
        "status": status,
        "updates": 256,
        "block_size": 8,
        "coefficients": copy.deepcopy(t.COEFFICIENTS),
        "optimizer": copy.deepcopy(t.OPTIMIZER),
        "clip_gradient_norm": 1.0,
        "generation": copy.deepcopy(t.GENERATION),
        "reference_teacher": "frozen_N16_source_snapshot",
        "n16_receipt": {"path": "/n16/receipt.json", "sha256": "a" * 64},
        "n16_training_input": {"path": "/n16/inputs.json", "sha256": "b" * 64},
        "physical_package_bank": {"path": "/bank/packages.json", "sha256": "c" * 64},
        "replay_acceptance": {"path": "/replay/acceptance.json", "sha256": "d" * 64},
        "throughput": {"api": "batched_aligned_logits(model,entries)",
                       "microbatch_size": 2, "activation_checkpointing": False},
        "runtime": {"world_sizes": [2, 8], "max_rank_seconds": 100,
                    "max_cuda_allocated_bytes": 1000, "max_cuda_reserved_bytes": 1000,
                    "max_rss_bytes": 1000, "max_model_forwards_per_rank": 1000,
                    "max_image_forwards_per_rank": 1000},
        "stage_permissions": {
            "authorized": list(t.STAGED_REPLAY_STAGES),
            "full_256": "requires_root_integrated_smoke_acceptance",
        },
        "materialization_raw_sources": [{"path": "/raw/train.jsonl", "sha256": "e" * 64}],
        "raw_source_composition": {
            "schema": t.RAW_SOURCE_COMPOSITION_SCHEMA,
            "status": "passed_exact_projection",
            "policy": "retain_configured_base_owner_add_required_only",
            "base_sources": [{"path": "/raw/train.jsonl", "sha256": "e" * 64}],
            "augmentation_sources": [{"path": "/raw/full.jsonl", "sha256": "f" * 64}],
            "allowed_difference_paths": list(t.RAW_SOURCE_ALLOWED_DIFFERENCES),
            "counts": {
                "base_records": 1, "augmentation_selected_records": 1,
                "allowed_equivalent_collisions": 0, "required_additions": 1,
                "required_records": 1, "final_lookup_records": 2,
            },
            "required_example_ids_sha256": "1" * 64,
            "required_materialization_projections_sha256": "2" * 64,
            "collision_records_sha256": "3" * 64,
            "addition_records_sha256": "4" * 64,
            "base_rows_replaced": 0,
        },
        "fork": {
            "gamma": 1.0,
            "target_gradient_ratio": 0.25,
            "selection_gradient": "detached",
            "denominator": "fixed_new_package_count",
            "comparison": "actual_repeat_token_vs_fixed_credible_token",
        },
        "old_package_ids": old_ids,
        "new_packages": packages,
        "schedule": t.balanced_schedule(old_ids, [p["package_id"] for p in packages]),
        "calibration": None,
    }
    if status == "sealed_root_grant_pending":
        value["calibration"] = {
            "schema": t.CALIBRATION_SCHEMA,
            "status": "passed",
            "active_events": 2,
            "fork_lambda": 0.75,
        }
    return value


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return t._binding(path)


def _coord_row(image_id, image_ref, *, description="person"):
    return {
        "images": [image_ref],
        "objects": [{
            "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
            "desc": description,
            "category_id": 1,
            "category_name": "person",
            "coco_ann_id": image_id * 10,
        }],
        "width": 10,
        "height": 10,
        "image_id": image_id,
        "file_name": f"images/train2017/{image_id:012d}.jpg",
        "metadata": {"source": "coco2017", "split": "train"},
    }


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                            for row in rows))
    return t._binding(path)


def test_probe_local_raw_composition_reproduces_duplicate_then_keeps_base_owner(tmp_path):
    image = tmp_path / "images" / "train2017" / "000000000001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"same-image")
    new_image = tmp_path / "images" / "train2017" / "000000000002.jpg"
    new_image.write_bytes(b"new-image")
    base = tmp_path / "base.jsonl"
    augmentation = tmp_path / "augmentation.jsonl"
    _write_rows(base, [_coord_row(1, "images/train2017/000000000001.jpg")])
    # The source-local spelling differs while resolving to the same image.
    _write_rows(augmentation, [
        _coord_row(1, "./images/train2017/000000000001.jpg"),
        _coord_row(2, "images/train2017/000000000002.jpg"),
    ])
    base_binding, augmentation_binding = t._binding(base), t._binding(augmentation)

    with pytest.raises(ValueError, match="duplicate materialization raw ID"):
        t.old_training._load_materialization_raw(
            {"materialization_raw_sources": [base_binding, augmentation_binding]},
            config_input=base,
        )

    lookup, receipt = t.compose_materialization_raw_lookup(
        base_sources=[base_binding], augmentation_sources=[augmentation_binding],
        required_example_ids=["coco2017_train_000000000001", "coco2017_train_000000000002"],
        config_input=base,
    )
    assert lookup["coco2017_train_000000000001"].source.source_path == base.resolve()
    assert lookup["coco2017_train_000000000002"].source.source_path == augmentation.resolve()
    assert receipt["counts"] == {
        "base_records": 1, "augmentation_selected_records": 2,
        "allowed_equivalent_collisions": 1, "required_additions": 1,
        "required_records": 2, "final_lookup_records": 2,
    }
    assert receipt["allowed_difference_paths"] == list(t.RAW_SOURCE_ALLOWED_DIFFERENCES)


@pytest.mark.parametrize("conflict", ["object", "resolved_image"])
def test_probe_local_raw_composition_rejects_same_id_research_content_conflict(tmp_path, conflict):
    image = tmp_path / "images" / "train2017" / "000000000001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"same-image")
    other = tmp_path / "other" / "images" / "train2017" / "000000000001.jpg"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"other-image")
    base = tmp_path / "base.jsonl"
    augmentation = tmp_path / "augmentation.jsonl"
    _write_rows(base, [_coord_row(1, "images/train2017/000000000001.jpg")])
    row = _coord_row(1, "./images/train2017/000000000001.jpg")
    if conflict == "object":
        row["objects"][0]["desc"] = "dog"
    else:
        row["images"] = ["other/images/train2017/000000000001.jpg"]
    _write_rows(augmentation, [row])

    with pytest.raises(ValueError, match="collision differs outside allowed source provenance"):
        t.compose_materialization_raw_lookup(
            base_sources=[t._binding(base)], augmentation_sources=[t._binding(augmentation)],
            required_example_ids=["coco2017_train_000000000001"], config_input=base,
        )


def test_cold_positive_consumer_requires_new_only_composed_raw(monkeypatch):
    old_packet = {"positive_records": [{"record_id": "old", "example_id": "old-id"}]}
    value = {"new_packages": [{
        "c_record": {"record_id": "new", "example_id": "new-only-id"},
    }]}

    def fake_materialize(record, *, raw, **_):
        return {"record": record, "raw": raw[record["example_id"]]}

    monkeypatch.setattr(t.old_training, "_materialize", fake_materialize)
    with pytest.raises(KeyError, match="new-only-id"):
        t._cold_positive_entries(
            old_packet, value, qwen=None, frontend=None, config=None,
            raw={"old-id": "configured-base"},
        )
    entries = t._cold_positive_entries(
        old_packet, value, qwen=None, frontend=None, config=None,
        raw={"old-id": "configured-base", "new-only-id": "augmentation"},
    )
    assert [entry["raw"] for entry in entries] == ["configured-base", "augmentation"]


def physical_sources(tmp_path):
    image_ids = list(range(100, 116))
    raw = tmp_path / "train.jsonl"
    raw.write_text("{}\n")
    pool = {
        "schema": "owner_successor_scale.supply_pool.v1",
        "status": "frozen_before_new_rollouts",
        "image_ids": [*image_ids, *range(1000, 5080)],
        "excluded_ids": [],
        "source": t._binding(raw),
    }
    pool_path = tmp_path / "pool.json"
    _write_json(pool_path, pool)
    confirmation = {
        "schema": "native_owner_successor_scale_throughput.confirmation_selection.v1",
        "status": "frozen_cpu_no_model_calls",
        "image_ids": [9000],
    }
    confirmation_path = tmp_path / "confirmation.json"
    _write_json(confirmation_path, confirmation)
    jobs, rows, images, review_rows, groups = [], [], [], [], []
    c_ids = [151646, 3, 151647, 151648, 4, 5, 6, 7, 151649]
    w_ids = [151646, 8, 151647, 151648, 10, 11, 12, 13, 151649]
    for image_index, image_id in enumerate(reversed(image_ids)):
        image_file = tmp_path / f"{image_id}.jpg"
        image_file.write_bytes(f"image-{image_id}".encode())
        example_id = f"coco2017_train_{image_id:012d}"
        images.append({
            "image_id": image_id,
            "frozen": {
                "image_id": image_id, "example_id": example_id, "prompt_token_ids": [9],
                "golden": {"gt": []},
                "case": {
                    "row_id": example_id, "row_index": image_index,
                    "image_path": str(image_file), "image_height": 10, "image_width": 10,
                    "input_record": {},
                    "image_plan": {
                        "status": "ok", "image_path": str(image_file),
                        "image_content_sha256": t.old_training.file_hash(image_file),
                        "observed_image_grid_thw": [1, 2, 3],
                        "executed_media_sha256": "d" * 64,
                    },
                },
            },
        })
        for candidate_index in range(2):
            job_id = f"{image_id}:h0:c{candidate_index}"
            job = {
                "job_id": job_id, "image_id": image_id, "example_id": example_id,
                "history_index": 0, "candidate_index": candidate_index,
                "h_token_ids": [1, 2], "h_text": "", "c_token_ids": c_ids,
            }
            row = {
                "job_id": job_id, "image_id": image_id,
                "local_w": {"status": "candidate_local_w", "w_token_ids": w_ids,
                            "w_token_ids_sha256": t._digest_ids(w_ids)},
            }
            job_ordinal, row_ordinal = len(jobs), len(rows)
            jobs.append(job)
            rows.append(row)
            group_id = f"group-{image_id}-{candidate_index}"
            groups.append({"visual_group_id": group_id, "image_id": image_id,
                           "job_ids": [job_id]})
            review_rows.append({
                "job_id": job_id, "visual_group_id": group_id, "image_id": image_id,
                "example_id": example_id, "shard": "shard-0",
                "source_job_ordinal": job_ordinal, "source_row_ordinal": row_ordinal,
                "source_identity": {"history_index": 0, "candidate_index": candidate_index},
                "c": {"token_ids_sha256": t._digest_ids(c_ids),
                      "physical_review": {"status": "pending_view_image"}},
                "w": {"token_ids_sha256": t._digest_ids(w_ids),
                      "physical_review": {"status": "pending_view_image"}},
                "first_owner_axis": {"status": "separate_physical_axis_not_a_c_or_w_label"},
                "same_image_aliases": [], "execution_aliases_same_visual_group": [job_id],
            })
    jobs_path, rows_path, images_path = (tmp_path / name for name in (
        "jobs.jsonl", "rows.jsonl", "images.jsonl"
    ))
    for path, values in ((jobs_path, jobs), (rows_path, rows), (images_path, images)):
        path.write_text("\n".join(json.dumps(value, sort_keys=True, separators=(",", ":"))
                                  for value in values) + "\n")
    for index, review in enumerate(review_rows):
        review["source_job_sha256"] = t._line_sha256(jobs_path.read_text().splitlines()[index])
        review["source_row_sha256"] = t._line_sha256(rows_path.read_text().splitlines()[index])
    review_index = {
        "schema": "owner_successor_scale.physical_admission_review_index.v1",
        "status": "review_complete",
        "source_bindings": {"shard-0": {
            "jobs": t._binding(jobs_path), "rows": t._binding(rows_path),
            "images": t._binding(images_path),
        }},
        "groups": groups, "rows": list(reversed(review_rows)),
    }
    review_path = tmp_path / "review.json"
    _write_json(review_path, review_index)
    decisions = {
        "schema": t.PHYSICAL_DECISIONS_SCHEMA, "status": "root_decisions_complete",
        "pool": t._binding(pool_path), "confirmation_selection": t._binding(confirmation_path),
        "review_indexes": [t._binding(review_path)],
        "decisions": [{
            "visual_group_id": group["visual_group_id"], "canonical_job_id": group["job_ids"][0],
            "disposition": "admit",
            "c_trust": {"status": "physically_trusted", "supported_not_yet_covered": True},
            "w_trust": {"status": "physically_trusted", "immediate_successor": True},
            "alias_history": {"status": "canonical_exact_job_history",
                              "canonical_job_id": group["job_ids"][0]},
        } for group in groups],
    }
    decisions_path = tmp_path / "decisions.json"
    _write_json(decisions_path, decisions)
    return pool_path, confirmation_path, review_path, decisions_path


def test_physical_bank_materializes_exact_sources_in_pool_order(tmp_path):
    pool, confirmation, review, decisions = physical_sources(tmp_path)
    output = tmp_path / "bank.json"
    bank = t.materialize_physical_bank(
        pool_path=pool, confirmation_selection_path=confirmation,
        review_index_paths=[review], root_decisions_path=decisions, output=output,
    )
    assert bank["status"] == "root_admitted"
    assert bank["counts"] == {"packages": 32, "images": 16}
    assert [package["image_id"] for package in bank["packages"][:16]] == list(range(100, 116))
    assert [package["image_id"] for package in bank["packages"][16:]] == list(range(100, 116))
    assert all(package["c_trust"]["evidence"] == t._binding(decisions)
               and package["w_trust"]["evidence"] == t._binding(decisions)
               for package in bank["packages"])


def test_physical_source_preflight_uses_real_shapes_without_admission(tmp_path):
    pool, confirmation, review, _ = physical_sources(tmp_path)
    receipt = t.physical_source_preflight(
        pool_path=pool, confirmation_selection_path=confirmation,
        review_index_paths=[review], output=tmp_path / "source-preflight.json",
    )
    assert receipt["status"] == "CPU_valid_pending_root_physical_decisions"
    assert receipt["counts"] == {
        "candidate_rows": 32, "visual_groups": 32, "images": 16,
        "physical_axis_statuses": {"pending_view_image": 64},
    }
    assert receipt["admission_created"] is False


@pytest.mark.parametrize("field,value,match", [
    ("c_trust", {"status": "physically_trusted", "supported_not_yet_covered": False},
     "separate physical c"),
    ("w_trust", {"status": "physically_trusted", "immediate_successor": False},
     "separate physical immediate-w"),
    ("alias_history", {"status": "singleton", "canonical_job_id": "wrong"},
     "canonical exact alias/history"),
])
def test_physical_bank_fails_closed_on_trust_and_alias_rulings(tmp_path, field, value, match):
    pool, confirmation, review, decisions = physical_sources(tmp_path)
    payload = json.loads(decisions.read_text())
    payload["decisions"][0][field] = value
    _write_json(decisions, payload)
    with pytest.raises(ValueError, match=match):
        t.materialize_physical_bank(
            pool_path=pool, confirmation_selection_path=confirmation,
            review_index_paths=[review], root_decisions_path=decisions, output=tmp_path / "bank.json",
        )


def test_staged_replay_acceptance_has_measured_mb2_envelope(tmp_path):
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(json.dumps({
        "schema": "owner_successor_scale.final_warm_start_diagnostic.v1",
        "status": "passed", "logical_optimizer_updates": 2,
    }))
    capacity = tmp_path / "capacity.json"
    capacity.write_text(json.dumps({
        "schema": "owner_successor_scale.capacity_probe.v1", "status": "passed",
        "selected": {"status": "passed", "microbatch_size": 2,
                     "activation_checkpointing": False},
    }))
    acceptance = {
        "schema": t.REPLAY_ACCEPTANCE_SCHEMA, "status": "lead_accepted_staged",
        "authorized_stages": list(t.STAGED_REPLAY_STAGES),
        "api": "batched_aligned_logits(model,entries)",
        "microbatch_size": 2, "activation_checkpointing": False,
        "technical_evidence": {
            "final_warm_start_diagnostic": t._binding(diagnostic),
            "capacity_probe": t._binding(capacity),
        },
        "runtime": {
            "world_sizes": [2, 8], "max_rank_seconds": 86_400,
            "max_cuda_allocated_bytes": 69_793_218_560,
            "max_cuda_reserved_bytes": 85_899_345_920,
            "max_rss_bytes": 17_179_869_184,
            "max_model_forwards_per_rank": 1_600_000,
            "max_image_forwards_per_rank": 1_600_000,
        },
    }
    t.validate_replay_acceptance(acceptance, verify_files=True)
    acceptance["authorized_stages"].append("full_256")
    with pytest.raises(ValueError, match="stage scope"):
        t.validate_replay_acceptance(acceptance, verify_files=False)


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p["generation"].update(max_new_tokens_total=100), "full greedy"),
    (lambda p: p["fork"].update(comparison="all_vocab_max"), "repeat-fork"),
    (lambda p: p["new_packages"].pop(), "admission floor"),
    (lambda p: p["new_packages"][0]["c_trust"].update(status="machine_nominated"), "physical c trust"),
    (lambda p: p["new_packages"][0]["w_trust"].update(status="machine_nominated"), "physical immediate-w"),
    (lambda p: p.update(reference_teacher="Stable50"), "N16 teacher"),
    (lambda p: p["optimizer"].update(lr=2e-5), "AdamW"),
])
def test_packet_contract_fails_closed(mutate, match):
    value = packet()
    mutate(value)
    with pytest.raises(ValueError, match=match):
        t.validate_packet(value)


def test_root_grant_binds_sealed_packet_and_exact_update_scope(tmp_path):
    value = packet("sealed_root_grant_pending")
    packet_path = tmp_path / "inputs.json"
    packet_path.write_text(json.dumps(value))
    grant = {
        "schema": t.GRANT_SCHEMA,
        "status": "granted",
        "packet": t._binding(packet_path),
        "authorized_updates": 1,
        "authorized_arm": "A",
        "authorized_world_size": 2,
        "authorized_output_root": str(tmp_path / "run"),
        "root_admission_floor": {"packages_at_least": 32, "images_at_least": 16},
    }
    grant_path = tmp_path / "grant.json"
    grant_path.write_text(json.dumps(grant))
    assert t.verify_root_grant(packet_path, grant_path, updates=1, arm="A", world_size=2,
                               output_root=tmp_path / "run") == grant
    with pytest.raises(ValueError, match="update scope"):
        t.verify_root_grant(packet_path, grant_path, updates=256)
    grant["authorized_updates"] = 256
    grant["authorized_world_size"] = 8
    _write_json(grant_path, grant)
    with pytest.raises(ValueError, match="integrated smoke acceptance"):
        t.verify_root_grant(packet_path, grant_path, updates=256, world_size=8)


def test_readiness_is_explicit_hold_not_a_launch_claim():
    receipt = t.readiness_receipt()
    assert receipt["status"].startswith("HOLD_")
    assert "optimizer_update" in receipt["not_executed"]


def test_planned_work_declares_refresh_and_optimizer_bounds_before_launch():
    value = packet("sealed_root_grant_pending")
    one = t.planned_work_upper_bound(value, rank=0, world=2, updates=1)
    full = t.planned_work_upper_bound(value, rank=0, world=8, updates=256)
    assert one["refresh_rounds"] == 1 and one["optimizer_steps"] == 1
    assert full["refresh_rounds"] == 32 and full["optimizer_steps"] == 256
    assert full["model_forwards_upper_bound"] > full["training_batched_forwards"]
