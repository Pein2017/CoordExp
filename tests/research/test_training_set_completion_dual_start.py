"""CPU contracts for the dual-start trial packet and durable readback seam."""
from __future__ import annotations

import copy
import subprocess
import time
from pathlib import Path

import pytest

from probes.training_set_completion import dual_start as d
from probes.training_set_completion import dual_start_distributed as distributed


@pytest.fixture(scope="module")
def trial(tmp_path_factory):
    output = tmp_path_factory.mktemp("dual-start-session") / "dual-start-v2"
    d.prepare(preparation_path=d.PREPARATION, teacher_path=d.TEACHER, output=output, shared_sources=d.DEFAULT_SHARED_SOURCES)
    value = d.read(output / "trial.json")
    d.validate_trial(value)
    return output, value


def _rehash(value: dict) -> dict:
    value = copy.deepcopy(value)
    value["content_sha256"] = d.digest({key: item for key, item in value.items() if key != "content_sha256"})
    return value


def test_trial_binds_both_adapters_embedding_delta_and_shared_code(trial):
    _output, value = trial
    assert set(value["arms"]) == {"A", "B"}
    assert value["training"]["image_forwards_per_arm"] == 2816
    assert value["readback"]["request_count"] == 88
    assert {Path(item["path"]).name for item in value["implementation_bindings"]} >= {"training.py", "dual_start.py", "runner.py", "context.py", "raw_axis_validity_hinge.py", "models.py"}


def test_wrong_adapter_embedding_or_code_identity_is_rejected(trial):
    _output, value = trial
    wrong = copy.deepcopy(value)
    wrong["arms"]["A"]["source_adapter"] = copy.deepcopy(wrong["arms"]["B"]["source_adapter"])
    with pytest.raises(ValueError, match="adapter identity"):
        d.validate_trial(_rehash(wrong))

    wrong = copy.deepcopy(value)
    wrong["shared_embedding_delta"]["semantics"] = "replace"
    with pytest.raises(ValueError, match="embedding delta identity"):
        d.validate_trial(_rehash(wrong))

    wrong = copy.deepcopy(value)
    wrong["implementation_bindings"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="implementation source"):
        d.validate_trial(_rehash(wrong))


def test_fresh_optimizer_and_new_step_labels_are_frozen(trial):
    _output, value = trial
    wrong = copy.deepcopy(value)
    wrong["arms"]["A"]["optimizer_mode"] = "resume"
    with pytest.raises(ValueError, match="optimizer/step mapping"):
        d.validate_trial(_rehash(wrong))
    wrong = copy.deepcopy(value)
    wrong["training"]["new_step_labels"] = [64, 128, 256]
    with pytest.raises(ValueError, match="training contract"):
        d.validate_trial(_rehash(wrong))


def test_readback_jobs_are_exactly_eleven_per_arm_and_new_step_not_old_step(trial):
    _output, value = trial
    jobs = d.readback_jobs(value, arm="A", step=0)
    assert len(jobs) == 11
    assert {job["step"] for job in jobs} == {0}
    assert all("new-step-0" in job["request_id"] for job in jobs)
    with pytest.raises(ValueError, match="readback arm/step"):
        d.readback_jobs(value, arm="A", step=2444)


def test_durable_request_completeness_and_mixed_checkpoint_rejection(trial):
    _output, value = trial
    jobs = d.readback_jobs(value, arm="B", step=64)
    rows = [{"arm": "B", "step": 64, "checkpoint_step": 64, "image_id": job["image_id"], "request_id": job["request_id"], "route_id": job["route_id"], "empty_assistant_prefix": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "max_new_tokens": d.READBACK_TOKEN_CAP, "generated_token_ids": [17, d.EOS], "generated_token_ids_sha256": d.digest([17, d.EOS]), "decode_stop_reason": "im_end"} for job in jobs]
    d.validate_readback_rows(rows, jobs)
    with pytest.raises(ValueError, match="mixed checkpoint"):
        d.validate_readback_rows(rows[:-1] + [{**rows[-1], "checkpoint_step": 256}], jobs)
    with pytest.raises(ValueError, match="token budget"):
        too_long = [17] * (d.READBACK_TOKEN_CAP + 1)
        d.validate_readback_rows(rows[:-1] + [{**rows[-1], "generated_token_ids": too_long, "generated_token_ids_sha256": d.digest(too_long)}], jobs)


def test_teacher_contract_rejects_wrong_route_or_owner_denominator(trial):
    _output, value = trial
    teacher = d.read(value["teacher_bank"]["path"])
    preparation = d.read(value["preparation"]["path"])
    wrong = copy.deepcopy(teacher)
    wrong["routes"][0]["image_id"] = 999
    with pytest.raises(ValueError, match="retained image order"):
        d.validate_teacher_bank(wrong, image_ids=preparation["image_ids"])
    wrong = copy.deepcopy(teacher)
    wrong["routes"][0]["provenance"]["selected_owner_ids"] = wrong["routes"][0]["provenance"]["selected_owner_ids"][:-1]
    with pytest.raises(ValueError, match="owner/box cardinality"):
        d.validate_teacher_bank(wrong, image_ids=preparation["image_ids"])


def _row_for_job(job, *, trial, output, adapter):
    manifest_path = Path(trial["arms"][job["arm"]]["training_manifest"]["path"])
    manifest = d.read(manifest_path)
    route = next(route for route in manifest["routes"] if int(route["image_id"]) == job["image_id"])
    ids = [17, d.EOS]
    return {"arm": job["arm"], "step": job["step"], "checkpoint_step": job["step"], "route_id": job["route_id"], "image_id": job["image_id"], "request_id": job["request_id"], "empty_assistant_prefix": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "max_new_tokens": d.READBACK_TOKEN_CAP, "generated_token_ids": ids, "generated_token_ids_sha256": d.digest(ids), "decode_stop_reason": "im_end", "prompt_token_ids": route["prompt_token_ids"], "executed_media_sha256": route["image_identity"]["executed_media_sha256"], "observed_image_grid_thw": route["image_identity"]["observed_image_grid_thw"], "adapter": adapter, "loaded_model": {"model_identity": {"adapter": {"adapter_path": adapter["root"], "merged_adapters": []}}}, "training_manifest": d.binding(manifest_path), "trial": d.binding(Path(trial["launch"]["trial_path"]))}


def test_launch_command_contains_parseable_output_and_release_arguments(trial):
    output, value = trial
    launch = d.read(output / "launch-command.json")
    d.validate_launch_command(launch, trial_path=output / "trial.json", output=output, release_path=output / "release.json")
    assert "--output" in launch["controller_argv"]

    broken = dict(launch)
    broken["controller_argv"] = [arg for arg in launch["controller_argv"] if arg != "--output" and str(arg) != str(output)]
    with pytest.raises(ValueError, match="launch controller argv"):
        d.validate_launch_command(broken, trial_path=output / "trial.json", output=output, release_path=output / "release.json")


def test_training_command_uses_four_rank_backend_and_arm_gpu_groups():
    manifest = Path("/tmp/dual-start-manifest.json")
    output = Path("/tmp/dual-start-training")
    command = d.distributed_training_command(manifest_path=manifest, output=output)
    assert command == ["python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4", "-m", "probes.training_set_completion.dual_start_distributed", "--manifest", str(manifest), "--output", str(output)]
    assert d.TRAINING_GPU_GROUPS == {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}


def test_two_update_qualification_manifests_are_rebound_to_distributed_producer(tmp_path):
    paths = d.prepare_distributed_qualification_manifests(output=tmp_path / "qualification")
    assert set(paths) == {"A", "B"}
    for path in paths.values():
        manifest = d.read(path)
        assert Path(manifest["sources"]["producer"]["path"]).name == "dual_start_distributed.py"
        assert manifest["runtime"] | {"updates": 2, "checkpoint_steps": [2], "max_model_forwards": 22, "wall_seconds": 600} == manifest["runtime"]
        distributed.validate_distributed_contract(manifest)


def test_partial_readback_recovery_retains_rows_and_returns_only_missing_jobs(trial):
    output, value = trial
    jobs = d.readback_jobs(value, arm="A", step=0)
    manifest_path = Path(value["arms"]["A"]["training_manifest"]["path"])
    adapter = d.read(manifest_path)["source_adapter"]
    row = _row_for_job(jobs[0], trial=value, output=output, adapter=adapter)
    row_path = output / "readback" / "A" / "new-step-000" / "rows" / f"image-{jobs[0]['image_id']:012d}.json"
    d.publish(row_path, row)
    all_jobs, missing = d.pending_readback_jobs(trial_path=output / "trial.json", output=output, arm="A", step=0, adapter_identity=adapter)
    assert all_jobs == jobs and len(missing) == 10 and jobs[0] not in missing
    with pytest.raises(ValueError, match="same trial output root"):
        d.pending_readback_jobs(trial_path=output / "trial.json", output=output.parent / "retry-root", arm="A", step=0, adapter_identity=adapter)


def test_collection_boundary_exposes_only_validated_full_endpoint(trial):
    output, value = trial
    jobs = d.readback_jobs(value, arm="A", step=0)
    manifest_path = Path(value["arms"]["A"]["training_manifest"]["path"])
    adapter = d.read(manifest_path)["source_adapter"]
    row_root = output / "readback" / "A" / "new-step-000" / "rows"
    rows = []
    for job in jobs:
        path = row_root / f"image-{job['image_id']:012d}.json"
        if path.is_file():
            row = d.read(path)
        else:
            row = _row_for_job(job, trial=value, output=output, adapter=adapter)
            d.publish(path, row)
        rows.append(row)
    result = {"schema": f"{d.SCHEMA}.readback_result", "status": "completed_unscored", "trial": d.binding(output / "trial.json"), "request_count": 11, "rows": [{"arm": "A", "step": 0, "image_id": row["image_id"], "generated_token_ids_sha256": row["generated_token_ids_sha256"]} for row in rows]}
    result_path = output / "readback-result.json"
    d.publish(result_path, result)
    admitted, receipt = d.load_admitted_readback_rows(result_path, arm="A", step=0)
    assert len(admitted) == 11 and receipt["request_count"] == 11


def test_recovered_terminal_is_repeatable_without_generation(trial):
    output, value = trial
    jobs = d.readback_jobs(value, arm="A", step=0)
    manifest_path = Path(value["arms"]["A"]["training_manifest"]["path"])
    adapter = d.read(manifest_path)["source_adapter"]
    terminal_path = output / "readback" / "A" / "new-step-000" / "terminal.json"
    d.publish(terminal_path, {"schema": f"{d.SCHEMA}.readback_terminal", "status": "completed", "arm": "A", "step": 0, "request_count": len(jobs), "recovered_without_generation": True})
    _all_jobs, missing = d.pending_readback_jobs(trial_path=output / "trial.json", output=output, arm="A", step=0, adapter_identity=adapter)
    assert missing == []


def test_readback_recovery_rejects_concrete_wrong_adapter_and_bad_binding(trial):
    output, value = trial
    jobs = d.readback_jobs(value, arm="A", step=0)
    a_manifest = d.read(Path(value["arms"]["A"]["training_manifest"]["path"]))
    b_manifest = d.read(Path(value["arms"]["B"]["training_manifest"]["path"]))
    row = _row_for_job(jobs[0], trial=value, output=output, adapter=b_manifest["source_adapter"])
    with pytest.raises(ValueError, match="adapter fingerprint"):
        d.validate_readback_rows([row], jobs[:1], trial_path=output / "trial.json", routes={int(route["image_id"]): route for route in a_manifest["routes"]}, manifest_path=Path(value["arms"]["A"]["training_manifest"]["path"]), expected_adapter=a_manifest["source_adapter"])


def test_readback_deadline_is_absolute_from_each_spawn_and_kills_owned_process():
    started = time.monotonic()
    first = subprocess.Popen(["python", "-c", "import time; time.sleep(.05)"], start_new_session=True)
    second = subprocess.Popen(["python", "-c", "import time; time.sleep(.5)"], start_new_session=True)
    entries = [(first, open("/dev/null", "w"), ["first"], "first", 0, time.monotonic()), (second, open("/dev/null", "w"), ["second"], "second", 1, time.monotonic())]
    with pytest.raises(subprocess.TimeoutExpired):
        d.wait_owned_processes(entries, wall_seconds=.15)
    elapsed = time.monotonic() - started
    assert elapsed < .35
    d._owned_kill(second)


def test_deadline_accepts_child_that_finished_before_observation_even_if_deadline_passed():
    process = subprocess.Popen(["python", "-c", "pass"], start_new_session=True)
    process.wait(timeout=2)
    entries = [(process, open("/dev/null", "w"), ["already-done"], "already-done", 0, time.monotonic() - 1)]
    result = d.wait_owned_processes(entries, wall_seconds=.01)
    assert result[0]["exit_code"] == 0
