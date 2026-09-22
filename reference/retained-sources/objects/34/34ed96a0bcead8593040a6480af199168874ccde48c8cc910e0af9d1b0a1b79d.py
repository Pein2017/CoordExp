"""CPU-only closeout of the fixed four-round course; reuses native reducers."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import torch
from scripts.research.analyze_ce_controls import _load_run, _pair
from scripts.research.analyze_sft256_dev128_baseline import _band, _sha256, _write_json
from scripts.research.prepare_source256_ce_rloo_round import (
    BASE_MODEL, _assign, _rloo_groups, inspect_dora_adapter_payload, validate_plan,
)
from scripts.research.train_source256_ce_rloo_round import _validate_optimizer_state_document
from src.config.inference import load_infer_config
from src.data import load_raw_examples

PORTFOLIO = Path(__file__).resolve().parent.parent / "2026-09-06-ce-controls-rloo-successor"
RUN = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/ce-rloo-v1")


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    return {"path": str(path), "sha256": _sha256(Path(path))}


def reduce(output):
    if output.exists():
        raise FileExistsError(output)
    subprocess.run(["sha256sum", "-c", str(PORTFOLIO / "source256-runtime-v1.sha256")], check=True, stdout=subprocess.DEVNULL, cwd=ROOT)
    log = (RUN / "launch-v1.log").read_text()
    assert log.count("ROUND_COMPLETED round=") == 4
    assert log.rstrip().endswith("SOURCE256_CE_RLOO_TERMINAL exit_code=0 utc=2026-09-06T16:54:40Z")
    baseline_plan = read(PORTFOLIO / "ce-evaluation-plan-v1.json")
    raw_train = list(load_raw_examples(Path(baseline_plan["splits"]["train256"]["input_jsonl"])))
    updates, banks, runs, curves = {}, {}, {}, {}
    source_ids = None
    for arm in ("ce", "rloo"):
        previous = None
        for round_index in range(1, 5):
            case = RUN / arm / f"round-{round_index}"
            plan = validate_plan(case / "plan.json")
            receipt = read(case / "update/receipt.json")
            assert receipt["mechanical_status"] == "MECHANICALLY_VALID"
            assert (receipt["arm"], receipt["round"]) == (arm, round_index)
            assert receipt["plan"] == {**reference(case / "plan.json"), "content_sha256": plan["content_sha256"]}
            assert receipt["objective"] == plan["objective"]
            assert receipt["source_embedding"] == plan["model"]["source_embedding"]
            assert receipt["composition"]["prior_adapter"] == plan["model"]["current_adapter"]
            saved = inspect_dora_adapter_payload(receipt["saved_adapter"]["root"], BASE_MODEL)
            assert saved == receipt["saved_adapter"]
            state_ref = receipt["optimizer_state"]
            assert reference(Path(state_ref["path"])) == state_ref
            state = torch.load(state_ref["path"], map_location="cpu", weights_only=True)
            _validate_optimizer_state_document(state, arm=arm, previous_round=round_index,
                layout=state["parameter_layout"], current_adapter_fingerprint=saved["fingerprint"],
                source_embedding_fingerprint=receipt["source_embedding"]["fingerprint"])
            assert len(state["parameter_layout"]) == receipt["trainable_surface"]["tensor_count"] == 588
            assert state["parameter_layout_sha256"] == receipt["trainable_surface"]["parameter_layout_sha256"]
            assert receipt["trainable_surface"]["scalar_count"] == 18006016
            assert receipt["optimizer"]["step_count"] == round_index
            assert receipt["optimizer"]["persistent_from_previous_round"] == (round_index > 1)
            if previous:
                assert receipt["trainable_surface"]["adapter_state_sha256_before"] == previous["trainable_surface"]["adapter_state_sha256_after"]
                assert plan["lineage"]["optimizer_state"] == previous["optimizer_state"]
            groups = plan["population"]["groups"]
            ids = [g["image_id"] for g in groups]
            source_ids = ids if source_ids is None else source_ids
            assert ids == source_ids and len(set(ids)) == 256
            per_rank = receipt["distributed"]["per_rank"]
            assert len(per_rank) == 8 and {p["rank"] for p in per_rank} == set(range(8))
            for rank in per_rank:
                selected = [g for g in groups if g["rank"] == rank["rank"]]
                assert rank["image_ids"] == [g["image_id"] for g in selected]
                assert rank["image_count"] == 32 and rank["synchronized_backward_count"] == 1
                assert rank["forward_count"] == rank["backward_count"] == (32 if arm == "ce" else 128)
                assert rank["action_token_count"] == sum(a["action_token_count"] for g in selected for a in g["actions"])
            for key in ("forward_count", "backward_count", "action_token_count"):
                assert receipt["runtime"][key] == sum(p[key] for p in per_rank)
            updates[f"{arm}:{round_index}"] = {"receipt": reference(case / "update/receipt.json"),
                "plan": receipt["plan"], "saved_adapter": saved, "optimizer_state": state_ref,
                "optimizer": receipt["optimizer"], "runtime": receipt["runtime"], "status": "lead-accepted-mechanics"}
            if arm == "rloo":
                artifacts = [(Path(s["path"]), read(s["path"])) for s in plan["sources"]["rollout_artifacts"]]
                assert len(artifacts) == 8
                rebuilt = _rloo_groups(raw_train, artifacts, round_index=round_index)
                _assign(rebuilt)
                assert rebuilt == groups  # Raw 256x4 cells, reward, token/EOS and media provenance.
                actions = [a for g in groups for a in g["actions"]]
                banks[str(round_index)] = {"artifacts": plan["sources"]["rollout_artifacts"],
                    "sampling": plan["sampling"], "action_count": len(actions),
                    "action_tokens": sum(a["action_token_count"] for a in actions),
                    "max_action_tokens": max(a["action_token_count"] for a in actions),
                    "max_joint_tokens": max(len(g["prompt_token_ids"]) + a["action_token_count"] for g in groups for a in g["actions"]),
                    "flat_groups_retained": sum(g["semantic_flat"] for g in groups),
                    "zero_advantage_actions_retained": sum(a["advantage"] == 0 for a in actions),
                    "stop_reasons": dict(Counter(a["stop_reason"] for a in actions)),
                    "mean_sampled_reward": sum(a["reward"] for a in actions) / len(actions),
                    "performance_per_rank": [a["performance"] for _, a in artifacts]}
            previous = receipt
            del state
    for split, split_spec in baseline_plan["splits"].items():
        source = _load_run(baseline_plan, baseline_plan["baseline"][split]["source"], split_spec)
        density = {band: [i for i, row in source["raw"].items() if _band(len(row["gt"])) == band] for band in split_spec["expected_density_rows"]}
        assert {k: len(v) for k, v in density.items()} == split_spec["expected_density_rows"]
        targets = {"source": source}
        for round_index in (range(1, 5) if split == "dev128" else (4,)):
            for arm in ("ce", "rloo"):
                config = ROOT / f"configs/coordexp_swift/infer/source256-{arm}-round{round_index}-{split}-v1.yaml"
                cfg = load_infer_config(config).config
                spec = {"config": str(config.relative_to(ROOT)), "config_sha256": _sha256(config),
                    "run_name": cfg.run.name, "artifact_dir": str(RUN / "cold" / cfg.run.name),
                    "adapter": cfg.adapter.path, "embedding_delta": cfg.embedding_delta.path}
                assert Path(cfg.adapter.path) == RUN / arm / f"round-{round_index}/update/adapter"
                targets[f"{arm}:{round_index}"] = _load_run(baseline_plan, spec, split_spec)
            for contrast, target, baseline in (("ce-source", targets[f"ce:{round_index}"], source),
                    ("rloo-source", targets[f"rloo:{round_index}"], source),
                    ("rloo-ce", targets[f"rloo:{round_index}"], targets[f"ce:{round_index}"])):
                curves[f"{split}:{round_index}:{contrast}"] = {
                    f"iou{int(t*100)}": _pair(output, f"{split}-round{round_index}-{contrast}", target, baseline, density, t)
                    for t in (0.5, 0.6, 0.8)}
        for key, evidence in targets.items():
            assert evidence["summary"]["score_failure_count"] == 0
            assert evidence["manifest"]["resolved_config_fingerprints"]["infer_config"] == evidence["config"]["resolved_fingerprint"]
            runs[f"{split}:{key}"] = {"run_dir": evidence["run_dir"], "config": evidence["config"],
                "artifact_hashes": evidence["artifact_hashes"], "summary": evidence["summary"]}
    result = {"schema": "source256_ce_rloo_closeout_v1", "terminal_status": "completed",
        "technical_status": "lead-accepted", "scientific_scope": "single-seed fixed-panel four-round comparison; not equal compute or architecture promotion",
        "updates": updates, "banks": banks, "runs": runs, "curves": curves,
        "provenance": {"reducer": reference(Path(__file__)), "baseline_plan": reference(PORTFOLIO / "ce-evaluation-plan-v1.json"),
            "runtime_hashes": reference(PORTFOLIO / "source256-runtime-v1.sha256"), "course_log": reference(RUN / "launch-v1.log")}}
    _write_json(output / "aggregate.json", result)
    print(json.dumps({"terminal_status": "completed", "aggregate": reference(output / "aggregate.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    reduce(parser.parse_args().out.resolve())
