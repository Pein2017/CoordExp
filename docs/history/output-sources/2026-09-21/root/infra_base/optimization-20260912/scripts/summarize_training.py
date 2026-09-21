"""Task-local eight-rank smoke checks and compact evidence projection."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    assert not args.receipt.exists(), "Receipt target already exists"
    expected = [int(x) for x in args.steps.split(",")]
    run = json.loads((args.run / "run.json").read_text())
    rows = [json.loads(line) for line in (args.run / "logging.jsonl").open()]
    train = [r for r in rows if r["split"] == "train"]
    evaluation = [r for r in rows if r["split"] == "eval"]
    assert run["status"] == "completed" and run["terminal_error"] is None
    assert run["runtime"]["world_size"] == 8
    assert run["completed_steps"] == 4
    assert [r["step"] for r in train] == expected
    assert all(r["finite_status"] == "finite" for r in rows)
    assert all(r["optimizer_update_status"] == "applied" for r in train)
    assert all(r["count/packs"] == 48 for r in train)
    assert [r["step"] for r in evaluation] == [s for s in expected if s in (2, 4)]
    assert all(r["pack_count"] == 13 and r["example_count"] == 128 for r in evaluation)
    policies = run["policy_identities"]
    assert policies["eval_reduction"]["effective_mode"] == "disjoint_shard"
    assert policies["attention_proof"]["proof_policy"] == "every_forward"
    measurement = run["measurement"]
    publications = measurement["checkpoint_publication_events"]
    assert [r["step"] for r in publications] == [s for s in expected if s in (2, 4)]
    for event in publications:
        assert event["status"] == "completed" and event["exact_training_state_enabled"]
        checkpoint = args.run / event["checkpoint_path"]
        assert (checkpoint / "inference_payload_manifest.json").is_file()
        assert (checkpoint / "training_state/manifest.json").is_file()
    assert (args.run / "checkpoints/final.json").is_file()
    semantic_keys = {"split", "step", "finite_status", "optimizer_update_status", "acc_top1", "acc_top5", "accuracy_stats", "example_count", "pack_count"}
    summary = {
        "status": "passed", "run": str(args.run), "world_size": 8,
        "completed_steps": run["completed_steps"], "executed_step_ids": expected,
        "global_pack_presentations": sum(r["count/packs"] for r in train),
        "policies": {k: policies[k] for k in ("attention_proof", "eval_reduction", "resume")},
        "semantic_rows": [{k: v for k, v in r.items() if k in semantic_keys or k.startswith(("loss/", "count/", "finite/"))} for r in rows],
        "entry_to_terminal": measurement["entry_to_terminal"],
        "phases": {k: {field: v.get(field) for field in ("status", "duration_seconds")} for k, v in measurement["phases"].items()},
        "resource_high_water": measurement["resource_high_water"],
        "publications": [{k: e[k] for k in ("step", "status", "duration_seconds", "exact_training_state_enabled")} for e in publications],
    }
    with args.receipt.open("x") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": "passed", "receipt": str(args.receipt), "steps": expected, "entry_seconds": summary["entry_to_terminal"]["duration_seconds"]}))


if __name__ == "__main__":
    main()
