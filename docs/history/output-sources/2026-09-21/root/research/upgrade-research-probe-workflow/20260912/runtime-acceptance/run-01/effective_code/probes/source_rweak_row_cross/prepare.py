#!/usr/bin/env python3
"""CPU-only, fail-closed reconstruction of the frozen original row-cross panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from src.data import load_raw_examples
from src.eval.assignment import global_matches as _global_matches
from src.inference.parsing import parse_compact_object_box_closed
from .reduce import _gt_objects, _pred_objects, load_manifest

# Historical provenance only. These paths are mapped to preserved bytes, never imported.
PROVIDER = Path("/data/CoordExp/.worktrees/coco-gt-correction-portfolio")
DEFAULT_MANIFEST = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/data-v1/manifest.json"
)

ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1"
)
INPUT_HASH = "62aff40429cfc10f0a640a6e86d298b560ab79d25ce0a776393757a12d15accd"
REDUCTION_HASH = "418bfeea25d07399917b644bca3361477373bb0d38a20f9da8cfc94c40f08491"
CAP = 3084
LIMITS = {"any_loss": 16, "gain_only": 8, "owner_set_equal_token_changed": 8}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binding(path, expected=None):
    path = Path(path)
    value = digest(path)
    require(expected is None or value == expected, f"source hash mismatch: {path}")
    return {"path": str(path), "sha256": value, "bytes": path.stat().st_size}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def indexed(path):
    records = read_jsonl(path)
    result = {r["row_id"]: r for r in records}
    require(len(result) == len(records), f"duplicate row identity: {path}")
    return result


def rank(row_id):
    return hashlib.sha256(f"20260909:{row_id}".encode()).hexdigest()


def parse(text, row):
    return parse_compact_object_box_closed(
        text,
        row_id=row["row_id"],
        row_index=row["row_index"],
        image_width=row["image_width"],
        image_height=row["image_height"],
    )


def verify_trace(entries, row, tokenizer):
    require(
        all(e["row_id"] == row["row_id"] for e in entries),
        "trace row identity mismatch",
    )
    entries = sorted(entries, key=lambda e: e["generated_step_index"])
    require(
        [e["generated_step_index"] for e in entries] == list(range(len(entries))),
        "trace step sequence mismatch",
    )
    active = [e for e in entries if not e["is_pad"]]
    require(entries[: len(active)] == active, "non-pad token after padding")
    ids = [e["token_id"] for e in active]
    require(0 < len(ids) <= CAP, "generated token cap mismatch")
    require(
        tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        == row["raw_decode_text"],
        "token/text reconstruction mismatch",
    )
    require(
        all(
            tokenizer.decode(
                [e["token_id"]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            == e["token_text"]
            for e in active
        ),
        "individual token text mismatch",
    )
    stops = [i for i, e in enumerate(active) if e["is_stop"]]
    if row["decode_stop_reason"] == "im_end":
        require(
            stops == [len(ids) - 1] and ids[-1] == 151645 and 151645 not in ids[:-1],
            "EOS trace mismatch",
        )
    else:
        require(
            row["decode_stop_reason"] == "length"
            and len(ids) == CAP
            and not stops
            and 151645 not in ids,
            "unknown or inconsistent historical stop",
        )
    parsed = parse(row["raw_decode_text"], row)
    require(
        parsed.predictions == row["pred"]
        and parsed.dropped_predictions == row["dropped_predictions"],
        "original parser replay mismatch",
    )
    return ids


def next_action(ids, start, tokenizer, row):
    require(start < len(ids), "missing_next_action")
    if ids[start] == 151645:
        require(start == len(ids) - 1, "nonterminal_eos")
        end, kind = start + 1, "eos"
    else:
        require(ids[start] == 151646, "next_action_not_row_start")
        try:
            end = ids.index(151649, start) + 1
        except ValueError:
            raise ValueError("incomplete_next_row") from None
        kind = "row"
    tokens = ids[start:end]
    text = tokenizer.decode(
        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if kind == "row":
        parsed = parse(text, row)
        require(
            parsed.valid_prediction_count == 1
            and parsed.dropped_prediction_count == 0
            and parsed.predictions[0]["char_start"] == 0
            and parsed.predictions[0]["char_end"] == len(text),
            "illegal_next_row",
        )
    return {
        "kind": kind,
        "token_ids": tokens,
        "text": text,
        "start_token_index": start,
        "end_token_index": end,
    }


def divergence(left, right, tokenizer, row):
    require(left != right, "identical_token_trajectories")
    offset = 0
    while True:
        a = next_action(left, offset, tokenizer, row)
        b = next_action(right, offset, tokenizer, row)
        if a["token_ids"] != b["token_ids"]:
            require(left[:offset] == right[:offset], "common_prefix_identity_mismatch")
            return left[:offset], {"source": a, "rweak": b}
        require(a["kind"] != "eos", "identical_eos_before_difference")
        offset = a["end_token_index"]


def verify_population(rows):
    require(set(rows) == {"source", "rweak"}, "arm denominator mismatch")
    require(
        len(rows["source"]) == 512 and set(rows["source"]) == set(rows["rweak"]),
        "image denominator mismatch",
    )
    owners = []
    for rid, row in rows["source"].items():
        require(
            row["row_id"] == rid and row["example_id"] == rid,
            "raw row identity mismatch",
        )
        peer = rows["rweak"][rid]
        require(
            all(
                row[k] == peer[k]
                for k in (
                    "gt",
                    "row_id",
                    "example_id",
                    "row_index",
                    "image_path",
                    "image_width",
                    "image_height",
                )
            ),
            "paired input identity mismatch",
        )
        owners.extend(int(g["object_id"]) for g in row["gt"])
    require(len(owners) == len(set(owners)) == 3759, "GT denominator mismatch")
    matches = {
        arm: {f"iou_{t:.2f}": _matched_by_image(data, t) for t in (0.5, 0.6, 0.8)}
        for arm, data in rows.items()
    }
    require(
        sum(map(len, matches["source"]["iou_0.50"].values())) == 2225,
        "Source full512 match count mismatch",
    )
    require(
        sum(map(len, matches["rweak"]["iou_0.50"].values())) == 2310,
        "Rweak full512 match count mismatch",
    )
    return matches


def qualify(cases):
    """EOS first, then prefix min/max, then longest action; hash ties; fill hash order."""
    chosen = []
    eos = [c for c in cases if any(a["kind"] == "eos" for a in c["actions"].values())]
    candidates = []
    if eos:
        candidates.append(min(eos, key=lambda c: rank(c["row_id"])))
    candidates.extend(
        [
            min(
                cases,
                key=lambda c: (len(c["common_prefix_token_ids"]), rank(c["row_id"])),
            ),
            min(
                cases,
                key=lambda c: (-len(c["common_prefix_token_ids"]), rank(c["row_id"])),
            ),
            min(
                cases,
                key=lambda c: (
                    -max(len(a["token_ids"]) for a in c["actions"].values()),
                    rank(c["row_id"]),
                ),
            ),
        ]
    )
    candidates.extend(sorted(cases, key=lambda c: rank(c["row_id"])))
    for case in candidates:
        if case["row_id"] not in chosen:
            chosen.append(case["row_id"])
        if len(chosen) == min(4, len(cases)):
            break
    return chosen


def _validate_rows_against_input(rows, input_jsonl: Path):
    expected = {
        str(example.example_id): example for example in load_raw_examples(input_jsonl)
    }
    if set(rows) != set(expected):
        raise ValueError("raw output image IDs differ from frozen input")
    for row_id, example in expected.items():
        row = rows[row_id]
        try:
            observed = {
                "row_id": row.get("row_id"),
                "example_id": row.get("example_id"),
                "image_path": row.get("image_path"),
                "image_width": row.get("image_width"),
                "image_height": row.get("image_height"),
                "gt": [
                    {
                        "object_id": obj.get("object_id"),
                        "description": obj.get("description"),
                        "bbox": obj.get("bbox"),
                    }
                    for obj in row.get("gt", [])
                ],
            }
        except AttributeError as exc:
            raise ValueError(
                f"raw output identity is malformed for {row_id!r}"
            ) from exc
        wanted = {
            "row_id": row_id,
            "example_id": row_id,
            "image_path": str(example.image.path),
            "image_width": example.image.width,
            "image_height": example.image.height,
            "gt": [
                {
                    "object_id": obj.object_id,
                    "description": obj.description,
                    "bbox": list(obj.bbox),
                }
                for obj in example.objects
            ],
        }
        if observed != wanted:
            raise ValueError(
                f"raw output identity differs from frozen input for {row_id!r}"
            )


def _matched_by_image(rows, threshold):
    result = {}
    for row_id, row in rows.items():
        gt = _gt_objects(row, row_id=row_id)
        pred, _ = _pred_objects(row)
        owners = [int(obj["object_id"]) for obj in row["gt"]]
        require(len(owners) == len(set(owners)), "duplicate GT owner identity")
        result[row_id] = {owners[g] for g, _, _ in _global_matches(gt, pred, threshold)}
    return result


def prepare(*, root=ROOT, original_manifest=DEFAULT_MANIFEST, original_code_root=None):
    frozen = load_manifest(original_manifest, original_code_root=original_code_root)
    root = Path(root)
    from transformers import AutoTokenizer
    from PIL import Image
    import yaml

    input_path = root / "inputs-v1/holdout512.jsonl"
    sources = {
        "input": binding(input_path, INPUT_HASH),
        "reference_reduction": binding(
            root / "evaluation/owner-focus-reduction-v1.json", REDUCTION_HASH
        ),
    }
    rows, plans, manifests, configs, traces = {}, {}, {}, {}, {}
    files = [
        "gt_vs_pred.jsonl",
        "gt_vs_pred_scored.jsonl",
        "pred_token_trace.jsonl",
        "image_plan.jsonl",
        "run_manifest.json",
        "summary.json",
        "parse_diagnostics.jsonl",
        "configs/resolved.json",
    ]
    for arm, original in [("source", "Source"), ("rweak", "Rweak")]:
        run = root / f"evaluation/{original}/holdout/{original}-holdout512-native-v1"
        rows[arm] = indexed(run / "gt_vs_pred.jsonl")
        plans[arm] = indexed(run / "image_plan.jsonl")
        _validate_rows_against_input(rows[arm], input_path)
        manifest = manifests[arm] = json.loads((run / "run_manifest.json").read_text())
        resolved = json.loads((run / "configs/resolved.json").read_text())
        config = configs[arm] = resolved["config"]
        require(
            manifest["terminal_status"] == "completed", "original run not completed"
        )
        require(
            manifest["generation_policy"]["repetition_penalty"] == 1.0
            and manifest["generation_policy"]["max_new_tokens"] == CAP,
            "historical policy mismatch",
        )
        require(
            config["model"]["dtype"] == "fp32"
            and config["backend"]["hf"]
            == {"attn_implementation": "sdpa", "patch_embed_linearization": "enabled"},
            "historical precision or attention mismatch",
        )
        require(set(plans[arm]) == set(rows[arm]), "image-plan denominator mismatch")
        entry = Path(resolved["resolution"]["entry_config_path"])
        require(
            yaml.safe_load((run / "configs/resolved.yaml").read_text())["config"]
            == config,
            "persisted resolved YAML differs from JSON config",
        )
        historical_entry = dict(
            resolved["resolution"]["sources"][0], available=entry.is_file()
        )
        if entry.is_file():
            require(
                yaml.safe_load(entry.read_text()) == config,
                "original YAML differs from resolved config",
            )
            binding(entry, historical_entry["sha256"])
        sources[arm] = {
            "run_root": str(run),
            "artifacts": {name: binding(run / name) for name in files},
            "original_yaml": binding(run / "configs/resolved.yaml"),
            "historical_entry_yaml": historical_entry,
            "config": config,
            "model_identity": manifest["backend_session"]["model_identity"],
            "tokenizer_identity": manifest["tokenizer_identity"],
            "runtime_effective_settings": manifest["backend_session"][
                "effective_settings"
            ],
            "prompt_policy_fingerprint": manifest["prompt_policy_fingerprint"],
            "processor_identity_fingerprint": manifest[
                "processor_identity_fingerprint"
            ],
        }
        groups = defaultdict(list)
        with (run / "pred_token_trace.jsonl").open() as stream:
            for line in stream:
                e = json.loads(line)
                require(
                    e["trace_type"] in ("generated_token", "selected_token_replay"),
                    "unknown trace type",
                )
                if e["trace_type"] != "generated_token":
                    continue
                groups[e["row_id"]].append(
                    {
                        k: e[k]
                        for k in (
                            "row_id",
                            "generated_step_index",
                            "token_id",
                            "token_text",
                            "is_pad",
                            "is_stop",
                        )
                    }
                )
        require(set(groups) == set(rows[arm]), "trace denominator mismatch")
        traces[arm] = groups
    base = Path(configs["source"]["model"]["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    sources["base_files"] = [
        binding(p)
        for p in sorted(base.iterdir())
        if p.is_file() and p.suffix in (".json", ".safetensors", ".jinja", ".txt")
    ]
    sources["composition_files"] = []
    for directory in sorted(
        {configs[a]["adapter"]["path"] for a in configs}
        | {configs[a]["embedding_delta"]["path"] for a in configs}
    ):
        sources["composition_files"].extend(
            binding(p) for p in sorted(Path(directory).iterdir()) if p.is_file()
        )
    checkpoint_path = root / "Rweak/checkpoint-000064/manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    sources["rweak_checkpoint"] = {
        "manifest": binding(checkpoint_path),
        "identity": checkpoint,
    }
    sources["bank"] = binding(
        checkpoint["bank"]["manifest_path"], checkpoint["bank"]["manifest_sha256"]
    )
    bank = json.loads(Path(checkpoint["bank"]["manifest_path"]).read_text())
    require(
        checkpoint["source_identity"] == bank["source_identity"],
        "checkpoint Source identity mismatch",
    )
    source = bank["source_identity"]
    require(
        configs["source"]["adapter"]["path"] == source["adapter_root"]
        and str(base) == source["base_model_path"],
        "bank Source composition mismatch",
    )
    require(
        all(
            configs[a]["embedding_delta"]["path"] == source["embedding_root"]
            for a in configs
        ),
        "bank selected embedding identity mismatch",
    )
    binding(
        Path(source["adapter_root"]) / "adapter_model.safetensors",
        source["adapter_tensor_sha256"],
    )
    for name, sha in checkpoint["payload"]["files"].items():
        binding(Path(checkpoint["payload"]["path"]) / name, sha)
    sources["loader"] = {
        "provider_worktree": str(PROVIDER),
        "source_opener": "src.inference.hf_backend.open_hf_backend_session",
        "rweak_opener": "scripts.research.eval_coco_owner_focus.session_opener",
        "rweak_opener_arguments": {
            "checkpoint": str(checkpoint_path.parent),
            "arm": "Rweak",
            "bank_manifest": checkpoint["bank"]["manifest_path"],
            "expected_completed_update": 64,
        },
        "note": "Build launch from original config, preserve Source selected additive embeddings and patch linearization; use recipient forward passes, never donor cache.",
    }
    sources["code"] = frozen["sources"][
        "code"
    ]  # verified original bytes, unchanged historical locators
    matches = verify_population(rows)
    tokens = {a: {} for a in rows}
    input_records = read_jsonl(input_path)
    require(len(input_records) == 512, "raw input denominator mismatch")
    require(
        sorted(row["row_index"] for row in rows["source"].values()) == list(range(512)),
        "original row-index denominator mismatch",
    )
    for rid, row in rows["source"].items():
        original = input_records[row["row_index"]]
        require(
            rid == f"coco2017_val_{original['image_id']:012d}"
            and (original["width"], original["height"])
            == (row["image_width"], row["image_height"]),
            "input row-index identity mismatch",
        )
        plan = plans["source"][rid]
        require(plan == plans["rweak"][rid], "paired image-plan identity mismatch")
        require(
            plan["status"] == "ok"
            and plan["row_id"] == row["row_id"]
            and plan["example_id"] == row["example_id"]
            and plan["row_index"] == row["row_index"],
            "image-plan row identity mismatch",
        )
        require(
            plan["image_path"] == row["image_path"]
            and not plan["do_resize"]
            and plan["logical_transform_id"] == "identity",
            "image-plan transform mismatch",
        )
        require(
            digest(row["image_path"]) == plan["image_content_sha256"],
            "image bytes changed",
        )
        with Image.open(row["image_path"]) as image:
            require(
                image.size
                == (row["image_width"], row["image_height"])
                == (plan["decoded_width"], plan["decoded_height"])
                == (plan["declared_width"], plan["declared_height"]),
                "image dimension mismatch",
            )
        require(
            plan["expected_image_grid_thw"] == plan["observed_image_grid_thw"],
            "image grid mismatch",
        )
        for arm in rows:
            tokens[arm][rid] = verify_trace(traces[arm][rid], rows[arm][rid], tokenizer)
    del traces
    eligible = {s: [] for s in LIMITS}
    exclusions, strata = [], Counter()
    for rid in sorted(rows["source"]):
        s, r = (matches[a]["iou_0.50"][rid] for a in ("source", "rweak"))
        stratum = (
            "any_loss"
            if s - r
            else "gain_only"
            if r - s
            else "owner_set_equal_token_changed"
            if tokens["source"][rid] != tokens["rweak"][rid]
            else None
        )
        if stratum is None:
            exclusions.append(
                {
                    "row_id": rid,
                    "stratum": None,
                    "reason": "identical_owner_sets_and_tokens",
                }
            )
            continue
        strata[stratum] += 1
        try:
            prefix, actions = divergence(
                tokens["source"][rid],
                tokens["rweak"][rid],
                tokenizer,
                rows["source"][rid],
            )
        except ValueError as exc:
            exclusions.append({"row_id": rid, "stratum": stratum, "reason": str(exc)})
            continue
        row = rows["source"][rid]
        case = {
            "row_id": rid,
            "stratum": stratum,
            "selection_sha256": rank(rid),
            "row_index": row["row_index"],
            "input_record": input_records[row["row_index"]],
            "image_path": row["image_path"],
            "image_width": row["image_width"],
            "image_height": row["image_height"],
            "image_plan": plans["source"][rid],
            "gt": row["gt"],
            "common_prefix_token_ids": prefix,
            "common_prefix_text": tokenizer.decode(
                prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            "actions": actions,
            "diagonals": {},
            "remaining_token_budgets": {
                a: CAP - len(prefix) - len(action["token_ids"])
                for a, action in actions.items()
            },
            "continuation_token_budgets": {
                a: 0
                if action["kind"] == "eos"
                else CAP - len(prefix) - len(action["token_ids"])
                for a, action in actions.items()
            },
        }
        for arm in rows:
            case["diagonals"][arm] = {
                "generated_token_ids": tokens[arm][rid],
                "raw_record": rows[arm][rid],
                "matched_gt_ids": {
                    t: sorted(by_row[rid]) for t, by_row in matches[arm].items()
                },
                "stop_reason": rows[arm][rid]["decode_stop_reason"],
            }
        eligible[stratum].append(case)
    cases = []
    for stratum, limit in LIMITS.items():
        candidates = sorted(eligible[stratum], key=lambda c: c["selection_sha256"])
        cases.extend(candidates[:limit])
        exclusions.extend(
            {
                "row_id": c["row_id"],
                "stratum": stratum,
                "reason": "eligible_beyond_stratum_cap",
            }
            for c in candidates[limit:]
        )
    require(len(cases) + len(exclusions) == 512, "selection accounting mismatch")
    policy = {
        "max_new_tokens": CAP,
        "repetition_penalty": 1.0,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "stop_token_id": 151645,
        "stop_policy": "qwen_im_end",
        "dtype": "fp32",
        "attn_implementation": "sdpa",
        "patch_embed_linearization": "enabled",
        "budget_keys": "Donor action arm; remaining_token_budgets is numeric cap minus prefix/action, continuation_token_budgets is zero for terminal EOS.",
        "row_boundary": "Immediately after box_end, no separator; common prefix is maximal sequence of identical legal complete rows.",
        "primary_matcher": "Original category-consistent cardinality-first maximum-IoU global one-to-one matcher",
        "thresholds": [0.5, 0.6, 0.8],
    }
    selection = {
        "seed": 20260909,
        "ranking": "SHA256 UTF-8 20260909:<row_id>, ascending hex",
        "limits": LIMITS,
        "population_images": 512,
        "population_gt": 3759,
        "population_matched_counts": {
            a: {t: sum(map(len, m.values())) for t, m in v.items()}
            for a, v in matches.items()
        },
        "stratum_population_counts": dict(strata),
        "eligible_counts": {s: len(v) for s, v in eligible.items()},
        "selected_counts": dict(Counter(c["stratum"] for c in cases)),
        "exclusion_counts": dict(Counter(e["reason"] for e in exclusions)),
        "exclusions": exclusions,
        "qualification_rule": "EOS first by selection hash if present; shortest prefix; longest prefix; longest single action. Deduplicate, then fill ascending selection hash to at most four.",
    }
    return {
        "schema": "row_cross_manifest_v1",
        "sources": sources,
        "policy": policy,
        "selection": selection,
        "qualification_case_ids": qualify(cases),
        "cases": cases,
    }


def validate_manifest(manifest):
    require(manifest["schema"] == "row_cross_manifest_v1", "manifest schema mismatch")
    require(
        manifest["selection"]["population_images"] == 512
        and manifest["selection"]["population_gt"] == 3759,
        "manifest denominator mismatch",
    )
    ids = [c["row_id"] for c in manifest["cases"]]
    require(len(ids) == len(set(ids)) and len(ids) <= 32, "case denominator mismatch")
    require(
        len(ids) + len(manifest["selection"]["exclusions"]) == 512,
        "selected/excluded denominator mismatch",
    )
    for case in manifest["cases"]:
        p = case["common_prefix_token_ids"]
        for arm, action in case["actions"].items():
            diag = case["diagonals"][arm]
            require(
                diag["raw_record"]["row_id"] == case["row_id"],
                "manifest row identity mismatch",
            )
            require(
                diag["generated_token_ids"][: len(p) + len(action["token_ids"])]
                == p + action["token_ids"],
                "manifest prefix/action token mismatch",
            )
            require(
                case["remaining_token_budgets"][arm]
                == CAP - len(p) - len(action["token_ids"]),
                "manifest remaining budget mismatch",
            )
            require(
                case["continuation_token_budgets"][arm]
                == (
                    0
                    if action["kind"] == "eos"
                    else case["remaining_token_budgets"][arm]
                ),
                "manifest EOS budget mismatch",
            )
    require(
        manifest["qualification_case_ids"] == qualify(manifest["cases"]),
        "qualification selection mismatch",
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--source-root", type=Path, default=ROOT)
    ap.add_argument("--original-code-root", type=Path)
    ap.add_argument(
        "--output",
        type=Path,
        help="Write a separately reconstructed manifest; never overwrite the frozen input",
    )
    args = ap.parse_args(argv)
    manifest = prepare(
        root=args.source_root,
        original_manifest=args.manifest,
        original_code_root=args.original_code_root,
    )
    validate_manifest(manifest)
    content = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    require(
        args.manifest.read_text() == content,
        "frozen manifest differs from complete reconstruction",
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(content)
    print(
        json.dumps(
            {
                "status": "verified_reconstruction",
                "manifest": str(args.manifest),
                "sha256": digest(args.manifest),
                "case_count": len(manifest["cases"]),
                "execution": {
                    "entry": str(Path(__file__).resolve()),
                    "sha256": digest(__file__),
                },
                "qualification_case_ids": manifest["qualification_case_ids"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
