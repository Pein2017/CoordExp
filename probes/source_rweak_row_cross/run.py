"""Exact-ID Source/Rweak row crossing using caller-owned Qwen native execution.

Without --execute this validates the frozen manifest, preserved source bindings,
selected cells and resource bounds, and writes a plan without loading a model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback

from src.eval.assignment import global_matches as _global_matches
from src.eval.native_rows import native_detection_record as native_record
from src.inference.bound_requests import build_bound_native_requests as build_requests
from .reduce import (
    CAP,
    EOS,
    MANIFEST_SHA256,
    _gt_objects,
    _pred_objects,
    original_code_bindings,
    require,
    sha256 as sha,
)
from .prepare import DEFAULT_MANIFEST, validate_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def token_ids(value):
    require(
        isinstance(value, list) and all(type(x) is int and x >= 0 for x in value),
        "invalid literal token IDs",
    )
    return value


def intervention(case, recipient, mode):
    donor = (
        recipient
        if mode == "qualify"
        else ("rweak" if recipient == "source" else "source")
    )
    prefix = token_ids(case["common_prefix_token_ids"])
    action = case["actions"][donor]
    ids = token_ids(action["token_ids"])
    require(action["kind"] in ("row", "eos") and len(ids) > 0, "illegal/empty action")
    require(
        len(prefix) + len(ids) <= CAP, "forced history exceeds whole trajectory budget"
    )
    require(
        case["remaining_token_budgets"][donor] == CAP - len(prefix) - len(ids),
        "manifest remaining budget differs",
    )
    require(
        case["continuation_token_budgets"][donor]
        == (0 if action["kind"] == "eos" else CAP - len(prefix) - len(ids)),
        "manifest terminal budget differs",
    )
    return {
        "case": case,
        "donor": donor,
        "prefix": prefix,
        "action": ids,
        "kind": action["kind"],
        "remaining": CAP - len(prefix) - len(ids),
    }


def validate_record(row, manifest_sha):
    require(row["manifest_sha256"] == manifest_sha, "manifest identity mismatch")
    ids = token_ids(row["generated_token_ids"])
    require(
        ids
        == row["prefix_token_ids"] + row["action_token_ids"] + row["suffix_token_ids"],
        "token history identity mismatch",
    )
    require(len(ids) == row["generated_token_count"] <= CAP, "token count/cap mismatch")
    require(
        row["remaining_budget"]
        == CAP - len(row["prefix_token_ids"]) - len(row["action_token_ids"]),
        "remaining budget mismatch",
    )
    require(
        len(row["suffix_token_ids"]) <= row["remaining_budget"],
        "per-request budget exceeded",
    )
    if row["decode_stop_reason"] == "forced_eos":
        require(not row["suffix_token_ids"], "forced EOS was reopened")


def selected_cases(manifest, args):
    require(manifest["schema"] == "row_cross_manifest_v1", "unknown manifest schema")
    all_cases = {str(c["row_id"]): c for c in manifest["cases"]}
    require(len(all_cases) == len(manifest["cases"]), "duplicate manifest cases")
    ids = (
        args.case_ids.split(",")
        if args.case_ids
        else (
            manifest["qualification_case_ids"]
            if args.mode == "qualify"
            else list(all_cases)
        )
    )
    require(
        len(set(ids)) == len(ids) and len(ids) > 0, "empty/duplicate case selection"
    )
    require(set(ids) <= all_cases.keys(), "unknown case IDs")
    if args.mode == "qualify":
        require(
            set(ids) <= set(manifest["qualification_case_ids"]),
            "qualification outside common fixtures",
        )
    return [all_cases[x] for x in ids]


def consume_rows(path, manifest_sha, tokenizer=None):
    """Fresh disk reload through the original parser and original global matcher."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    seen = set()
    for row in rows:
        validate_record(row, manifest_sha)
        if tokenizer is not None:
            require(
                tokenizer.decode(row["generated_token_ids"], skip_special_tokens=False)
                == row["raw_decode_text"],
                "cold token/text identity mismatch",
            )
        require(row["case_id"] not in seen, "duplicate output case")
        seen.add(row["case_id"])
        raw = row["parsed"]
        reparsed = native_record(
            row["raw_decode_text"],
            {"row_id": row["case_id"]},
            raw,
            row["decode_stop_reason"],
        )
        require(reparsed == raw, "cold parser reload mismatch")
        gt, (pred, invalid) = (
            _gt_objects(raw, row_id=row["case_id"]),
            _pred_objects(raw),
        )
        require(invalid == 0, "native parser produced invalid prediction geometry")
        row["consumer_matches"] = {
            str(t): [
                {
                    "gt_index": g,
                    "owner_id": str(raw["gt"][g]["object_id"]),
                    "pred_index": p,
                    "iou": i,
                }
                for g, p, i in _global_matches(gt, pred, t)
            ]
            for t in (0.5, 0.6, 0.8)
        }
    return rows


def verify_bindings(manifest, original_code_root=None):
    checked = {}

    def visit(value):
        if isinstance(value, dict):
            if all(k in value for k in ("path", "sha256", "bytes")) and isinstance(
                value["path"], str
            ):
                path = Path(value["path"])
                require(
                    "confirmation512" not in str(path),
                    "protected confirmation binding forbidden",
                )
                require(path.is_file(), "missing bound source: " + str(path))
                if str(path) not in checked:
                    checked[str(path)] = sha(path)
                require(
                    checked[str(path)] == value["sha256"],
                    "source digest mismatch: " + str(path),
                )
                require(
                    path.stat().st_size == value["bytes"],
                    "source size mismatch: " + str(path),
                )
            else:
                for key, item in value.items():
                    if key != "historical_entry_yaml":
                        visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for key, value in manifest["sources"].items():
        if key != "code":
            visit(value)
    original = original_code_bindings(manifest, original_code_root)
    require(
        manifest["policy"]["max_new_tokens"] == CAP
        and manifest["policy"]["repetition_penalty"] == 1.0
        and manifest["policy"]["stop_token_id"] == EOS,
        "manifest policy mismatch",
    )
    return {"live_files": checked, "original_code": original}


def load_rweak_checkpoint(manifest):
    """Validate only the frozen Rweak DoRA successor, without importing its trainer."""
    binding = manifest["sources"]["rweak_checkpoint"]["manifest"]
    path = Path(binding["path"])
    require(sha(path) == binding["sha256"], "checkpoint manifest digest differs")
    checkpoint = read(path)
    require(
        checkpoint == manifest["sources"]["rweak_checkpoint"]["identity"],
        "checkpoint identity differs from frozen panel",
    )
    require(
        checkpoint.get("schema") == "coco-gt-correction-checkpoint.v1"
        and checkpoint.get("mechanical_status") == "COMPLETED_UPDATE",
        "checkpoint is not a completed update",
    )
    require(
        checkpoint["arm"] == "Rweak"
        and checkpoint["surface"] == "dora"
        and checkpoint["completed_update"] == 64,
        "checkpoint arm/surface/update differs",
    )
    bank_path = Path(checkpoint["bank"]["manifest_path"]).resolve(strict=True)
    require(
        bank_path == Path(manifest["sources"]["bank"]["path"]).resolve(strict=True),
        "checkpoint names another bank",
    )
    require(
        sha(bank_path)
        == checkpoint["bank"]["manifest_sha256"]
        == manifest["sources"]["bank"]["sha256"],
        "bank digest differs",
    )
    bank = read(bank_path)
    require(
        checkpoint["bank"]["bank_id"] == bank["bank_id"]
        and checkpoint["source_identity"] == bank["source_identity"],
        "checkpoint Source/bank identity differs",
    )
    fields = {
        key: checkpoint[key]
        for key in (
            "arm",
            "surface",
            "completed_update",
            "bank",
            "source_identity",
            "trainable_surface",
            "recipe",
            "payload",
        )
    }
    digest = hashlib.sha256(
        json.dumps(
            fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    require(
        checkpoint["checkpoint_id"] == digest, "checkpoint manifest identity changed"
    )
    recipe = checkpoint["recipe"]
    require(
        all(
            recipe.get(k) == v
            for k, v in {
                "experiment_id": "2026-09-07-coco-owner-focus-ablation",
                "objective_variant": "Rweak",
                "seed": 20260908,
                "global_batch_size": 32,
            }.items()
        ),
        "checkpoint owner-focus recipe differs",
    )
    payload = checkpoint["payload"]
    require(payload["kind"] == "dora_adapter", "checkpoint is not a DoRA payload")
    require(
        Path(payload["path"]).resolve()
        == Path(manifest["sources"]["rweak"]["config"]["adapter"]["path"]).resolve(),
        "checkpoint payload is not configured adapter",
    )
    for name, digest in payload["files"].items():
        require(sha(Path(payload["path"]) / name) == digest, "adapter payload changed")
    return checkpoint


def prepare_sources(manifest, recipient):
    """Read the actual frozen run config; the new native execution is separate."""
    import yaml

    source = manifest["sources"][recipient]
    config = copy.deepcopy(source["config"])
    require(
        yaml.safe_load(Path(source["original_yaml"]["path"]).read_text())["config"]
        == config,
        "saved runtime config differs from frozen manifest",
    )
    require(config["model"]["dtype"] == "fp32", "original dtype changed")
    require(
        config["backend"]["hf"]
        == {"attn_implementation": "sdpa", "patch_embed_linearization": "enabled"},
        "original attention changed",
    )
    require(
        config["generation"]
        == {
            "batch_size": 4,
            "max_new_tokens": CAP,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "repetition_penalty": 1.0,
        },
        "original generation changed",
    )
    require(
        config["model"]["processor"]["do_resize"] is False, "original resize changed"
    )
    if recipient == "rweak":
        load_rweak_checkpoint(manifest)
    return config


def load_components(config, source, device):
    """Load once, attach the requested payloads and report observed native state."""
    import torch
    from src.qwen.runtime_loading import (
        QwenLoadOptions,
        load_qwen_components_from_options,
    )
    from src.adapters.dora import attach_dora_adapter
    from src.qwen.special_token_embeddings import attach_embedding_delta

    qwen = load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=config["model"]["base_model"],
            dtype=config["model"]["dtype"],
            attn_implementation=config["backend"]["hf"]["attn_implementation"],
            patch_embed_linearization=config["backend"]["hf"][
                "patch_embed_linearization"
            ],
            load_model=True,
        )
    )
    adapter = config["adapter"]
    adapter_receipt = attach_dora_adapter(
        qwen.model,
        adapter_path=adapter["path"],
        adapter_name=adapter["name"],
        base_model_path=qwen.base_model_path,
    )
    delta = config["embedding_delta"]
    delta_receipt = attach_embedding_delta(
        delta_path=delta["path"],
        qwen=qwen,
        source_gate_root=delta.get("source_gate_root"),
    )
    qwen.model.to(device).eval()
    actual = {
        "base": {"path": str(qwen.base_model_path)},
        "adapter": adapter_receipt,
        "embedding_delta": delta_receipt,
    }
    for key, value in actual.items():
        require(
            value == source["model_identity"][key],
            "loaded " + key + " identity differs from frozen run",
        )
    tokenizer_identity = qwen.token_identity.to_artifact_dict()
    require(
        tokenizer_identity == source["tokenizer_identity"],
        "loaded tokenizer identity differs",
    )
    counts = Counter()
    for parameter in qwen.model.parameters():
        counts[str(parameter.dtype)] += parameter.numel()
    dtype = {
        "parameter_dtype_counts": dict(sorted(counts.items())),
        "parameter_dtype_names": sorted(counts),
    }
    require(
        dtype == source["runtime_effective_settings"]["observed_model_dtype"],
        "loaded parameter dtype/count differs",
    )
    attention = getattr(qwen.model.config, "_attn_implementation", None)
    require(attention == "sdpa", "loaded attention differs")
    return qwen, {
        "kind": "qwen_native",
        "model_class": type(qwen.model).__name__,
        "model_identity": actual,
        "tokenizer_identity": tokenizer_identity,
        "processor_class": type(qwen.processor).__name__,
        "observed_model_dtype": dtype,
        "observed_attn_implementation": attention,
        "device": str(torch.device(device)),
        "device_name": torch.cuda.get_device_name(device)
        if torch.device(device).type == "cuda"
        else "cpu",
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
        if torch.device(device).type == "cuda"
        else None,
        "package_versions": dict(qwen.package_versions),
        "runtime_patches": qwen.runtime_patches,
    }


def execute(args, manifest, cases, receipt):
    import torch
    from src.qwen.native import prepare_native_inputs
    from src.qwen.generation import generate_continuations

    device = torch.device(args.device)
    if device.type == "cuda":
        require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "CUDA execution requires exactly one granted visible device",
        )
    config = prepare_sources(manifest, args.recipient)
    write(
        args.output_dir / "runtime-config.json",
        {
            "original": config,
            "native_batch_size": args.batch_size,
            "device": args.device,
        },
    )
    started = time.monotonic()
    qwen, observed = load_components(
        config, manifest["sources"][args.recipient], device
    )
    receipt["native_execution"] = observed
    receipt["initialization_seconds"] = time.monotonic() - started
    requests, receipt["prompt_meta"] = build_requests(qwen, config, cases)
    receipt["effective_generation"] = {
        "do_sample": False,
        "repetition_penalty": 1.0,
        "trace": "none",
        "output_scores": False,
        "output_logits": False,
        "output_hidden_states": False,
        "output_attentions": False,
        "cap": CAP,
    }
    model = qwen.model
    eos = int(qwen.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    pad = qwen.tokenizer.pad_token_id
    require(
        eos == EOS and type(pad) is int and pad >= 0, "runtime EOS/pad identity differs"
    )
    jobs = [
        intervention(c, args.recipient, args.mode) | {"request": r}
        for c, r in zip(cases, requests, strict=True)
    ]
    for job in jobs:
        require(eos not in job["prefix"], "EOS in common prefix")
        require(
            job["action"] == [eos]
            if job["kind"] == "eos"
            else eos not in job["action"],
            "row/EOS action mismatch",
        )
    counters = {"model_forwards": 0, "image_forwards": 0, "image_instances": 0}
    visuals = [m for name, m in model.named_modules() if name.endswith("visual")]
    require(len(visuals) == 1, "ambiguous visual module for observed counters")

    def count_model(*_):
        counters["model_forwards"] += 1

    def count_image(*_):
        counters["image_forwards"] += 1

    model_handle = model.register_forward_pre_hook(count_model)
    image_handle = visuals[0].register_forward_pre_hook(count_image)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    path = args.output_dir / "rows.jsonl"
    receipt["batches"] = []
    try:
        with path.open("x") as stream:

            def save(job, suffix, stop):
                case = job["case"]
                ids = job["prefix"] + job["action"] + list(suffix)
                text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
                row = {
                    "case_id": case["row_id"],
                    "recipient": args.recipient,
                    "action_source": job["donor"],
                    "cell": ("0" if args.recipient == "source" else "1")
                    + ("0" if job["donor"] == "source" else "1"),
                    "mode": args.mode,
                    "manifest_sha256": receipt["manifest_sha256"],
                    "prefix_token_ids": job["prefix"],
                    "action_token_ids": job["action"],
                    "suffix_token_ids": list(suffix),
                    "generated_token_ids": ids,
                    "raw_decode_text": text,
                    "decode_stop_reason": stop,
                    "generated_token_count": len(ids),
                    "remaining_budget": job["remaining"],
                    "parsed": native_record(
                        text,
                        case,
                        case["diagonals"][args.recipient]["raw_record"],
                        stop,
                    ),
                }
                validate_record(row, receipt["manifest_sha256"])
                stream.write(json.dumps(row) + "\n")
                stream.flush()

            active = []
            for job in jobs:
                if job["kind"] == "eos":
                    save(job, [], "forced_eos")
                elif job["remaining"] == 0:
                    save(job, [], "length")
                else:
                    active.append(job)
            active.sort(
                key=lambda j: len(j["request"].expected_token_ids)
                + len(j["prefix"])
                + len(j["action"])
            )
            for offset in range(0, len(active), args.batch_size):
                group = active[offset : offset + args.batch_size]
                batch = prepare_native_inputs(
                    qwen.processor,
                    [j["request"] for j in group],
                    device=device,
                    record_media_identity=True,
                )
                for job, grid, media in zip(
                    group, batch.image_grids, batch.media_sha256, strict=True
                ):
                    plan = job["case"]["image_plan"]
                    require(
                        list(grid) == plan["observed_image_grid_thw"]
                        and media == plan["executed_media_sha256"],
                        "historical media/grid identity differs",
                    )
                tick = time.monotonic()
                before = dict(counters)
                results = generate_continuations(
                    model,
                    batch,
                    extensions=[j["prefix"] + j["action"] for j in group],
                    budgets=[j["remaining"] for j in group],
                    eos_token_id=eos,
                    pad_token_id=pad,
                    trace="none",
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                counters["image_instances"] += len(group)
                for job, result in zip(group, results, strict=True):
                    require(
                        result.request_id == job["case"]["row_id"],
                        "native result request association differs",
                    )
                    save(job, result.token_ids, result.stop_reason)
                receipt["batches"].append(
                    {
                        "case_ids": [j["case"]["row_id"] for j in group],
                        "seconds": time.monotonic() - tick,
                        "batch_size": len(group),
                        "budgets": [j["remaining"] for j in group],
                        "model_forwards": counters["model_forwards"]
                        - before["model_forwards"],
                        "image_forwards": counters["image_forwards"]
                        - before["image_forwards"],
                    }
                )
        cold = consume_rows(path, receipt["manifest_sha256"], qwen.tokenizer)
        require(
            {r["case_id"] for r in cold} == {c["row_id"] for c in cases},
            "requested/returned cases differ",
        )
        write(args.output_dir / "consumer.json", cold)
        if args.mode == "qualify":
            for row in cold:
                case = next(c for c in cases if c["row_id"] == row["case_id"])
                golden = case["diagonals"][args.recipient]
                require(
                    row["generated_token_ids"] == golden["generated_token_ids"],
                    "same-arm exact token replay mismatch: " + row["case_id"],
                )
                require(
                    row["parsed"]["pred"] == golden["raw_record"]["pred"]
                    and row["parsed"]["parse_status"]
                    == golden["raw_record"]["parse_status"],
                    "same-arm parser mismatch",
                )
                for threshold in (0.5, 0.6, 0.8):
                    require(
                        sorted(
                            int(m["owner_id"])
                            for m in row["consumer_matches"][str(threshold)]
                        )
                        == golden["matched_gt_ids"][f"iou_{threshold:.2f}"],
                        "same-arm owner mismatch",
                    )
            receipt["diagonal_token_mismatches"] = []
        if args.compare_to:
            prior = {
                r["case_id"]: r
                for r in consume_rows(
                    args.compare_to / "rows.jsonl",
                    receipt["manifest_sha256"],
                    qwen.tokenizer,
                )
            }
            require(
                set(prior) == {r["case_id"] for r in cold},
                "batch comparison case-set mismatch",
            )
            fields = (
                "recipient",
                "action_source",
                "cell",
                "generated_token_ids",
                "raw_decode_text",
                "decode_stop_reason",
                "parsed",
                "consumer_matches",
            )
            failures = [
                r["case_id"]
                for r in cold
                if any(r[k] != prior[r["case_id"]][k] for k in fields)
            ]
            receipt["batch_comparison"] = {
                "path": str(args.compare_to),
                "rows_sha256": sha(args.compare_to / "rows.jsonl"),
                "mismatched_ids": failures,
            }
            require(
                not failures,
                "single/batched exact identity mismatch: " + ",".join(failures),
            )
        receipt["returned_case_ids"] = [r["case_id"] for r in cold]
        receipt["generated_tokens"] = sum(r["generated_token_count"] for r in cold)
        receipt["new_suffix_tokens"] = sum(len(r["suffix_token_ids"]) for r in cold)
        receipt["stop_counts"] = dict(Counter(r["decode_stop_reason"] for r in cold))
    finally:
        receipt["decode_seconds"] = time.monotonic() - started
        receipt["counters"] = counters
        receipt["peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        receipt["peak_reserved_bytes"] = (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        )
        receipt["host_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        model_handle.remove()
        image_handle.remove()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--original-code-root", type=Path)
    parser.add_argument("--recipient", choices=["source", "rweak"], required=True)
    parser.add_argument("--mode", choices=["qualify", "cross"], required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--case-ids")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args(argv)
    require(
        args.batch_size > 0 and args.max_wall_seconds > 0,
        "positive batch/time required",
    )
    args.output_dir = args.output_dir.resolve()
    require(not args.output_dir.exists(), "refusing to overwrite prior evidence")
    require(sha(args.manifest) == MANIFEST_SHA256, "manifest differs from root freeze")
    manifest = read(args.manifest)
    validate_manifest(manifest)
    bindings = verify_bindings(manifest, args.original_code_root)
    cases = selected_cases(manifest, args)
    config = prepare_sources(manifest, args.recipient)
    jobs = [intervention(c, args.recipient, args.mode) for c in cases]
    args.output_dir.mkdir(parents=True)
    receipt = {
        "status": "running" if args.execute else "planned",
        "manifest_sha256": sha(args.manifest),
        "execution": {
            "entry": str(Path(__file__).resolve()),
            "entry_sha256": sha(__file__),
            "git_revision": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
        },
        "verified_source_files": bindings,
        "requested_case_ids": [c["row_id"] for c in cases],
        "recipient": args.recipient,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "device": args.device,
        "max_wall_seconds": args.max_wall_seconds,
        "max_suffix_tokens": sum(j["remaining"] for j in jobs if j["kind"] != "eos"),
        "max_model_forwards": sum(j["remaining"] for j in jobs if j["kind"] != "eos"),
        "checkpoint_paths": {
            "base": config["model"]["base_model"],
            "adapter": config["adapter"]["path"],
            "embedding_delta": config["embedding_delta"]["path"],
        },
    }
    receipt["execution"]["git_dirty"] = bool(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            text=True,
        ).strip()
    )
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    try:
        if args.execute:

            def deadline(signum, frame):
                raise TimeoutError("allocated wall deadline reached")

            signal.signal(signal.SIGALRM, deadline)
            signal.alarm(args.max_wall_seconds)
            execute(args, manifest, cases, receipt)
            receipt["status"] = "success"
    except BaseException as exc:
        receipt.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        receipt["wall_seconds"] = time.monotonic() - started
        write(args.output_dir / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output_dir": str(args.output_dir),
                "case_count": len(cases),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
