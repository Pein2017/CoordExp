#!/usr/bin/env python3
"""One matched Rweak64 continuation with refreshed generated-coordinate UL."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import train_coco_gt_correction as base
from scripts.research.coco_geometric_dedup_loss import duplicate_coordinate_spans, geometric_dedup_loss
from scripts.research.coco_gt_correction_bank import canonical_sha256, file_sha256, load_bank, require
from scripts.research.reduce_coco_owner_focus import _validate_rows_against_input

EXPERIMENT_ID = "2026-09-08-coco-owner-recovery-dedup"
ROOT = base.FOCUS_ROOT.parents[1] / EXPERIMENT_ID
INITIAL_ID = "6b1f5b32dc1a5b7503ad81b691855ffcda3571f6d5433bc09838de07a88bf964"
INITIAL = base.FOCUS_ROOT / "Rweak/checkpoint-000064"
LOSS_VARIANT = "pixel-iou-gt095-coordinate-geomean-unlikelihood-v1"
REPLAY_SCHEMA = "coco-owner-recovery-replay.v1"


def seal_replay(run: Path, output: Path, checkpoint: Path, bank_path: Path) -> dict:
    """Admit exact saved natural train256 tokens, never retokenized teachers."""
    bank, records = load_bank(bank_path)
    checkpoint_manifest = base.load_cold_checkpoint(checkpoint, expected_surface="dora", expected_bank_id=bank["bank_id"])
    manifest = json.loads((run / "run_manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    require(summary.get("terminal_status") == "completed", "replay decode is incomplete")
    policy = manifest["generation_policy"]
    require(all(policy.get(key) == value for key, value in {
        "batch_size": 4, "do_sample": False, "max_new_tokens": 3084,
        "repetition_penalty": 1.0, "temperature": 0.0, "top_p": 1.0,
    }.items()), "replay generation policy changed")
    identity_key = "coco_owner_focus" if checkpoint_manifest["completed_update"] == 64 else "coco_owner_recovery"
    identity = manifest["model_identity"][identity_key]
    require(identity["checkpoint_id"] == checkpoint_manifest["checkpoint_id"], "replay checkpoint differs")
    require(manifest["backend"] == "hf" and manifest["backend_mode"] == "generate", "replay is not natural HF generation")
    from scripts.research.compare_clean_rollout_owner_coverage import _read_jsonl
    rows = _read_jsonl(run / "gt_vs_pred.jsonl")
    _validate_rows_against_input(rows, Path(bank["inputs"]["train_jsonl"]["path"]))
    require(len(rows) == len(records) == 256, "replay must retain train256")
    traces = {key: [] for key in rows}
    with (run / "pred_token_trace.jsonl").open() as handle:
        for line in handle:
            token = json.loads(line)
            if token.get("trace_type") == "generated_token" and token.get("is_pad") is not True:
                traces[token["row_id"]].append(token)
    replay = []
    for record in records:
        key = record["example_id"]
        row = rows[key]
        tokens = sorted(traces[key], key=lambda value: value["generated_step_index"])
        require([t["generated_step_index"] for t in tokens] == list(range(len(tokens))), "replay token offsets are not contiguous")
        require("".join(t["token_text"] for t in tokens) == row["raw_decode_text"], "replay token text differs from natural output")
        action = [int(t["token_id"]) for t in tokens]
        require(0 < len(action) <= 3084, "empty or over-cap replay action")
        _spans, receipt = duplicate_coordinate_spans(action, image_width=row["image_width"], image_height=row["image_height"])
        replay.append({
            "image_id": int(record["image_id"]), "example_id": key,
            "image_width": row["image_width"], "image_height": row["image_height"],
            "action_token_ids": action, "decode_stop_reason": row["decode_stop_reason"],
            **receipt,
        })
    result = {
        "schema": REPLAY_SCHEMA, "checkpoint_id": checkpoint_manifest["checkpoint_id"],
        "completed_update": checkpoint_manifest["completed_update"], "arm": checkpoint_manifest["arm"],
        "source_run": str(run.resolve()), "bank_id": bank["bank_id"],
        "source_files": {name: file_sha256(run / name) for name in ("run_manifest.json", "gt_vs_pred.jsonl", "pred_token_trace.jsonl")},
        "records": replay,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"replay": str(output), "images": len(replay), "eligible_images": sum(r["eligible_later_rows"] > 0 for r in replay), "eligible_rows": sum(r["eligible_later_rows"] for r in replay)}), flush=True)
    return result


def load_replay(path: Path, checkpoint: dict, records: list[dict]) -> dict[int, dict]:
    replay = json.loads(path.read_text())
    require(replay["schema"] == REPLAY_SCHEMA and replay["bank_id"] == base.FOCUS_BANK_ID, "foreign replay bank/schema")
    start = int(checkpoint["completed_update"])
    anchor = 64 if start < 72 else 72
    require(replay["completed_update"] == anchor, "replay refresh epoch differs")
    if start == anchor:
        require(replay["checkpoint_id"] == checkpoint["checkpoint_id"], "replay was generated from another checkpoint")
    else:
        previous = checkpoint["recipe"]["owner_recovery"]
        require(previous["replay_manifest_sha256"] == file_sha256(path), "mid-stage resume changed replay")
    by_id = {int(r["image_id"]): r for r in replay["records"]}
    require(len(by_id) == len(replay["records"]) == 256, "replay denominator differs")
    require(set(by_id) == {int(r["image_id"]) for r in records}, "replay image identity differs")
    for record in records:
        row = by_id[int(record["image_id"])]
        require(row["example_id"] == record["example_id"], "replay example identity differs")
        _spans, receipt = duplicate_coordinate_spans(row["action_token_ids"], image_width=row["image_width"], image_height=row["image_height"])
        require(all(row[key] == value for key, value in receipt.items()), "replay geometry receipt changed")
    return by_id


def recovery_recipe(arm: str, replay_path: Path, parent: dict) -> dict:
    require(arm in {"control", "dedup"}, "foreign recovery arm")
    return {
        "initial_checkpoint_id": INITIAL_ID, "initial_completed_update": 64,
        "dedup_coefficient": float(arm == "dedup"), "refresh_interval_updates": 8,
        "loss_variant": LOSS_VARIANT,
        "replay_manifest_path": str(replay_path), "replay_manifest_sha256": file_sha256(replay_path),
        "stage_parent_checkpoint_id": parent["checkpoint_id"],
        "producer_sha256": file_sha256(Path(__file__)),
        "loss_sha256": file_sha256(Path(__file__).with_name("coco_geometric_dedup_loss.py")),
    }


def run(args: argparse.Namespace) -> None:
    bank_path = args.bank_manifest.resolve(strict=True)
    bank, records = load_bank(bank_path)
    require(bank["bank_id"] == base.FOCUS_BANK_ID and bank["source_identity"]["sha256"] == base.FOCUS_SOURCE_ID, "frozen Rweak bank/Source differs")
    parent, state = base.load_checkpoint(args.checkpoint, expected_surface="dora", expected_bank_id=base.FOCUS_BANK_ID, expected_source_identity_sha256=base.FOCUS_SOURCE_ID)
    start = int(parent["completed_update"])
    if start == 64:
        require(parent["checkpoint_id"] == INITIAL_ID and parent["arm"] == "Rweak", "initial checkpoint is not Rweak64")
    else:
        require(parent["arm"] == args.arm and parent["recipe"]["experiment_id"] == EXPERIMENT_ID, "continuation arm/experiment differs")
        inherited = parent["recipe"]["owner_recovery"]
        require(inherited["initial_checkpoint_id"] == INITIAL_ID and inherited["dedup_coefficient"] == float(args.arm == "dedup") and inherited["loss_variant"] == LOSS_VARIANT, "continuation loss/lineage differs")
    stage_end = 72 if start < 72 else 80
    require(64 <= start < args.end_update <= stage_end, "run crosses fixed refresh or terminal boundary")
    require(args.end_update in {65, 66, 72, 80}, "unsupported smoke or terminal update")
    replay_path = args.replay_manifest.resolve(strict=True)
    replay = load_replay(replay_path, parent, records)
    metadata = recovery_recipe(args.arm, replay_path, parent)
    rank, world, local_rank = (int(os.environ.get(name, default)) for name, default in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    require(world == 4 and 0 <= rank < world, "recovery uses four ranks per arm")
    require(args.microbatch_images in {1, 2}, "unsupported physical microbatch")
    require(not args.check_control_parity or (args.arm == "control" and args.smoke_image_ids), "parity is control smoke only")
    by_id = {int(r["image_id"]): r for r in records}
    population = sorted(by_id)
    if args.smoke_image_ids:
        population = sorted(int(value) for value in args.smoke_image_ids.split(","))
        require(len(population) == len(set(population)) == 32 and set(population) <= set(by_id) and args.end_update <= 66, "invalid smoke population")
    else:
        require(args.end_update in {72, 80}, "production ends only at stage boundaries")
    root = args.output_root.resolve()
    require(root.is_relative_to(ROOT.resolve()), "recovery output is outside its owned root")
    if start == 64:
        require(not root.exists(), "new output root already exists")
    else:
        require(Path(args.checkpoint).resolve().parent == root, "continuation output root changed")
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=torch.device("cuda", local_rank))
    device = torch.device("cuda", local_rank)
    began = time.monotonic()
    try:
        require(len(set(base._dist_values((parent["checkpoint_id"], metadata["replay_manifest_sha256"], tuple(population), args.arm, args.end_update)))) == 1, "ranks disagree on run identity")
        if rank == 0:
            root.mkdir(parents=True, exist_ok=start > 64)
            base._prepare_source_gate(root / "source-gate")
        base._barrier()
        from src.config.fingerprint import sha256_json
        from src.config.inference import load_infer_config
        from src.data import load_raw_examples
        from src.inference.backend import open_backend_session
        from src.inference.hf_backend import HFBackendSession
        from src.inference.runtime import assemble_frontend

        resolved = load_infer_config(base.SOURCE_CONFIG)
        config = resolved.config.model_copy(update={"embedding_delta": resolved.config.embedding_delta.model_copy(update={"source_gate_root": str(root / "source-gate")})})
        require(config.model.dtype == "fp32" and config.backend.hf.attn_implementation == "sdpa", "model numerics changed")
        frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
        raw = {str(value.example_id): value for value in load_raw_examples(bank["inputs"]["train_jsonl"]["path"])}
        with open_backend_session(frontend.launch) as opened:
            require(isinstance(opened, HFBackendSession), "non-HF training backend")
            model = opened._model
            all_named = tuple(model.named_parameters())
            for _, parameter in all_named:
                parameter.requires_grad_(False)
            require(all(not isinstance(module, torch.nn.Dropout) or module.p == 0 for module in model.modules()), "active dropout")
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            named = tuple((name, p) for name, p in all_named if base._is_dora_adapter_trainable_name(name, adapter_name="default"))
            require(len(named) == 588 and all("language_model" in name for name, _ in named), "DoRA trainable surface changed")
            for _, p in named:
                p.requires_grad_(True)
            base._load_adapter(model, Path(parent["payload"]["path"]))
            require(base.parameter_layout(named) == parent["trainable_surface"]["parameter_layout"], "checkpoint trainable layout differs")
            require(base.tensor_state_sha256(named) == parent["trainable_surface"]["state_sha256"], "initial adapter tensors differ")
            scorer = base.TrajectoryScorer(model).to(device)
            ddp = torch.nn.parallel.DistributedDataParallel(scorer, device_ids=[local_rank], broadcast_buffers=False)
            # DDP initialization broadcasts even frozen parameters, incrementing
            # version counters without changing values. Guard training after it.
            frozen = {name: p._version for name, p in all_named if not p.requires_grad}
            optimizer = torch.optim.AdamW([p for _, p in named], lr=base.LEARNING_RATE, betas=base.BETAS, eps=base.EPSILON, weight_decay=0., foreach=False)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            torch.set_rng_state(state["torch_rng_state"])
            torch.cuda.set_rng_state(state["cuda_rng_state"])
            used = sorted({i for update in range(start, args.end_update) for i in base.schedule_batch(population, update, global_batch_size=32, seed=base.FOCUS_SEED)[rank::world]})
            cache = {i: base._materialize(opened, frontend, config, raw[by_id[i]["example_id"]], by_id[i]) for i in used}
            require(all((raw[by_id[i]["example_id"]].image.width, raw[by_id[i]["example_id"]].image.height) == (replay[i]["image_width"], replay[i]["image_height"]) for i in used), "replay pixel dimensions differ")
            del state
            torch.cuda.reset_peak_memory_stats(device)
            for update in range(start, args.end_update):
                update_start = time.monotonic()
                batch = base.schedule_batch(population, update, global_batch_size=32, seed=base.FOCUS_SEED)
                local = batch[rank::world]
                optimizer.zero_grad(set_to_none=True)
                losses, padded_receipts = [], []
                replay_receipts = []
                forwards = backwards = 0
                # All replay work is no_sync and precedes the same final supervised
                # sync on every rank. Zero-repeat images need no model forward.
                if args.arm == "dedup":
                    for i in local:
                        row = replay[i]
                        if not row["eligible_later_rows"]:
                            continue
                        prompt, inputs, grid = cache[i]
                        with ddp.no_sync():
                            ce = ddp([inputs], [grid], [prompt], [row["action_token_ids"]], True)[0]
                            penalty, receipt = geometric_dedup_loss(ce, row["action_token_ids"], image_width=row["image_width"], image_height=row["image_height"])
                            (penalty * world / 32).backward()
                        replay_receipts.append({"image_id": i, **receipt, "loss": float(penalty.detach())})
                        forwards += 1
                        backwards += 1
                        del ce, penalty
                local_dedup_grad_sq = sum(float(p.grad.detach().double().square().sum()) for _, p in named if p.grad is not None)
                require(math.isfinite(local_dedup_grad_sq), "nonfinite dedup gradient")
                if replay_receipts:
                    require(local_dedup_grad_sq > 0, "eligible replay produced no parameter gradient")
                components = []
                for i in local:
                    anchor, correction = base._objective_record(by_id[i], "Rweak")
                    components.append((i, "canonical", anchor, 1.))
                    if correction is not None:
                        components.append((i, "correction", correction, base.correction_multiplier(by_id[i], "Rweak")))
                chunks = [local[k:k + args.microbatch_images] for k in range(0, len(local), args.microbatch_images)]
                for index, ids in enumerate(chunks):
                    chunk = [component for component in components if component[0] in ids]
                    context = nullcontext() if index == len(chunks) - 1 else ddp.no_sync()
                    with context:
                        ce_batch = ddp([cache[i][1] for i, *_ in chunk], [cache[i][2] for i, *_ in chunk], [cache[i][0] for i, *_ in chunk], [obj["action_token_ids"] for _, _, obj, _ in chunk], True)
                        values = [base.component_loss(ce, kind, obj, coefficient) for (_, kind, obj, coefficient), ce in zip(chunk, ce_batch, strict=True)]
                        (torch.stack(values).sum() * world / 32).backward()
                    losses.extend({"image_id": i, "kind": kind, "loss": float(value.detach()), "coefficient": coefficient} for (i, kind, _, coefficient), value in zip(chunk, values, strict=True))
                    padded_receipts.append({"image_ids": ids, **scorer.last_batch_receipt})
                    forwards += 1
                    backwards += 1
                    del ce_batch, values
                parity = None
                if args.check_control_parity:
                    # Independent scalar legacy image-loss path against the new
                    # padded control path, same weights/optimizer step and data.
                    reference_grad = [p.grad.detach().cpu().clone() for _, p in named]
                    optimizer.zero_grad(set_to_none=True)
                    scalar_losses = []
                    for index, (i, kind, objective, coefficient) in enumerate(components):
                        context = nullcontext() if index == len(components) - 1 else ddp.no_sync()
                        prompt, inputs, grid = cache[i]
                        with context:
                            ce = ddp(inputs, grid, prompt, objective["action_token_ids"])
                            if kind == "canonical":
                                loss = ce.mean()
                            else:
                                mask = torch.tensor(objective["direct_loss_mask"], device=device, dtype=ce.dtype)
                                loss = (ce * mask).sum() / objective["fixed_denominator"] * coefficient
                            (loss * world / 32).backward()
                        scalar_losses.append(float(loss.detach()))
                        forwards += 1
                        backwards += 1
                    difference_sq = sum(float((p.grad.detach().cpu().double() - ref.double()).square().sum()) for (_, p), ref in zip(named, reference_grad, strict=True))
                    reference_sq = sum(float(ref.double().square().sum()) for ref in reference_grad)
                    relative = math.sqrt(difference_sq / max(reference_sq, 1e-30))
                    loss_max = max(abs(r["loss"] - value) for r, value in zip(losses, scalar_losses, strict=True))
                    parity = {"scalar_vs_padded_gradient_relative_l2": relative, "component_loss_max_abs": loss_max}
                    require(relative <= 1e-4 and loss_max <= 4e-6, "lambda0 scalar/padded parity failed")
                    # Preserve the exact qualified padded update, not its scalar check.
                    for (_, p), ref in zip(named, reference_grad, strict=True):
                        p.grad.copy_(ref.to(device))
                require(base._all_true(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named), device), "missing/nonfinite trainable gradient")
                require(base._all_true(all(p.grad is None for _, p in all_named if not p.requires_grad), device), "frozen parameter gradient")
                norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1., error_if_nonfinite=True))
                optimizer.step()
                require(all(p._version == frozen[name] for name, p in all_named if not p.requires_grad), "frozen parameters changed")
                per_rank = base._dist_values({
                    "rank": rank, "image_ids": local, "supervised_components": losses,
                    "padded_batches": padded_receipts, "replay": replay_receipts,
                    "eligible_images": sum(replay[i]["eligible_later_rows"] > 0 for i in local),
                    "eligible_rows": sum(replay[i]["eligible_later_rows"] for i in local),
                    "dedup_local_gradient_norm": math.sqrt(local_dedup_grad_sq),
                    "unclipped_global_gradient_norm": norm, "parity": parity,
                    "model_forward_count": forwards, "backward_count": backwards,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                })
                if rank == 0:
                    runtime = {
                        "world_size": world, "batch_image_ids": batch, "per_rank": per_rank,
                        "update_seconds": time.monotonic() - update_start, "elapsed_seconds": time.monotonic() - began,
                        "global_supervised_loss": sum(r["loss"] for rank_receipt in per_rank for r in rank_receipt["supervised_components"]) / 32,
                        "global_dedup_loss": sum(r["loss"] for rank_receipt in per_rank for r in rank_receipt["replay"]) / 32,
                        "frozen_parameter_versions_unchanged": True, "unexpected_frozen_gradient_count": 0,
                        "peak_gpu_allocated_bytes": max(r["peak_gpu_allocated_bytes"] for r in per_rank),
                        "peak_gpu_reserved_bytes": max(r["peak_gpu_reserved_bytes"] for r in per_rank),
                        "peak_host_rss_bytes": max(r["peak_host_rss_bytes"] for r in per_rank),
                        "smoke": bool(args.smoke_image_ids),
                    }
                    saved = base._publish_checkpoint(output_root=root, completed_update=update + 1, arm=args.arm, surface="dora", bank_manifest_path=bank_path, bank_manifest=bank, model=model, residual=None, named=named, optimizer=optimizer, runtime=runtime, global_batch_size=32, seed=base.FOCUS_SEED, experiment_id=EXPERIMENT_ID, microbatch_images=args.microbatch_images, owner_recovery=metadata)
                    print(json.dumps({"arm": args.arm, "completed_update": update + 1, "checkpoint_id": saved["checkpoint_id"], "supervised_loss": runtime["global_supervised_loss"], "dedup_loss": runtime["global_dedup_loss"], "seconds": runtime["update_seconds"]}), flush=True)
                base._barrier()
    finally:
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal-replay")
    seal.add_argument("--run", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--checkpoint", type=Path, required=True)
    seal.add_argument("--bank-manifest", type=Path, default=base.DEFAULT_BANK / "manifest.json")
    train = commands.add_parser("run")
    train.add_argument("--arm", choices=("control", "dedup"), required=True)
    train.add_argument("--checkpoint", type=Path, default=INITIAL)
    train.add_argument("--bank-manifest", type=Path, default=base.DEFAULT_BANK / "manifest.json")
    train.add_argument("--replay-manifest", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--end-update", type=int, required=True)
    train.add_argument("--smoke-image-ids", default="")
    train.add_argument("--microbatch-images", type=int, default=2)
    train.add_argument("--check-control-parity", action="store_true")
    args = parser.parse_args()
    if args.command == "seal-replay":
        seal_replay(args.run.resolve(strict=True), args.output.resolve(), args.checkpoint.resolve(strict=True), args.bank_manifest.resolve(strict=True))
    else:
        run(args)


if __name__ == "__main__":
    main()
