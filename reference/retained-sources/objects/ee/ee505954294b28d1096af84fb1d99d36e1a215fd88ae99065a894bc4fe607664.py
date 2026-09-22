"""Eight-rank Stable50 geometric-dedup pilot trainer.

The input packet owns the selected populations and their semantic masks.  This
module owns only the shared DDP execution, lagged replay, and sealed training
receipt.  It never introduces ground-truth labels into the objective.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import torch

from src.artifacts import load_canonical_json
from .candidate_opportunity import digest, file_hash, require
from .route_access import CONFIG, ROOT, checked_ids, checkpoint_config, publish


OUTPUT = ROOT / "2026-09-11-stable50-geometric-dedup"
ONLINE_IMAGE_IDS = (
    "9813", "158044", "248167", "274509", "351017", "417044", "477415", "502725",
)
REFRESH_STEPS = (0, 8, 16, 24)
UPDATES = 32
WORLD_SIZE = 8
REFERENCE_COUNT = 56
REFERENCE_ACTION_STATES = 6056
ONLINE_ACTION_STATES = 13_340
CAP = 3084
EOS = 151645
OPTIMIZER = dict(lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False)
EXPECTED_TRAINABLE_TENSORS = 588
EXPECTED_TRAINABLE_SCALARS = 18_006_016
MODEL_FORWARD_CEILING = 4 * (CAP + 1) + 267
IMAGE_FORWARD_COUNT = 271
PARITY_RANK = 1
PARITY_FORWARDS = 2
DECODER_LAYER_COUNT = 28
PARITY_RELATIVE_L2_TOLERANCE = 1e-5
PARITY_MAX_ABS_TOLERANCE = 1e-5


def rank_model_forward_ceiling(rank):
    return MODEL_FORWARD_CEILING + (PARITY_FORWARDS if rank == PARITY_RANK else 0)


def rank_image_forward_count(rank):
    return IMAGE_FORWARD_COUNT + (PARITY_FORWARDS if rank == PARITY_RANK else 0)


def install_language_decoder_checkpointing(model):
    """Checkpoint exactly the language decoder blocks while preserving eval mode."""
    from torch.utils.checkpoint import checkpoint

    require(not model.training, "checkpoint installation requires eval-mode model")
    owners = [(name, module) for name, module in model.named_modules()
              if name.endswith("language_model")]
    require(len(owners) == 1, "single named language decoder owner")
    owner_name, owner = owners[0]
    layers = owner.layers
    require(len(layers) == DECODER_LAYER_COUNT and
            all(not layer.training for layer in layers),
            "exact 28 eval-mode language decoder blocks")
    state = dict(
        enabled=False, phase="disabled", decoder_layers=len(layers),
        decoder_module_name=owner_name,
        use_reentrant=False, layer_calls=0, checkpoint_invocations=0,
        checkpoint_body_calls=0, bypass_no_grad_calls=0,
        bypass_disabled_calls=0,
    )

    def wrapper(original_forward):
        # Capture each already-bound method here; never close over the loop variable.
        def checkpointed_forward(*args, **kwargs):
            state["layer_calls"] += 1
            if not torch.is_grad_enabled():
                state["bypass_no_grad_calls"] += 1
                return original_forward(*args, **kwargs)
            if not state["enabled"]:
                state["bypass_disabled_calls"] += 1
                return original_forward(*args, **kwargs)
            state["checkpoint_invocations"] += 1

            def invoke(*inner_args, **inner_kwargs):
                state["checkpoint_body_calls"] += 1
                return original_forward(*inner_args, **inner_kwargs)

            return checkpoint(invoke, *args, use_reentrant=False, **kwargs)
        return checkpointed_forward

    for layer in layers:
        layer.forward = wrapper(layer.forward)
    require(not model.training and all(not layer.training for layer in layers),
            "checkpoint installation changed eval mode")
    return state


def checkpointing_receipt(model, state):
    layers = model.get_submodule(state["decoder_module_name"]).layers
    require(len(layers) == DECODER_LAYER_COUNT and not model.training and
            all(not layer.training for layer in layers),
            "checkpointed model left eval mode")
    body_calls = int(state["checkpoint_body_calls"])
    invocations = int(state["checkpoint_invocations"])
    return dict(
        mode="non_reentrant_per_language_decoder_block_grad_only",
        decoder_layers=DECODER_LAYER_COUNT, use_reentrant=False,
        decoder_module_name=state["decoder_module_name"],
        enabled=bool(state["enabled"]), phase=str(state["phase"]),
        layer_calls=int(state["layer_calls"]),
        checkpoint_invocations=invocations,
        checkpoint_body_calls=body_calls,
        recompute_body_calls=max(0, body_calls - invocations),
        bypass_no_grad_calls=int(state["bypass_no_grad_calls"]),
        bypass_disabled_calls=int(state["bypass_disabled_calls"]),
        model_eval=True, all_decoder_layers_eval=True,
    )


def parity_metrics(loss_off, loss_on, gradient_off, gradient_on):
    require(gradient_off.shape == gradient_on.shape and gradient_off.ndim == 1,
            "parity gradient vector shape")
    difference = (gradient_on.double() - gradient_off.double())
    denominator = max(float(gradient_off.double().norm()), 1e-30)
    result = dict(
        loss_off=float(loss_off), loss_on=float(loss_on),
        loss_abs_error=abs(float(loss_on) - float(loss_off)),
        gradient_relative_l2=float(difference.norm()) / denominator,
        gradient_max_abs=float(difference.abs().max()) if difference.numel() else 0.0,
        gradient_off_l2=float(gradient_off.double().norm()),
        gradient_on_l2=float(gradient_on.double().norm()),
    )
    result["passed"] = (
        math.isfinite(result["loss_abs_error"]) and
        result["loss_abs_error"] <= PARITY_MAX_ABS_TOLERANCE and
        result["gradient_relative_l2"] <= PARITY_RELATIVE_L2_TOLERANCE and
        result["gradient_max_abs"] <= PARITY_MAX_ABS_TOLERANCE
    )
    return result


def validate_packet(packet):
    """Validate the trainer boundary, leaving row semantics to geometric_dedup."""
    required = {
        "model", "config", "online_cases", "reference_cases", "source_files",
        "numerical_policy", "objective", "optimizer", "limits",
    }
    require(required <= set(packet), "geometric-dedup input packet fields")
    online, references = packet["online_cases"], packet["reference_cases"]
    require(len(online) == WORLD_SIZE and len(references) == REFERENCE_COUNT,
            "fixed 8/56 populations")
    require(tuple(str(c["image_id"]) for c in online) == ONLINE_IMAGE_IDS,
            "fixed rank-ordered online panel")
    required_case = {
        "key", "example_id", "image_id", "group", "prompt_token_ids", "action_ids",
        "action_ids_sha256", "stop_reason", "initial_layout", "case",
        "image_width", "image_height",
    }
    require(all(required_case <= set(c) for c in online + references),
            "case packet fields")
    keys = [str(c["key"]) for c in online + references]
    require(len(set(keys)) == len(keys) and
            not {str(c["image_id"]) for c in online} &
                {str(c["image_id"]) for c in references},
            "population identity/disjointness")
    for case in online + references:
        ids = checked_ids(case["action_ids"], case["stop_reason"])
        require(digest(ids) == case["action_ids_sha256"] and
                ids == case["action_ids"], "saved Stable action identity")
        require(str(case["key"]) == str(case["example_id"]) and
                list(case["prompt_token_ids"]) == list(case["group"]["prompt_token_ids"]),
                "case/group identity")
    require(sum(len(c["action_ids"]) for c in references) == REFERENCE_ACTION_STATES,
            "fixed reference action-state count")
    require(sum(len(c["action_ids"]) for c in online) == ONLINE_ACTION_STATES and
            sum(c["stop_reason"] == "length" and len(c["action_ids"]) == CAP
                for c in online) == 4,
            "fixed online action-state/cap count")
    require(sum(len(c["initial_layout"]["duplicate_row_indices"]) for c in online) == 495 and
            not online[0]["initial_layout"]["duplicate_row_indices"] and
            all(c["initial_layout"]["action_token_count"] == len(c["action_ids"])
                for c in online + references) and
            all(not c["initial_layout"]["duplicate_row_indices"] for c in references),
            "frozen initial mask counts")
    observed_optimizer = dict(packet["optimizer"])
    if isinstance(observed_optimizer.get("betas"), list):
        observed_optimizer["betas"] = tuple(observed_optimizer["betas"])
    require(observed_optimizer == OPTIMIZER, "fixed fresh AdamW recipe")
    require(packet["limits"]["world_size"] == WORLD_SIZE and
            packet["limits"]["updates"] == UPDATES and
            packet["limits"]["refresh_steps"] == list(REFRESH_STEPS) and
            packet["limits"]["train_seconds_per_rank"] == 3600 and
            packet["limits"]["train_forwards_per_rank"] == 256 and
            packet["limits"]["teacher_forwards_per_rank"] == 11 and
            packet["limits"]["max_model_forwards_per_rank"] == MODEL_FORWARD_CEILING and
            packet["limits"]["max_image_forwards_per_rank"] == IMAGE_FORWARD_COUNT and
            packet["limits"]["decoder_checkpoint_layers"] == DECODER_LAYER_COUNT and
            packet["limits"]["parity_rank"] == PARITY_RANK and
            packet["limits"]["parity_forwards"] == PARITY_FORWARDS and
            packet["limits"]["max_model_forwards_by_rank"] ==
                [rank_model_forward_ceiling(rank) for rank in range(WORLD_SIZE)] and
            packet["limits"]["max_image_forwards_by_rank"] ==
                [rank_image_forward_count(rank) for rank in range(WORLD_SIZE)],
            "fixed dose/refresh/time limit")
    model = packet["model"]
    require(model["dtype"] == "fp32" and model["attention_implementation"] == "sdpa",
            "FP32 SDPA packet")
    require(packet["numerical_policy"]["activation_checkpointing"] ==
            "non_reentrant_language_blocks_grad_only_eval_mode",
            "fixed activation-checkpoint policy")
    return packet


def rank_items(packet, rank):
    require(0 <= rank < WORLD_SIZE, "rank range")
    items = [dict(kind="online", case=packet["online_cases"][rank])]
    items.extend(dict(kind="reference", case=c)
                 for c in packet["reference_cases"][rank::WORLD_SIZE])
    require(len(items) == 8 and sum(x["kind"] == "reference" for x in items) == 7,
            "one online plus seven references per rank")
    return items


def ddp_item_loss(kind, unlikelihood, kl):
    """Return the loss to backpropagate before DDP's rank mean.

    Each rank owns one of eight online images and seven of 56 references.  DDP
    averages reduced gradients, so local global-normalized terms are multiplied
    by eight before backward.
    """
    if kind == "online":
        return unlikelihood + 10.0 * kl
    require(kind == "reference", "loss item kind")
    return (100.0 / 7.0) * kl


def backward_rank_items(ddp, items, loss_for_item):
    """Accumulate equal eight-item rank work and synchronize only item eight."""
    require(len(items) == 8, "fixed local item count")
    records = []
    for index, item in enumerate(items):
        synchronized = index == len(items) - 1
        with nullcontext() if synchronized else ddp.no_sync():
            loss, details = loss_for_item(item)
            require(loss.ndim == 0, "scalar item loss")
            loss.backward()
        records.append(dict(index=index, kind=item["kind"], synchronized=synchronized,
                            loss=float(loss.detach()), **details))
    require([r["synchronized"] for r in records] == [False] * 7 + [True],
            "DDP synchronization choreography")
    return records


def update_scalar_status(step, raw_gradient_norm, movement_l2, total_movement_l2):
    """Enforce nonzero mechanics only for the first two real updates."""
    values = (raw_gradient_norm, movement_l2, total_movement_l2)
    require(all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values),
            "finite nonnegative update scalars")
    if step <= 2:
        require(raw_gradient_norm > 0.0 and movement_l2 > 0.0,
                "two-step gate requires nonzero gradient and update")
    return dict(zero_gradient=raw_gradient_norm == 0.0,
                zero_movement=movement_l2 == 0.0)


class GeometricDedupScorer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, prompt_ids, action_ids, layout, reference_logp, kind):
        from src.qwen.native import prepare_replay
        from .geometric_dedup import duplicate_unlikelihood, preservation_kl

        replay = prepare_replay(
            self.model, inputs, prompt_token_ids=prompt_ids,
            continuation_token_ids=action_ids,
        )
        logits = replay.aligned_logits(self.model(**replay.inputs).logits)
        require(logits.shape[0] == len(action_ids) and
                replay.target_ids.tolist() == list(action_ids),
                "exact lagged-replay causal targets")
        kl = preservation_kl(logits, reference_logp, layout["kl_positions"])
        unlikelihood = (duplicate_unlikelihood(logits, replay.target_ids, layout)
                        if kind == "online" else kl * 0.0)
        loss = ddp_item_loss(kind, unlikelihood, kl)
        return loss, unlikelihood.detach(), kl.detach()


def _teacher_reference(model, inputs, prompt_ids, action_ids, layout):
    from src.qwen.native import prepare_replay

    positions = layout["kl_positions"]
    with torch.no_grad():
        replay = prepare_replay(
            model, inputs, prompt_token_ids=prompt_ids,
            continuation_token_ids=action_ids,
        )
        logits = replay.aligned_logits(model(**replay.inputs).logits)
        require(logits.shape[0] == len(action_ids) and
                replay.target_ids.tolist() == list(action_ids),
                "exact teacher causal targets")
        reference = (torch.log_softmax(logits[positions], dim=-1)
                     if positions else logits.new_empty((0, logits.shape[1]))).detach().cpu()
    require(not reference.requires_grad and reference.grad_fn is None and
            reference.shape == (len(positions), logits.shape[1]),
            "detached teacher reference")
    return reference


def _materialize_cases(qwen, packet, items, device):
    from src.qwen.native import prepare_native_inputs
    from probes.source_rweak_row_cross.run import build_requests

    result = []
    for item in items:
        case = item["case"]
        requests, _ = build_requests(qwen, packet["config"], [case["case"]])
        request = requests[0]
        require(list(request.expected_token_ids) == case["prompt_token_ids"],
                "reconstructed prompt identity")
        batch = prepare_native_inputs(
            qwen.processor, requests, device=device, record_media_identity=True,
        )
        group = case["group"]
        require(list(batch.prompt_token_ids[0]) == case["prompt_token_ids"] and
                batch.media_sha256[0] == group["executed_media_sha256"] and
                list(batch.image_grids[0]) == group["observed_image_grid_thw"],
                "executed prompt/media/grid identity")
        result.append(dict(item=item, batch=batch,
                           inputs=dict(batch.inputs), prompt_ids=list(batch.prompt_token_ids[0])))
    return result


def execute_rank(output):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from src.adapters.dora import select_dora_parameters
    from src.config.inference import load_research_infer_config
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from .geometric_dedup import trajectory_layout
    from .runtime import load_policy
    from .selective_preservation_dense import optimizer_hash
    from .train import (
        _all_true, _dist_values, _parameter_layout, _save_adapter_only,
        _tensor_state_hash,
    )

    rank, local, world = [int(os.environ.get(k, "-1"))
                          for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")]
    require(world == WORLD_SIZE and rank == local and 0 <= rank < WORLD_SIZE and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3,4,5,6,7",
            "single-node eight-rank topology")
    training = output / "training"
    run = training / "ranks" / f"rank{rank}"
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counters = dict(
        model_loads=0, model_forwards=0, image_forwards=0,
        student_generation_forwards=0, student_train_forwards=0,
        parity_forwards=0,
        teacher_reference_forwards=0, continuations=0, generated_tokens=0,
        refreshes=0, updates=0, ddp_synchronized_backwards=0,
        explicit_collective_calls=0, zero_gradient_updates=0, zero_movement_updates=0,
    )
    status, error, final_state = "failed", None, {}

    def expired(*_):
        raise TimeoutError("3600-second rank invocation ceiling")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(3600)
    with (run / "execution.log").open("x") as log, \
            redirect_stdout(log), redirect_stderr(log):
      try:
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)

        def gather(value):
            counters["explicit_collective_calls"] += 1
            return _dist_values(value)

        def all_true(value):
            counters["explicit_collective_calls"] += 1
            return _all_true(value, device)

        def barrier():
            counters["explicit_collective_calls"] += 1
            dist.barrier()

        packet = validate_packet(load_canonical_json(output / "inputs.json"))
        require(len(set(gather(file_hash(output / "inputs.json")))) == 1,
                "rank input identity")
        for path, sha in packet["source_files"].items():
            require(file_hash(path) == sha, f"frozen source changed: {path}")
        base_config = load_research_infer_config(CONFIG).config
        require(base_config.model_dump(mode="json") == packet["config"],
                "packet prompt/runtime configuration identity")
        config = checkpoint_config(
            base_config,
            packet["model"]["current_adapter"]["root"],
        )
        require(str(config.model.base_model) == packet["model"]["base_model_path"] and
                str(config.embedding_delta.path) == packet["model"]["source_embedding"]["root"] and
                config.model.dtype == "fp32" and
                config.backend.hf.attn_implementation == "sdpa" and
                config.backend.hf.patch_embed_linearization == "enabled",
                "Stable50 base/embedding/numerical identity")
        for identity in (packet["model"]["current_adapter"],
                         packet["model"]["source_embedding"]):
            for entry in identity["files"]:
                require(file_hash(Path(identity["root"]) / entry["relative_path"]) ==
                        entry["sha256"], "immutable starting model bytes")

        student_qwen, student_identity = load_policy(config, device=device)
        teacher_qwen, teacher_identity = load_policy(config, device=device)
        counters["model_loads"] = 2
        student, teacher = student_qwen.model, teacher_qwen.model
        student.eval(); teacher.eval()
        require(student_identity == teacher_identity,
                "student/teacher loaded checkpoint identity")
        for parameter in student.parameters():
            parameter.requires_grad_(False)
        named = select_dora_parameters(student, towers=("language",), adapter_name="default")
        require(len(named) == EXPECTED_TRAINABLE_TENSORS and
                sum(p.numel() for _, p in named) == EXPECTED_TRAINABLE_SCALARS and
                all("language_model" in n and not any(x in n for x in
                    ("visual", "merger", "embed_tokens", "lm_head")) for n, _ in named),
                "language-only DoRA training surface")
        for _, parameter in named:
            parameter.requires_grad_(True)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        publish(run / "student-model.json", student_identity)
        publish(run / "teacher-model.json", teacher_identity)

        active_model = "idle"
        def student_count(*_):
            counters["model_forwards"] += 1
            if active_model == "generation":
                counters["student_generation_forwards"] += 1
            elif active_model == "train":
                counters["student_train_forwards"] += 1
            elif active_model == "parity":
                counters["parity_forwards"] += 1
            require(counters["model_forwards"] <= rank_model_forward_ceiling(rank),
                    "rank model-forward ceiling")
        def teacher_count(*_):
            counters["model_forwards"] += 1
            counters["teacher_reference_forwards"] += 1
            require(counters["model_forwards"] <= rank_model_forward_ceiling(rank),
                    "rank model-forward ceiling")
        student.register_forward_pre_hook(student_count)
        teacher.register_forward_pre_hook(teacher_count)
        student_visual = [m for n, m in student.named_modules() if n.endswith("visual")]
        teacher_visual = [m for n, m in teacher.named_modules() if n.endswith("visual")]
        require(len(student_visual) == len(teacher_visual) == 1, "vision module identity")
        student_visual[0].register_forward_pre_hook(
            lambda *_: counters.__setitem__("image_forwards", counters["image_forwards"] + 1))
        teacher_visual[0].register_forward_pre_hook(
            lambda *_: counters.__setitem__("image_forwards", counters["image_forwards"] + 1))

        selected = {id(p) for _, p in named}
        frozen = [(n, p) for n, p in student.named_parameters() if id(p) not in selected]
        initial_adapter = _tensor_state_hash(named)
        initial_frozen = _tensor_state_hash(frozen)
        student_hash = _tensor_state_hash(list(student.named_parameters()))
        teacher_hash = _tensor_state_hash(list(teacher.named_parameters()))
        require(student_hash == teacher_hash, "student/teacher initial tensor identity")
        require(len(set(gather((initial_adapter, initial_frozen, teacher_hash)))) == 1,
                "initial rank tensors differ")
        checkpointing = install_language_decoder_checkpointing(student)
        scorer = GeometricDedupScorer(student)
        layout = _parameter_layout(named)
        publish(run / "trainable_layout.json", layout)

        assigned = rank_items(packet, rank)
        materialized = _materialize_cases(student_qwen, packet, assigned, device)
        online = materialized[0]
        references = materialized[1:]
        torch.cuda.reset_peak_memory_stats()
        reference_cache = {}
        active_model = "teacher"
        for entry in references:
            case = entry["item"]["case"]
            reference_cache[str(case["key"])] = _teacher_reference(
                teacher, entry["inputs"], entry["prompt_ids"],
                case["action_ids"], case["initial_layout"],
            )
        active_model = "idle"
        reference_cards = [
            dict(key=key, shape=list(value.shape),
                 teacher_logp_sha256=_tensor_state_hash([("teacher_logp", value)]),
                 cache_bytes=value.numel() * value.element_size())
            for key, value in reference_cache.items()
        ]
        reference_cache_hashes = {
            card["key"]: card["teacher_logp_sha256"] for card in reference_cards
        }
        publish(run / "reference-cache.json", reference_cards)

        refresh_records = []
        online_state = None
        policy = NativeGenerationPolicy(
            temperature=0.0, top_p=1.0, repetition_penalty=1.0,
            top_k=0, use_model_defaults=False,
        )

        def refresh(step):
            nonlocal active_model
            case, batch = online["item"]["case"], online["batch"]
            active_model = "generation"
            checkpointing["phase"] = "generation"
            tick = time.monotonic()
            result = generate_continuations(
                student, batch, extensions=[[]], budgets=[CAP],
                eos_token_id=EOS, pad_token_id=student_qwen.tokenizer.pad_token_id,
                policy=policy, trace="none",
            )[0]
            counters["continuations"] += 1
            counters["generated_tokens"] += len(result.token_ids)
            ids = list(result.token_ids)
            checked_ids(ids, result.stop_reason)
            if step == 0:
                require(ids == case["action_ids"] and
                        result.stop_reason == case["stop_reason"],
                        "refresh0 must reproduce saved Stable trajectory")
            current_layout = trajectory_layout(
                ids, student_qwen.tokenizer,
                image_width=case["image_width"], image_height=case["image_height"],
                row_id=str(case["example_id"]),
            )
            if step == 0:
                require(current_layout == case["initial_layout"],
                        "refresh0 parsed layout identity")
            active_model = "teacher"
            checkpointing["phase"] = "teacher"
            teacher_logp = _teacher_reference(
                teacher, online["inputs"], online["prompt_ids"], ids, current_layout,
            )
            active_model = "idle"
            checkpointing["phase"] = "idle"
            card = dict(
                refresh_step=step, key=case["key"], image_id=str(case["image_id"]),
                action_ids=ids, action_ids_sha256=digest(ids), stop_reason=result.stop_reason,
                generation_seconds=time.monotonic() - tick,
                layout=current_layout, layout_sha256=digest(current_layout),
                valid_rows=current_layout["valid_row_count"],
                duplicate_rows=len(current_layout["duplicate_row_indices"]),
                parser_drops=current_layout["parser_drops"],
                invalid_geometry_rows=len(current_layout["invalid_geometry_rows"]),
                teacher_states=teacher_logp.shape[0],
                teacher_logp_sha256=_tensor_state_hash([("teacher_logp", teacher_logp)]),
                teacher_cache_bytes=teacher_logp.numel() * teacher_logp.element_size(),
            )
            publish(run / f"refresh-{step:02d}.json", card)
            counters["refreshes"] += 1
            refresh_records.append(card)
            return dict(action_ids=ids, layout=current_layout,
                        teacher_logp=teacher_logp)

        online_state = refresh(0)
        initial_reference_bytes = sum(x["cache_bytes"] for x in reference_cards)
        parity_local = None
        if rank == PARITY_RANK:
            case = online["item"]["case"]
            require(str(case["image_id"]) == "158044" and
                    len(online_state["action_ids"]) == 246 and
                    len(online_state["layout"]["duplicate_row_indices"]) == 5,
                    "fixed short real parity trajectory")

            def parity_pass(enabled):
                nonlocal active_model
                checkpointing["enabled"] = enabled
                checkpointing["phase"] = "parity_on" if enabled else "parity_off"
                student.zero_grad(set_to_none=True)
                before_counts = checkpointing_receipt(student, checkpointing)
                active_model = "parity"
                reference = online_state["teacher_logp"].to(device)
                loss, ul, kl = scorer(
                    online["inputs"], online["prompt_ids"],
                    online_state["action_ids"], online_state["layout"],
                    reference, "online",
                )
                require(bool(torch.isfinite(loss)) and abs(float(kl)) <= 1e-6,
                        "finite same-trajectory parity loss")
                loss.backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for _, p in named), "finite parity gradients")
                vector = torch.cat([p.grad.detach().flatten() for _, p in named]).cpu()
                after_counts = checkpointing_receipt(student, checkpointing)
                active_model = "idle"
                student.zero_grad(set_to_none=True)
                del reference
                return dict(
                    loss=float(loss.detach()), unlikelihood=float(ul), kl=float(kl),
                    gradient=vector,
                    gradient_sha256=_tensor_state_hash([("gradient", vector)]),
                    checkpoint_invocations=(after_counts["checkpoint_invocations"] -
                                            before_counts["checkpoint_invocations"]),
                    checkpoint_body_calls=(after_counts["checkpoint_body_calls"] -
                                           before_counts["checkpoint_body_calls"]),
                    bypass_disabled_calls=(after_counts["bypass_disabled_calls"] -
                                           before_counts["bypass_disabled_calls"]),
                )

            adapter_before_parity = _tensor_state_hash(named)
            off = parity_pass(False)
            on = parity_pass(True)
            metrics = parity_metrics(off["loss"], on["loss"],
                                     off.pop("gradient"), on.pop("gradient"))
            require(metrics["passed"] and metrics["gradient_off_l2"] > 0 and
                    metrics["gradient_on_l2"] > 0 and
                    off["bypass_disabled_calls"] == DECODER_LAYER_COUNT and
                    off["checkpoint_invocations"] == 0 and
                    on["checkpoint_invocations"] == DECODER_LAYER_COUNT and
                    on["checkpoint_body_calls"] > on["checkpoint_invocations"] and
                    _tensor_state_hash(named) == adapter_before_parity,
                    "real short-trajectory checkpoint parity")
            parity_local = dict(
                status="passed", rank=rank, image_id=str(case["image_id"]),
                action_tokens=len(online_state["action_ids"]), duplicate_rows=5,
                tolerances=dict(relative_l2=PARITY_RELATIVE_L2_TOLERANCE,
                                max_abs=PARITY_MAX_ABS_TOLERANCE),
                metrics=metrics, checkpoint_off=off, checkpoint_on=on,
                no_optimizer_step=True, adapter_hash=adapter_before_parity,
            )
        checkpointing["enabled"] = True
        checkpointing["phase"] = "idle"
        parity_records = gather(parity_local)
        parity_record = parity_records[PARITY_RANK]
        require(sum(record is not None for record in parity_records) == 1 and
                parity_record["status"] == "passed",
                "single-rank real checkpoint parity evidence")
        if rank == 0:
            publish(training / "checkpoint-parity.json", parity_record)
        barrier()
        require(load_canonical_json(training / "checkpoint-parity.json") == parity_record,
                "checkpoint parity publication identity")

        ddp = DDP(scorer, device_ids=[local], output_device=local,
                  broadcast_buffers=False, init_sync=False)
        frozen_hash = _tensor_state_hash(frozen)
        require(frozen_hash == initial_frozen, "DDP construction changed frozen bytes")
        frozen_versions = [(p, p._version) for _, p in frozen]
        original = [p.detach().clone() for _, p in named]
        optimizer = torch.optim.AdamW([p for _, p in named], **OPTIMIZER)
        require(not optimizer.state and all(p.grad is None for _, p in named),
                "fresh optimizer after cleared parity gradients")
        update_records = []
        smoke = {}
        for step in range(1, UPDATES + 1):
            if step - 1 in REFRESH_STEPS[1:]:
                online_state = refresh(step - 1)
                if step - 1 == REFRESH_STEPS[-1]:
                    require(_tensor_state_hash(list(teacher.named_parameters())) == teacher_hash,
                            "teacher changed before last refresh")
                    require(all(p.grad is None and not p.requires_grad
                                for p in teacher.parameters()),
                            "teacher must remain frozen and gradient-free")
                    del teacher_qwen, teacher
                    torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            before = [p.detach().clone() for _, p in named]

            def loss_for_item(item):
                nonlocal active_model
                entry = next(x for x in materialized if x["item"] is item)
                case = item["case"]
                if item["kind"] == "online":
                    action_ids = online_state["action_ids"]
                    current_layout = online_state["layout"]
                    reference = online_state["teacher_logp"].to(device)
                else:
                    action_ids = case["action_ids"]
                    current_layout = case["initial_layout"]
                    reference = reference_cache[str(case["key"])].to(device)
                active_model = "train"
                checkpointing["phase"] = "train"
                loss, ul, kl = ddp(
                    entry["inputs"], entry["prompt_ids"], action_ids,
                    current_layout, reference, item["kind"],
                )
                active_model = "idle"
                checkpointing["phase"] = "idle"
                finite = bool(torch.isfinite(loss)) and bool(torch.isfinite(ul)) and \
                    bool(torch.isfinite(kl)) and float(kl) >= -1e-6
                require(finite, "finite per-image objective")
                if step == 1:
                    require(abs(float(kl)) <= 1e-6,
                            "initial KL only on the same Stable trajectory")
                details = dict(unlikelihood=float(ul), kl=float(kl),
                               states=len(current_layout["kl_positions"]),
                               key=str(case["key"]), image_id=str(case["image_id"]),
                               layout_sha256=digest(current_layout),
                               valid_rows=current_layout["valid_row_count"],
                               duplicate_rows=len(current_layout["duplicate_row_indices"]),
                               parser_drops=current_layout["parser_drops"],
                               invalid_geometry_rows=len(current_layout["invalid_geometry_rows"]))
                del reference
                return loss, details

            local_records = backward_rank_items(ddp, assigned, loss_for_item)
            counters["student_train_forwards"] += 0  # counted by the model hook
            counters["ddp_synchronized_backwards"] += 1
            finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                         for _, p in named)
            require(all_true(finite), "nonfinite or missing reduced gradients")
            require(all(p.grad is None and not p.requires_grad for _, p in frozen),
                    "frozen gradients")
            gradient_hash = _tensor_state_hash([(n, p.grad) for n, p in named])
            gradient_hashes = gather(gradient_hash)
            require(len(set(gradient_hashes)) == 1, "reduced gradient differs by rank")
            raw_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for _, p in named], 1.0, error_if_nonfinite=True, foreach=False,
            ))
            require(math.isfinite(raw_norm) and raw_norm >= 0.0,
                    "finite nonnegative reduced gradient")
            clipped_norm = math.sqrt(sum(float(p.grad.double().square().sum())
                                         for _, p in named))
            require(clipped_norm <= 1.000001, "global clip bound")
            optimizer.step()
            counters["updates"] = step
            require(all(p._version == version for p, version in frozen_versions),
                    "frozen version changed")
            adapter_hash = _tensor_state_hash(named)
            opt_hash = optimizer_hash(optimizer, named)
            state_hashes = gather((adapter_hash, opt_hash))
            require(len(set(state_hashes)) == 1,
                    "adapter/optimizer differs across ranks")
            require({int(state["step"]) for state in optimizer.state.values()} == {step},
                    "optimizer step identity")
            movement = math.sqrt(sum(float((p.detach() - old).double().square().sum())
                                     for (_, p), old in zip(named, before)))
            total_movement = math.sqrt(sum(float((p.detach() - old).double().square().sum())
                                           for (_, p), old in zip(named, original)))
            scalar_status = update_scalar_status(step, raw_norm, movement, total_movement)
            counters["zero_gradient_updates"] += int(scalar_status["zero_gradient"])
            counters["zero_movement_updates"] += int(scalar_status["zero_movement"])
            frozen_bytes_unchanged = True
            if step <= 2:
                frozen_bytes_unchanged = all_true(
                    _tensor_state_hash(frozen) == frozen_hash)
                require(frozen_bytes_unchanged,
                        "two-step frozen student bytes changed")
            per_rank = gather(dict(
                rank=rank, records=local_records, gradient_hash=gradient_hash,
                adapter_hash=adapter_hash, optimizer_hash=opt_hash,
                raw_gradient_norm=raw_norm, clipped_gradient_norm=clipped_norm,
                movement_l2=movement, total_movement_l2=total_movement,
                frozen_bytes_unchanged=frozen_bytes_unchanged,
                **scalar_status,
            ))
            row = dict(
                update=step, rank=rank, local_items=local_records,
                gradient_hash=gradient_hash, adapter_hash=adapter_hash,
                optimizer_hash=opt_hash, raw_gradient_norm=raw_norm,
                clipped_gradient_norm=clipped_norm,
                parameter_movement_l2=movement,
                source_parameter_delta_l2=total_movement,
                **scalar_status,
                counters=dict(counters), seconds=time.monotonic() - started,
            )
            publish(run / f"update-{step:02d}.json", row)
            if rank == 0:
                global_row = dict(update=step, ranks=per_rank)
                publish(training / f"update-{step:02d}.json", global_row)
                update_records.append(global_row)
                if step == 2:
                    layouts = [r for x in per_rank for r in x["records"]
                               if r["kind"] == "online"]
                    mask_counts = {r["image_id"]: r["duplicate_rows"] for r in layouts}
                    initial_kl = [r["kl"] for x in update_records[0]["ranks"]
                                  for r in x["records"]]
                    first_two_ranks = update_records[0]["ranks"] + per_rank
                    smoke = dict(
                        status="passed", updates=[1, 2],
                        masks_exercised=(len(layouts) == 8 and mask_counts["9813"] == 0 and
                                         all(mask_counts[image_id] > 0
                                             for image_id in ONLINE_IMAGE_IDS[1:])),
                        online_duplicate_rows=mask_counts,
                        initial_same_trajectory_KL_max_abs=max(abs(x) for x in initial_kl),
                        initial_same_trajectory_KL_zero=max(abs(x) for x in initial_kl) <= 1e-6,
                        finite_nonzero_gradients=all(
                            math.isfinite(x["raw_gradient_norm"]) and
                            x["raw_gradient_norm"] > 0 for x in first_two_ranks),
                        finite_nonzero_updates=all(
                            math.isfinite(x["movement_l2"]) and x["movement_l2"] > 0
                            for x in first_two_ranks),
                        reduced_gradient_rank_identity=len({x["gradient_hash"] for x in per_rank}) == 1,
                        adapter_optimizer_rank_identity=len({(x["adapter_hash"], x["optimizer_hash"]) for x in per_rank}) == 1,
                        frozen_bytes_unchanged=all(
                            x["frozen_bytes_unchanged"] for x in first_two_ranks),
                        longest_online_action_tokens=max(len(c["action_ids"]) for c in packet["online_cases"]),
                    )
                    require(smoke["longest_online_action_tokens"] == CAP,
                            "first two updates did not include capped trajectory")
                    publish(training / "two-step-smoke.json", smoke)
            if step == 2:
                barrier()
                gate = load_canonical_json(training / "two-step-smoke.json")
                require(gate["status"] == "passed" and gate["masks_exercised"] and
                        gate["initial_same_trajectory_KL_zero"] and
                        gate["finite_nonzero_gradients"] and
                        gate["finite_nonzero_updates"] and
                        gate["reduced_gradient_rank_identity"] and
                        gate["adapter_optimizer_rank_identity"] and
                        gate["frozen_bytes_unchanged"], "two-step mechanical gate")
            del before
            print(json.dumps(dict(rank=rank, update=step,
                                  seconds=time.monotonic() - started)), flush=True)

        require(all_true(_tensor_state_hash(frozen) == frozen_hash),
                "terminal frozen bytes changed")
        require(all(_tensor_state_hash([("teacher_logp", value)]) ==
                    reference_cache_hashes[key]
                    for key, value in reference_cache.items()),
                "static teacher cache changed")
        require(counters["updates"] == UPDATES and counters["refreshes"] == 4 and
                counters["continuations"] == 4 and
                counters["student_train_forwards"] == 256 and
                counters["parity_forwards"] ==
                    (PARITY_FORWARDS if rank == PARITY_RANK else 0) and
                counters["teacher_reference_forwards"] == 11 and
                counters["image_forwards"] == rank_image_forward_count(rank) and
                counters["model_forwards"] <= rank_model_forward_ceiling(rank),
                "rank execution counters")
        checkpoint_receipt = checkpointing_receipt(student, checkpointing)
        require(checkpoint_receipt["checkpoint_invocations"] ==
                    DECODER_LAYER_COUNT * (256 +
                        (PARITY_FORWARDS // 2 if rank == PARITY_RANK else 0)) and
                checkpoint_receipt["recompute_body_calls"] > 0 and
                checkpoint_receipt["bypass_no_grad_calls"] > 0,
                "activation-checkpoint execution counts")
        final_state = dict(
            adapter_hash=adapter_hash, optimizer_hash=opt_hash,
            frozen_hash=frozen_hash, source_adapter_hash=initial_adapter,
            static_reference_cache_bytes=initial_reference_bytes,
            final_online_reference_cache_bytes=online_state["teacher_logp"].numel() *
                online_state["teacher_logp"].element_size(),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            parameter_delta_l2=total_movement,
            activation_checkpointing=checkpoint_receipt,
        )
        ranks = gather(dict(rank=rank, counters=dict(counters), state=final_state,
                            refreshes=refresh_records))
        saved = None
        if rank == 0:
            try:
                adapter = _save_adapter_only(
                    student, source_root=Path(config.adapter.path), output=training / "adapter",
                )
                provisional = dict(
                    schema_version="geometric_dedup.training.v1",
                    status="unsealed_candidate", adapter=adapter,
                    source_adapter=packet["model"]["current_adapter"],
                    source_embedding=packet["model"]["source_embedding"],
                    config=packet["config"], updates=UPDATES,
                    refresh_steps=list(REFRESH_STEPS), stop_reason="fixed_steps",
                    optimizer=packet["optimizer"], clip_gradient_norm=1.0,
                    objective=packet["objective"], online_image_ids=list(ONLINE_IMAGE_IDS),
                    reference_image_ids=[str(c["image_id"]) for c in packet["reference_cases"]],
                    rank_training_states=ranks, trainable_layout=layout,
                    update_records=[dict(
                        update=step,
                        path=str(training / f"update-{step:02d}.json"),
                        sha256=file_hash(training / f"update-{step:02d}.json"),
                    ) for step in range(1, UPDATES + 1)],
                    inputs_sha256=file_hash(output / "inputs.json"),
                    code_identity_sha256=file_hash(training / "code_identity.json"),
                    checkpoint_parity=dict(
                        path=str(training / "checkpoint-parity.json"),
                        sha256=file_hash(training / "checkpoint-parity.json"),
                    ),
                    activation_checkpointing=(
                        "non_reentrant_language_blocks_grad_only_eval_mode"),
                    two_step=dict(path=str(training / "two-step-smoke.json"),
                                  sha256=file_hash(training / "two-step-smoke.json")),
                    numerical_policy=packet["numerical_policy"],
                )
                publish(training / "provisional.json", provisional)
                saved = dict(ok=True, fingerprint=adapter["fingerprint"])
            except BaseException as exc:
                saved = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
        saved = gather(saved)[0]
        require(saved["ok"], f"rank0 save failed: {saved.get('error')}")
        status = "completed"
      except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
      finally:
        signal.alarm(0)
        publish(run / "terminal.json", dict(
            rank=rank, status=status, error=error, pid=os.getpid(),
            updates=counters["updates"], counters=counters, state=final_state,
            artifact_bytes=sum(p.stat().st_size for p in run.rglob("*") if p.is_file()),
            cumulative_model_seconds=time.monotonic() - started,
        ))
        if dist.is_initialized():
            dist.destroy_process_group()


def finalize(output):
    training = output / "training"
    require(not (training / "receipt.json").exists(), "occupied sealed receipt")
    exit_path = training / "launcher_exit.json"
    require(load_canonical_json(exit_path)["exit_code"] == 0, "launcher failed")
    terminals, terminal_refs = [], []
    require({p.name for p in (training / "ranks").iterdir() if p.is_dir()} ==
            {f"rank{rank}" for rank in range(WORLD_SIZE)}, "eight rank coverage")
    for rank in range(WORLD_SIZE):
        path = training / "ranks" / f"rank{rank}" / "terminal.json"
        terminal = load_canonical_json(path)
        counters = terminal["counters"]
        require(terminal["status"] == "completed" and terminal["rank"] == rank and
                terminal["updates"] == UPDATES and counters["updates"] == UPDATES and
                counters["refreshes"] == 4 and counters["continuations"] == 4 and
                counters["student_train_forwards"] == 256 and
                counters["parity_forwards"] ==
                    (PARITY_FORWARDS if rank == PARITY_RANK else 0) and
                counters["teacher_reference_forwards"] == 11 and
                counters["image_forwards"] == rank_image_forward_count(rank) and
                counters["model_forwards"] <= rank_model_forward_ceiling(rank) and
                terminal["state"]["activation_checkpointing"]["model_eval"] and
                terminal["state"]["activation_checkpointing"]["checkpoint_invocations"] ==
                    DECODER_LAYER_COUNT * (256 +
                        (PARITY_FORWARDS // 2 if rank == PARITY_RANK else 0)),
                "complete rank counters")
        terminals.append(terminal)
        terminal_refs.append(dict(rank=rank, path=str(path), sha256=file_hash(path)))
    require(len({t["state"]["adapter_hash"] for t in terminals}) == 1 and
            len({t["state"]["optimizer_hash"] for t in terminals}) == 1,
            "terminal rank state identity")
    provisional = load_canonical_json(training / "provisional.json")
    require(provisional["status"] == "unsealed_candidate" and
            provisional["updates"] == UPDATES and
            provisional["online_image_ids"] == list(ONLINE_IMAGE_IDS) and
            len(provisional["reference_image_ids"]) == REFERENCE_COUNT and
            len(provisional["update_records"]) == UPDATES and
            all(record["update"] == index and
                file_hash(record["path"]) == record["sha256"]
                for index, record in enumerate(provisional["update_records"], 1)),
            "provisional recipe")
    smoke = load_canonical_json(Path(provisional["two_step"]["path"]))
    parity = load_canonical_json(Path(provisional["checkpoint_parity"]["path"]))
    require(file_hash(provisional["two_step"]["path"]) == provisional["two_step"]["sha256"] and
            smoke["status"] == "passed" and smoke["updates"] == [1, 2] and
            smoke["masks_exercised"] and smoke["initial_same_trajectory_KL_zero"] and
            smoke["finite_nonzero_gradients"] and smoke["finite_nonzero_updates"] and
            smoke["reduced_gradient_rank_identity"] and
            smoke["adapter_optimizer_rank_identity"] and
            smoke["frozen_bytes_unchanged"], "two-step mechanical evidence")
    require(file_hash(provisional["checkpoint_parity"]["path"]) ==
                provisional["checkpoint_parity"]["sha256"] and
            parity["status"] == "passed" and parity["rank"] == PARITY_RANK and
            parity["image_id"] == "158044" and parity["action_tokens"] == 246 and
            parity["duplicate_rows"] == 5 and parity["metrics"]["passed"] and
            parity["no_optimizer_step"], "real checkpoint parity evidence")
    for identity in (provisional["adapter"], provisional["source_embedding"]):
        for entry in identity["files"]:
            require(file_hash(Path(identity["root"]) / entry["relative_path"]) ==
                    entry["sha256"], "checkpoint bytes")
    resources = dict(
        model_loads=sum(t["counters"]["model_loads"] for t in terminals),
        model_forwards=sum(t["counters"]["model_forwards"] for t in terminals),
        image_forwards=sum(t["counters"]["image_forwards"] for t in terminals),
        continuations=sum(t["counters"]["continuations"] for t in terminals),
        generated_tokens=sum(t["counters"]["generated_tokens"] for t in terminals),
        teacher_reference_forwards=sum(t["counters"]["teacher_reference_forwards"]
                                       for t in terminals),
        parity_forwards=sum(t["counters"]["parity_forwards"] for t in terminals),
        train_forwards=sum(t["counters"]["student_train_forwards"] for t in terminals),
        cumulative_rank_model_seconds=sum(t["cumulative_model_seconds"] for t in terminals),
        max_rank_model_seconds=max(t["cumulative_model_seconds"] for t in terminals),
        static_reference_cache_bytes=sum(t["state"]["static_reference_cache_bytes"]
                                         for t in terminals),
        final_online_reference_cache_bytes=sum(
            t["state"]["final_online_reference_cache_bytes"] for t in terminals),
        peak_cuda_allocated_bytes_max_rank=max(t["state"]["peak_cuda_allocated_bytes"]
                                               for t in terminals),
        peak_cuda_reserved_bytes_max_rank=max(t["state"]["peak_cuda_reserved_bytes"]
                                              for t in terminals),
        peak_rss_bytes_max_rank=max(t["state"]["peak_rss_bytes"] for t in terminals),
        rank_artifact_bytes=sum(t["artifact_bytes"] for t in terminals),
    )
    receipt = {
        **provisional, "status": "completed", "scientific_status": "candidate",
        "rank_terminals": terminal_refs,
        "launcher_exit": dict(path=str(exit_path), sha256=file_hash(exit_path)),
        "provisional_sha256": file_hash(training / "provisional.json"),
        "resources": resources,
    }
    publish(training / "receipt.json", receipt)
    return receipt


def run(output):
    validate_packet(load_canonical_json(output / "inputs.json"))
    training = output / "training"
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3,4,5,6,7" and
            not training.exists(), "single eight-GPU launch")
    training.mkdir(parents=False, exist_ok=False)
    code_paths = [
        Path(__file__), Path(__file__).with_name("geometric_dedup.py"),
        Path(__file__).with_name("selective_preservation_stable.py"),
        Path(__file__).with_name("runtime.py"), Path(__file__).with_name("route_access.py"),
        Path(__file__).parents[2] / "tests" / "test_geometric_dedup_train.py",
    ]
    publish(training / "code_identity.json", dict(
        files=[dict(path=str(p.resolve()), sha256=file_hash(p)) for p in code_paths],
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
    ))
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc-per-node=8", "-m", "probes.dora_owner_learning.geometric_dedup_train",
        "rank", "--output", str(output),
    ]
    publish(training / "launcher_owner.json", dict(
        pid=os.getpid(), command=command, started=time.time(),
        processes=subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
             "--format=csv,noheader"], text=True,
        ),
    ))
    started = time.monotonic()
    with (training / "torchrun.log").open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    publish(training / "launcher_exit.json", dict(
        exit_code=result.returncode, elapsed_seconds=time.monotonic() - started,
        finished=time.time(),
    ))
    if result.returncode:
        return result.returncode
    receipt = finalize(output)
    print(json.dumps(dict(status=receipt["status"], resources=receipt["resources"])))
    return 0


def verify(output):
    training = output / "training"
    receipt = load_canonical_json(training / "receipt.json")
    require(receipt["status"] == "completed" and
            receipt["scientific_status"] == "candidate" and
            receipt["updates"] == UPDATES, "sealed candidate receipt")
    require(file_hash(output / "inputs.json") == receipt["inputs_sha256"] and
            file_hash(training / "code_identity.json") == receipt["code_identity_sha256"] and
            file_hash(training / "provisional.json") == receipt["provisional_sha256"],
            "sealed primary artifacts")
    for ref in receipt["rank_terminals"] + [
            receipt["launcher_exit"], receipt["two_step"], receipt["checkpoint_parity"]]:
        require(file_hash(ref["path"]) == ref["sha256"], "sealed proof bytes")
    require(len(receipt["update_records"]) == UPDATES and
            all(record["update"] == index and
                file_hash(record["path"]) == record["sha256"]
                for index, record in enumerate(receipt["update_records"], 1)),
            "sealed update records")
    for identity in (receipt["adapter"], receipt["source_embedding"]):
        for entry in identity["files"]:
            require(file_hash(Path(identity["root"]) / entry["relative_path"]) ==
                    entry["sha256"], "sealed model bytes")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("rank", "run", "finalize", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "rank":
        execute_rank(args.output)
    elif args.command == "run":
        raise SystemExit(run(args.output))
    elif args.command == "finalize":
        print(json.dumps(finalize(args.output), indent=2))
    else:
        receipt = verify(args.output)
        print(json.dumps(dict(status=receipt["status"], updates=receipt["updates"],
                              resources=receipt["resources"]), indent=2))


if __name__ == "__main__":
    main()
