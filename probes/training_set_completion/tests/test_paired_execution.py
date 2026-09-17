"""Run the real paired loop with four CPU ranks and a tiny substituted model.

This tests control flow, objective dispatch, collective scaling and receipts.
It deliberately does not claim Qwen loading, native image or GPU parity.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from probes.training_set_completion import source256_training as source
from probes.training_set_completion import source256_normalized_training as normalized
from probes.training_set_completion import replay, training


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.3))


def records():
    result = []
    for image in range(256):
        routes = []
        for complete in (False, True):
            prefix = [30, 31] if complete else []
            suffix = [40, 41, 42, 43, 99] if complete else list(range(40, 49)) + [99]
            routes.append(
                {
                    "route_id": f"{image}:{complete}",
                    "image_id": image,
                    "continuation_token_ids": prefix + suffix,
                    "ce_weights": [0] * len(prefix) + [1] * len(suffix),
                    "trusted_boxes": [],
                    "provenance": {
                        "route_kind": "fixed_source_prefix_completion"
                        if complete
                        else "canonical",
                        "prefix_token_ids": prefix,
                        "suffix_token_ids": suffix,
                        "prefix_owner_ids": ["old"] if complete else [],
                        "suffix_owner_ids": ["new"] if complete else ["old", "new"],
                    },
                }
            )
        result.append(
            {
                "image_id": image,
                "canonical_route": routes[0],
                "completion_route": routes[1],
                "eligibility": {"fully_eligible": True, "fallback_reason": None},
            }
        )
    return result


def schedule():
    return {
        "updates": [
            {
                "step": step,
                "common_image_ids": list(range(32)),
                "variable_image_ids": list(range(64, 96)),
            }
            for step in (1, 2)
        ]
    }


def logits_for(model, routes):
    vocabulary = torch.arange(128).float().view(1, -1) / 128
    return [
        model.weight * vocabulary.expand(len(route["continuation_token_ids"]), -1)
        for route in routes
    ]


def _worker(rank, init_path, manifest_path, output, variant):
    from probes.dora_owner_learning import runtime
    from src.config import inference
    from src.qwen import checkpointing

    class CpuTorch:
        cuda = SimpleNamespace(
            is_available=lambda: True,
            set_device=lambda *_: None,
            manual_seed=lambda *_: None,
            reset_peak_memory_stats=lambda *_: None,
        )

        def __getattr__(self, name):
            return getattr(torch, name)

        def device(self, *args, **kwargs):
            return torch.device("cpu")

    class Gloo:
        def __getattr__(self, name):
            return getattr(dist, name)

        def init_process_group(self, backend, *, timeout):
            assert backend == "nccl"  # The production recipe still selects NCCL.
            dist.init_process_group(
                "gloo",
                init_method=f"file://{init_path}",
                rank=rank,
                world_size=4,
                timeout=timeout,
            )

    manifest = json.loads(Path(manifest_path).read_text())
    producer = Path(normalized.__file__ if variant == "normalized" else source.__file__)
    prepared = {"records": records(), "schedule": schedule()}
    model = ScalarModel()
    counted = {"terms": 0}
    terms = normalized._route_terms if variant == "normalized" else replay.route_terms
    resolver = (
        normalized.resolve_update_presentations
        if variant == "normalized"
        else source.resolve_update_presentations
    )

    def route_terms(*args):
        counted["terms"] += 1
        return terms(*args)

    def checkpoint(output, **kwargs):
        directory = output / "checkpoints" / f"step-{kwargs['step']:05d}"
        directory.mkdir(parents=True)
        training.publish(
            directory / "cpu-state.json",
            {
                "weight": float(model.weight.detach()),
                "optimizer_steps": [
                    int(state["step"]) for state in kwargs["optimizer"].state.values()
                ],
            },
        )
        return {
            "step": kwargs["step"],
            "cpu_state": training.binding(directory / "cpu-state.json"),
        }

    with pytest.MonkeyPatch.context() as patch:
        for key, value in {"RANK": rank, "LOCAL_RANK": rank, "WORLD_SIZE": 4}.items():
            patch.setenv(key, str(value))
        patch.setattr(source, "torch", CpuTorch())
        patch.setattr(source, "dist", Gloo())
        patch.setattr(source, "validate_preparation", lambda value: prepared)
        patch.setattr(source, "hydrate_bound_cases", lambda value: value["records"])
        patch.setattr(source, "source_adapter_scalar_count", lambda value: 1)
        patch.setattr(inference.InferConfig, "model_validate", lambda value: value)
        patch.setattr(
            runtime,
            "load_policy",
            lambda *args, **kwargs: (
                SimpleNamespace(model=model, tokenizer=SimpleNamespace(pad_token_id=0)),
                {"cpu_fixture": True},
            ),
        )
        patch.setattr(
            runtime,
            "bind_source256_language_dora",
            lambda *args, **kwargs: (tuple(model.named_parameters()), ()),
        )
        patch.setattr(
            checkpointing,
            "install_language_decoder_checkpointing",
            lambda *args, **kwargs: {},
        )
        patch.setattr(
            checkpointing,
            "language_decoder_checkpointing_receipt",
            lambda *args: {"cpu_fixture": True},
        )
        patch.setattr(
            replay,
            "prepare_microbatches",
            lambda qwen, manifest, routes, **kwargs: (
                [{"routes": routes, "inputs": {}}],
                {"prompt_padding_tokens": 0},
            ),
        )
        patch.setattr(
            replay,
            "batched_aligned_logits",
            lambda model, inputs, routes, **kwargs: (
                logits_for(model, routes),
                {"history_padding_tokens": 0},
            ),
        )
        patch.setattr(training, "_checkpoint", checkpoint)
        source.run_paired_training(
            Path(manifest_path),
            output=Path(output),
            producer_path=producer,
            receipt_schema=normalized.SCHEMA
            if variant == "normalized"
            else source.SCHEMA,
            validate_manifest=lambda value: manifest,
            resolve_presentations=resolver,
            route_terms=route_terms,
            dependency_bindings={"cpu_fixture": {"variant": variant}},
            enrich_update=normalized._enrich_update
            if variant == "normalized"
            else None,
        )
    training.publish(Path(output) / f"calls-{rank}.json", counted)


@pytest.mark.parametrize("variant", ["original", "normalized"])
def test_real_four_rank_loop_matches_serial_update_and_persists_variant_evidence(
    tmp_path, variant
):
    if not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    producer = Path(normalized.__file__ if variant == "normalized" else source.__file__)
    preparation = tmp_path / "preparation.json"
    preparation.write_text("{}")
    manifest = {
        "arm": normalized.B_NORMALIZED if variant == "normalized" else "B",
        "mode": "qualification",
        "sources": {"producer": training.binding(producer)},
        "preparation": training.binding(preparation),
        "model_config": {},
        "source_adapter": {"semantic_identity": {"tensor_key_count": 1}},
        "optimizer": training.DEFAULT_OPTIMIZER,
        "scheduler": {"total_updates": 64, "min_lr_ratio": 0.0},
        "validity_hinge": {
            "coordinate_token_ids": list(range(128)),
            "coordinate_bin_values": list(range(128)),
            "weight": 0.01,
            "margin": 1 / 999,
        },
        "runtime": {
            "wall_seconds": 90,
            "updates": 2,
            "microbatch_size": 2,
            "gradient_clip_norm": 1.0,
            "eos_token_id": 99,
            "checkpoint_steps": [2],
            "max_model_forwards": 128,
            "max_model_calls": 64,
        },
    }
    path = tmp_path / "manifest.json"
    training.publish(path, manifest)
    output = tmp_path / "execution"
    mp.spawn(
        _worker,
        args=(str(tmp_path / "gloo"), str(path), str(output), variant),
        nprocs=4,
        join=True,
    )

    serial = ScalarModel()
    optimizer = torch.optim.AdamW(
        serial.parameters(),
        **{
            **training.DEFAULT_OPTIMIZER,
            "betas": tuple(training.DEFAULT_OPTIMIZER["betas"]),
        },
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=64, eta_min=0
    )
    resolver = (
        normalized.resolve_update_presentations
        if variant == "normalized"
        else source.resolve_update_presentations
    )
    terms = normalized._route_terms if variant == "normalized" else replay.route_terms
    for update in schedule()["updates"]:
        presentations = resolver(records(), update, arm=manifest["arm"])
        routes = [item["route"] for item in presentations]
        values = [
            terms(logits, route, manifest["validity_hinge"])
            for logits, route in zip(logits_for(serial, routes), routes, strict=True)
        ]
        optimizer.zero_grad(set_to_none=True)
        source.objective_from_presentation_terms(
            [item[0] for item in values],
            [item[1] for item in values],
            [item["branch"] for item in presentations],
        ).backward()
        torch.nn.utils.clip_grad_norm_(serial.parameters(), 1.0, foreach=False)
        optimizer.step()
        scheduler.step()
    state = json.loads((output / "checkpoints/step-00002/cpu-state.json").read_text())
    assert state["weight"] == pytest.approx(float(serial.weight.detach()), abs=1e-7)
    assert state["optimizer_steps"] == [2]
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "completed" and terminal["model_calls"] == 64
    assert terminal["distributed"]["backend"] == "gloo"
    assert terminal["distributed"]["source_bindings"][
        "training_backend"
    ] == training.binding(producer)
    assert len(terminal["distributed"]["checkpoint_consensus"]) == 1
    for rank in range(4):
        assert json.loads((output / f"calls-{rank}.json").read_text()) == {"terms": 32}
    update = json.loads((output / "updates/step-00002.json").read_text())
    assert ("ce_normalization" in update) == (variant == "normalized")
    assert len(update["presentations"]) == 64
    if variant == "normalized":
        scales = update["ce_normalization"]["branches"]
        assert scales["common"]["scale_distribution"]["values"] == [1.0] * 32
        assert scales["variable"]["scale_distribution"]["values"] == [0.5] * 32
