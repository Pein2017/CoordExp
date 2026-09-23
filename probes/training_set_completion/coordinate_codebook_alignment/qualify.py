"""Bounded qualification wrapper for the maintained training entrypoint.

This module instruments ``src.train`` through its existing pipeline and trainer
objects.  It does not implement a training loop or replace the loss/trainer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from unittest.mock import patch

import torch
from torch.nn import functional as F

from src.losses.context import LossContext
from src.training.supervised_trainer import LossRunnerBoundary, SupervisedTrainer


REFERENCE_TOLERANCE = 2e-4


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != data:
            raise FileExistsError(f"qualification artifact collision: {path}")
        return
    path.write_text(data)


def _tensor_hash(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _parameter_hashes(model: Any) -> dict[str, str]:
    model = _unwrap_model(model)
    return {
        name: _tensor_hash(parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def _embedding_delta_ids(model: Any) -> tuple[set[int], set[int]]:
    model = _unwrap_model(model)
    input_delta = getattr(model.get_input_embeddings(), "shared_embed_delta", None)
    output_delta = getattr(model.get_output_embeddings(), "shared_embed_delta", None)
    return (
        {id(input_delta)} if isinstance(input_delta, torch.Tensor) else set(),
        {id(output_delta)} if isinstance(output_delta, torch.Tensor) else set(),
    )


def _parameter_gradient_summary(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        result[name] = {
            "present": gradient is not None,
            "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "norm": None if gradient is None else float(gradient.detach().float().norm()),
            "nonzero": bool(gradient is not None and bool(gradient.detach().abs().max() > 0)),
        }
    return result


def _bare_parameter_name(name: str) -> str:
    return name.removeprefix("module.")


def _parameter_category(
    name: str,
    parameter: torch.Tensor,
    *,
    input_ids: set[int],
    output_ids: set[int],
) -> str | None:
    bare = _bare_parameter_name(name)
    if id(parameter) in input_ids:
        return "input"
    if id(parameter) in output_ids:
        return "output"
    if bare.endswith("coordinate_codebook.raw_gain") or bare.endswith("coordinate_codebook.projection.weight"):
        return "codebook"
    if ".visual.merger" in bare or bare.startswith("visual.merger") or "deepstack_merger" in bare:
        return "aligner"
    if "visual." in bare or bare.startswith("visual."):
        return "vision"
    if "language_model." in bare or bare.startswith("language_model."):
        return "language"
    return None


def _group_coverage(runtime: Any) -> dict[str, Any]:
    model_names = dict(runtime.model.named_parameters())
    parameter_to_name = {id(parameter): name for name, parameter in model_names.items()}
    groups = []
    seen: list[str] = []
    for index, group in enumerate(runtime.optimizer.param_groups):
        actual_names = tuple(
            parameter_to_name[id(parameter)]
            for parameter in group["params"]
            if id(parameter) in parameter_to_name
        )
        configured_names = tuple(str(name) for name in group.get("parameter_names", ()))
        groups.append(
            {
                "index": index,
                "name": group.get("name"),
                "parameter_names": list(configured_names),
                "actual_parameter_names": list(actual_names),
            }
        )
        seen.extend(actual_names)
    trainable = {name for name, parameter in model_names.items() if parameter.requires_grad}
    return {
        "groups": groups,
        "trainable_parameter_count": len(trainable),
        "optimizer_parameter_count": len(seen),
        "exact_name_coverage": sorted(trainable) == sorted(seen)
        and len(seen) == len(set(seen)),
        "missing": sorted(trainable - set(seen)),
        "extra": sorted(set(seen) - trainable),
    }


class _LossQualificationProbe:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.first_loss: dict[str, Any] | None = None
        self.first_loss_micro_steps: list[dict[str, Any]] = []
        self.planned_micro_step_count: int | None = None
        self.post_backward: dict[str, Any] | None = None
        self.post_backward_updates: list[dict[str, Any]] = []
        self.pre_update: dict[str, str] | None = None
        self.pre_update_values: dict[str, torch.Tensor] | None = None
        self.update_deltas: dict[str, float] | None = None
        self.update_deltas_by_step: list[dict[str, float]] = []
        self.runtime: Any | None = None
        self.pipeline_result: Mapping[str, Any] | None = None
        self.model_forward_count = 0
        self.vision_forward_count = 0
        self.started = time.perf_counter()
        self.first_forward_memory: int | None = None
        self.merger_activation: torch.Tensor | None = None
        self.merger_grid: torch.Tensor | None = None
        self.address_checks: dict[str, Any] = {}

    def capture_forward_hooks(self, model: Any) -> list[Any]:
        from src.qwen.coordinate_codebook import _visual_module
        handles = []
        handles.append(model.register_forward_pre_hook(lambda *_: self._model_forward()))
        model = _unwrap_model(model)
        visual = _visual_module(model)
        if visual is not None and hasattr(visual, "register_forward_hook"):
            handles.append(visual.register_forward_pre_hook(lambda *_: self._vision_forward()))
            merger = getattr(visual, "merger", None)
            if merger is not None:
                handles.append(merger.register_forward_hook(self._capture_merger, prepend=True))
        return handles

    def _capture_merger(self, _module: Any, _args: Any, output: Any) -> None:
        if self.merger_activation is not None or not isinstance(output, torch.Tensor):
            return
        self.merger_activation = output.detach()
        codebook = (
            getattr(_unwrap_model(self.runtime.model), "coordinate_codebook", None)
            if self.runtime is not None else None
        )
        grid = getattr(codebook, "_grid", None)
        self.merger_grid = None if grid is None else grid.detach().clone()

    def _model_forward(self) -> None:
        self.model_forward_count += 1
        self._memory()

    def _vision_forward(self) -> None:
        self.vision_forward_count += 1

    def _memory(self) -> None:
        if torch.cuda.is_available():
            value = int(torch.cuda.max_memory_allocated())
            self.first_forward_memory = value if self.first_forward_memory is None else max(self.first_forward_memory, value)

    def capture_loss(self, context: Any, bundle: Any, plan: Any, *, local_index: int) -> None:
        if (
            not isinstance(context, LossContext)
            or self.planned_micro_step_count is not None
            and len(self.first_loss_micro_steps) >= self.planned_micro_step_count
        ):
            return
        logits, targets, atoms = context.select_logits_fp32()
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        segment_values: dict[int, list[torch.Tensor]] = {}
        for value, atom in zip(cross_entropy, atoms, strict=True):
            segment_values.setdefault(int(atom.segment_index), []).append(value)
        segment_means = torch.stack([torch.stack(values).mean() for values in segment_values.values()])
        base_term = bundle.term_by_name("base_ce")
        denominator = plan.denominators["base_ce"]
        reference = segment_means.sum() / float(denominator.eligible_segment_count)
        reference = reference * float(plan.backend_gradient_scale)
        wrong_token_mean = cross_entropy.mean()
        reference_gradient = torch.autograd.grad(reference, context.logits, retain_graph=True, allow_unused=True)[0]
        actual_gradient = torch.autograd.grad(base_term.weighted_loss, context.logits, retain_graph=True, allow_unused=True)[0]
        gradient_delta = None
        if reference_gradient is not None and actual_gradient is not None:
            gradient_delta = float((reference_gradient - actual_gradient.float()).abs().max())
        counts = sorted(len(values) for values in segment_values.values())
        unequal = len(set(counts)) > 1
        wrong_delta = float((reference - wrong_token_mean).abs().detach())
        # Unequal segment sizes make the two reducers structurally different even
        # when this particular batch happens to have equal per-token losses.
        wrong_formula_structurally_distinct = unequal
        record = {
            "local_micro_step_index": int(local_index),
            "logit_shape": [int(value) for value in context.logits.shape],
            "vocab_size": int(context.logits.shape[-1]),
            "target_count": len(atoms),
            "segment_count": len(segment_values),
            "segment_target_counts": counts,
            "segment_balanced_reference": float(reference.detach()),
            "maintained_base_ce_weighted_loss": float(base_term.weighted_loss.detach()),
            "maintained_base_ce_raw_loss": float(base_term.raw_loss.detach()),
            "reference_loss_delta": float((reference - base_term.weighted_loss.float()).abs().detach()),
            "reference_gradient_max_delta": gradient_delta,
            "wrong_token_mean": float(wrong_token_mean.detach()),
            "wrong_token_mean_delta": wrong_delta,
            "wrong_token_mean_rejected": (
                True if wrong_formula_structurally_distinct else "not_applicable_equal_segment_counts"
            ),
            "denominator": denominator.to_artifact_dict(),
            "backend_gradient_scale": float(plan.backend_gradient_scale),
            "plan_counts": dict(plan.counts),
        }
        self.first_loss_micro_steps.append(record)
        if self.first_loss is None:
            self.first_loss = record
            self._address_checks(context.logits.device)

    def _address_checks(self, device: torch.device) -> None:
        if self.runtime is None:
            self.address_checks = {"status": "unavailable"}
            return
        model = _unwrap_model(self.runtime.model)
        codebook = getattr(model, "coordinate_codebook", None)
        if codebook is None or self.merger_activation is None or self.merger_grid is None:
            self.address_checks = {"status": "unavailable"}
            return
        try:
            live = codebook.inject(self.merger_activation, self.merger_grid.to(device))
            _, output_ids = _embedding_delta_ids(model)
            output_parameter = next(parameter for parameter in model.parameters() if id(parameter) in output_ids)
            objective_weights = torch.arange(
                live.numel(), device=live.device, dtype=torch.float32
            ).reshape_as(live)
            live_objective = (live.float() * (1.0 + objective_weights / max(1, live.numel()))).sum()
            live_gradient = torch.autograd.grad(live_objective, output_parameter, retain_graph=True, allow_unused=True)[0]
            wrong_grid = self.merger_grid.clone()
            wrong_grid[:, 1:] = wrong_grid[:, [2, 1]]
            if torch.equal(wrong_grid, self.merger_grid):
                # Equal H/W would make an axis swap numerically invisible; make
                # the boundary itself invalid so the maintained geometry check
                # must reject the mutation.
                wrong_grid[0, 2] += 2
            try:
                wrong = codebook.inject(self.merger_activation, wrong_grid.to(device))
                wrong_grid_detected = not torch.equal(live.detach(), wrong.detach())
            except (TypeError, ValueError, RuntimeError):
                wrong_grid_detected = True
            rows, hidden = codebook._effective_rows()
            with patch.object(codebook, "_effective_rows", return_value=(rows.detach(), hidden)):
                stale = codebook.inject(self.merger_activation, self.merger_grid.to(device))
                stale_objective = (stale.float() * (1.0 + objective_weights / max(1, live.numel()))).sum()
                stale_gradient = torch.autograd.grad(stale_objective, output_parameter, retain_graph=True, allow_unused=True)[0]
            self.address_checks = {
                "status": "checked",
                "live_output_delta_gradient_nonzero": bool(live_gradient is not None and bool(live_gradient.abs().max() > 0)),
                "stale_detached_output_delta_gradient_none": stale_gradient is None,
                "wrong_grid_detected": wrong_grid_detected,
            }
        except (TypeError, ValueError, RuntimeError, StopIteration) as exc:
            self.address_checks = {"status": "failed", "error": repr(exc)}

    def capture_post_backward(self, runtime: Any) -> None:
        gradients = _parameter_gradient_summary(runtime.model)
        parameter_map = dict(runtime.model.named_parameters())
        input_ids, output_ids = _embedding_delta_ids(runtime.model)
        codebook = {
            name: value for name, value in gradients.items()
            if name.endswith("coordinate_codebook.raw_gain")
        }
        input_delta = {
            name: value for name, value in gradients.items()
            if id(parameter_map[name]) in input_ids
        }
        output_delta = {
            name: value for name, value in gradients.items()
            if id(parameter_map[name]) in output_ids
        }
        vision = {name: value for name, value in gradients.items() if "visual" in name}
        category_gradients = {category: {} for category in ("language", "vision", "aligner", "input", "output", "codebook")}
        for name, value in gradients.items():
            category = _parameter_category(
                name,
                parameter_map[name],
                input_ids=input_ids,
                output_ids=output_ids,
            )
            if category is not None:
                category_gradients[category][name] = value
        self.post_backward = {
            "all": gradients,
            "coordinate_codebook_raw_gain": codebook,
            "input_delta": input_delta,
            "output_delta": output_delta,
            "vision": vision,
            "category_gradients": category_gradients,
            "gradient_checkpointing": bool(getattr(_unwrap_model(runtime.model), "is_gradient_checkpointing", False)),
            "checkpointed_module_names": [name for name, module in _unwrap_model(runtime.model).named_modules()
                                          if getattr(module, "gradient_checkpointing", False)],
            "live_vision_gradient_nonzero": any(item["nonzero"] for item in vision.values()),
        }
        self.post_backward_updates.append(self.post_backward)

    def capture_update(self, runtime: Any) -> None:
        self.capture_post_backward(runtime)
        self.pre_update = {
            name: _tensor_hash(parameter)
            for name, parameter in runtime.model.named_parameters()
            if parameter.requires_grad
        }
        self.pre_update_values = {
            name: parameter.detach().cpu().clone()
            for name, parameter in runtime.model.named_parameters()
            if parameter.requires_grad
        }

    def finalize_update(self, runtime: Any) -> None:
        if self.pre_update is None:
            return
        deltas = {}
        for name, parameter in runtime.model.named_parameters():
            if (
                not parameter.requires_grad
                or name not in self.pre_update
                or self.pre_update_values is None
            ):
                continue
            delta = parameter.detach().cpu().float() - self.pre_update_values[name].float()
            deltas[name] = float(delta.abs().max())
        self.update_deltas = deltas
        self.update_deltas_by_step.append(deltas)

    def validate(self) -> None:
        all_counts = [count for item in self.first_loss_micro_steps for count in item['segment_target_counts']]
        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, all_counts)
            all_counts = [count for counts in gathered for count in counts]
        self.global_segment_target_counts = all_counts
        if not self.post_backward or not self.post_backward['gradient_checkpointing']:
            raise AssertionError('actual training model did not enable gradient checkpointing')
        if self.planned_micro_step_count is None:
            raise AssertionError("qualification did not observe a planned-step size")
        if len(self.first_loss_micro_steps) != self.planned_micro_step_count:
            raise AssertionError("qualification did not observe every first planned-step micro-step")
        if any(
            item["reference_loss_delta"] > REFERENCE_TOLERANCE
            or item["reference_gradient_max_delta"] is None
            or item["reference_gradient_max_delta"] > REFERENCE_TOLERANCE
            for item in self.first_loss_micro_steps
        ):
            raise AssertionError("maintained segment-balanced loss or gradient comparison exceeded tolerance")
        if len(set(all_counts)) <= 1:
            raise AssertionError("qualification first planned batch did not contain unequal segment lengths")
        if any(
            len(set(item["segment_target_counts"])) > 1
            and item["wrong_token_mean_rejected"] is not True
            for item in self.first_loss_micro_steps
        ):
            raise AssertionError("deliberately wrong token-mean reducer was not rejected")
        if self.runtime is None or not _group_coverage(self.runtime)["exact_name_coverage"]:
            raise AssertionError("optimizer coverage is not exact")
        if _parameter_hashes(self.runtime.model) != getattr(self, "frozen_before", {}):
            raise AssertionError("frozen parameters changed during qualification")
        if len(self.post_backward_updates) < 2 or len(self.update_deltas_by_step) < 2:
            raise AssertionError("qualification requires two observed optimizer updates")
        if any(
            not value["finite"]
            for update in self.post_backward_updates[:2]
            for value in update["all"].values()
        ):
            raise AssertionError("qualification observed a non-finite trainable gradient")
        if any(
            not torch.isfinite(torch.tensor(value))
            for update in self.update_deltas_by_step[:2]
            for value in update.values()
        ):
            raise AssertionError("qualification observed a non-finite parameter update")
        gradient_categories = {
            "language": False,
            "vision": False,
            "aligner": False,
            "input": False,
            "output": False,
            "codebook": False,
        }
        update_categories = dict(gradient_categories)
        input_ids, output_ids = _embedding_delta_ids(self.runtime.model)
        parameter_map = dict(self.runtime.model.named_parameters())
        for gradients, deltas in zip(self.post_backward_updates[:2], self.update_deltas_by_step[:2], strict=True):
            for name, value in gradients["all"].items():
                if not value["finite"] or not value["nonzero"]:
                    continue
                category = _parameter_category(
                    name,
                    parameter_map[name],
                    input_ids=input_ids,
                    output_ids=output_ids,
                )
                if category is not None:
                    gradient_categories[category] = True
            for name, value in deltas.items():
                if value <= 0 or name not in parameter_map:
                    continue
                category = _parameter_category(
                    name,
                    parameter_map[name],
                    input_ids=input_ids,
                    output_ids=output_ids,
                )
                if category is not None:
                    update_categories[category] = True
        if not all(gradient_categories.values()) or not all(update_categories.values()):
            raise AssertionError(
                "intended trainable categories lacked gradient/update evidence: "
                f"gradients={gradient_categories}, updates={update_categories}"
            )
        if self.address_checks.get("status") != "checked" or not all(
            self.address_checks.get(key, False)
            for key in (
                "live_output_delta_gradient_nonzero",
                "stale_detached_output_delta_gradient_none",
                "wrong_grid_detected",
            )
        ):
            raise AssertionError("live coordinate-codebook mutation checks did not pass")

    def artifact(self, runtime: Any, result: Mapping[str, Any] | None, error: BaseException | None) -> dict[str, Any]:
        frozen_before = getattr(self, "frozen_before", {})
        frozen_after = _parameter_hashes(runtime.model) if runtime is not None else {}
        return {
            "status": "failed" if error is not None else "candidate",
            "error": None if error is None else repr(error),
            "elapsed_seconds": time.perf_counter() - self.started,
            "model_forward_count": self.model_forward_count,
            "vision_forward_count": self.vision_forward_count,
            "max_cuda_memory_allocated_bytes": self.first_forward_memory,
            "first_loss": self.first_loss,
            "first_loss_micro_steps": self.first_loss_micro_steps,
            "global_segment_target_counts": getattr(self, 'global_segment_target_counts', None),
            "post_backward": self.post_backward,
            "post_backward_updates": self.post_backward_updates,
            "post_update_parameter_max_abs_delta": self.update_deltas,
            "post_update_parameter_max_abs_delta_by_step": self.update_deltas_by_step,
            "address_checks": self.address_checks,
            "frozen_parameter_hashes_before": frozen_before,
            "frozen_parameter_hashes_after": frozen_after,
            "trainable_parameter_hashes_after": {} if runtime is None else {
                name: _tensor_hash(parameter) for name, parameter in _unwrap_model(runtime.model).named_parameters()
                if parameter.requires_grad
            },
            "frozen_parameters_unchanged": frozen_before == frozen_after,
            "device_dora_initialization": getattr(runtime, 'dora_initialization_receipt', None),
            "optimizer_coverage": None if runtime is None else _group_coverage(runtime),
            "pipeline_result": None if result is None else dict(result),
            "rank": int(os.environ.get("RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "artifact_path": str(self.output),
        }


class _LossRunnerProbe:
    def __init__(self, inner: LossRunnerBoundary, probe: _LossQualificationProbe) -> None:
        self._inner = inner
        self._probe = probe

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def compute_micro_step(self, context: Any, plan: Any, *, local_micro_step_index: int) -> Any:
        bundle = self._inner.compute_micro_step(
            context, plan, local_micro_step_index=local_micro_step_index
        )
        self._probe.capture_loss(context, bundle, plan, local_index=local_micro_step_index)
        return bundle


class _InstrumentedTrainer(SupervisedTrainer):
    def __init__(self, *, qualification_probe: _LossQualificationProbe, **kwargs: Any) -> None:
        self.qualification_probe = qualification_probe
        runtime = kwargs["runtime"]
        qualification_probe.runtime = runtime
        schedule = kwargs["schedule"]
        qualification_probe.planned_micro_step_count = int(
            schedule.runtime_batch.resolved_grad_accum_steps
        )
        kwargs["loss_runner"] = _LossRunnerProbe(kwargs["loss_runner"], qualification_probe)
        self._hook_handles = qualification_probe.capture_forward_hooks(kwargs["model"])
        qualification_probe.frozen_before = _parameter_hashes(kwargs["model"])
        original_optimizer_step = runtime.optimizer_step

        def optimizer_step(*, planned_step_id: int) -> None:
            qualification_probe.capture_update(runtime)
            original_optimizer_step(planned_step_id=planned_step_id)
            qualification_probe.finalize_update(runtime)

        runtime.optimizer_step = optimizer_step
        self._restore_optimizer_step = lambda: setattr(runtime, "optimizer_step", original_optimizer_step)
        super().__init__(**kwargs)

    def run(self) -> Any:
        try:
            return super().run()
        finally:
            self._restore_optimizer_step()
            for handle in self._hook_handles:
                handle.remove()


@contextmanager
def _patch_trainer(probe: _LossQualificationProbe):
    import src.training.pipeline as pipeline

    original = pipeline.SupervisedTrainer
    pipeline.SupervisedTrainer = lambda **kwargs: _InstrumentedTrainer(
        qualification_probe=probe, **kwargs
    )
    try:
        yield
    finally:
        pipeline.SupervisedTrainer = original


def _rank_output(path: Path) -> Path:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return path
    rank = int(os.environ.get("RANK", "0"))
    return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")


def run(config: Path, output: Path) -> dict[str, Any]:
    output = _rank_output(output)
    probe = _LossQualificationProbe(output)
    error: BaseException | None = None
    try:
        from src import train as train_entry

        def runner(config_path: Path) -> Mapping[str, Any]:
            import src.training.pipeline as pipeline

            result = pipeline.run_training_pipeline(config_path)
            probe.pipeline_result = result
            return result

        with _patch_trainer(probe):
            train_entry.main(["--config", str(config)], runner=runner)
        probe.validate()
    except BaseException as exc:
        error = exc
    finally:
        artifact = probe.artifact(probe.runtime, probe.pipeline_result, error)
        _write_once(output, artifact)
    if error is not None:
        raise error
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qualify the maintained coordinate-codebook training entrypoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
