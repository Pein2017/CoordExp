"""Instrument the first two real updates of the maintained scale training run.

The training entrypoint and trainer remain the production implementation.  This
wrapper only observes the first two optimizer boundaries, checks their frozen
v3 pack schedule, and then leaves all later updates untouched.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from probes.training_set_completion.coordinate_codebook_alignment.qualify import (
    _embedding_delta_ids,
    _group_coverage,
    _parameter_category,
    _parameter_gradient_summary,
    _parameter_hashes,
    _tensor_hash,
)
from src.training.supervised_trainer import SupervisedTrainer


EXPECTED_WORLD_SIZE = 4
EXPECTED_GRAD_ACCUM = 2
EXPECTED_EFFECTIVE_BATCH = EXPECTED_WORLD_SIZE * EXPECTED_GRAD_ACCUM
EXPECTED_UPDATES = 1968
OBSERVED_UPDATES = 2
EXPECTED_CATEGORIES = ("language", "vision", "aligner", "input", "output", "codebook")


class ScaleTrainingError(RuntimeError):
    """A first-production invariant failed."""


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise FileExistsError(f"scale receipt collision: {path}")
        return
    path.write_text(data, encoding="utf-8")


def _rank_output(path: Path) -> Path:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return path
    rank = int(os.environ.get("RANK", "0"))
    return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _packet_schedule(packet: Mapping[str, Any]) -> dict[str, int]:
    schedule = packet.get("schedule")
    runtime_batch = schedule.get("runtime_batch") if isinstance(schedule, Mapping) else None
    rank_checks = packet.get("rank_checks")
    rank_check = rank_checks.get(str(EXPECTED_WORLD_SIZE)) if isinstance(rank_checks, Mapping) else None
    if not isinstance(schedule, Mapping) or not isinstance(runtime_batch, Mapping):
        raise ScaleTrainingError("packing exposure has no runtime schedule")
    if not isinstance(rank_check, Mapping):
        raise ScaleTrainingError("packing exposure has no declared-rank schedule check")
    values = {
        "updates": int(schedule.get("resolved_max_steps", -1)),
        "world_size": int(runtime_batch.get("world_size", -1)),
        "grad_accum": int(runtime_batch.get("resolved_grad_accum_steps", -1)),
        "effective_batch": int(runtime_batch.get("effective_batch_size", -1)),
        "rank_grad_accum": int(rank_check.get("gradient_accumulation", -1)),
        "rank_microsteps": int(rank_check.get("microsteps_per_rank", -1)),
    }
    expected = {
        "updates": EXPECTED_UPDATES,
        "world_size": EXPECTED_WORLD_SIZE,
        "grad_accum": EXPECTED_GRAD_ACCUM,
        "effective_batch": EXPECTED_EFFECTIVE_BATCH,
        "rank_grad_accum": EXPECTED_GRAD_ACCUM,
        "rank_microsteps": EXPECTED_UPDATES * EXPECTED_GRAD_ACCUM,
    }
    if values != expected or rank_check.get("rank_major_exact") is not True:
        raise ScaleTrainingError(
            f"v3 declared-rank schedule mismatch: observed={values} expected={expected}"
        )
    return values


def _validate_packing_plan(packet: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    _packet_schedule(packet)
    global_indices = packet.get("global_pack_indices")
    packs = packet.get("packs")
    if not isinstance(global_indices, list) or not isinstance(packs, list):
        raise ScaleTrainingError("packing exposure is missing global indices or packs")
    if len(global_indices) != EXPECTED_UPDATES * EXPECTED_EFFECTIVE_BATCH:
        raise ScaleTrainingError("v3 global pack schedule has the wrong length")
    by_index: dict[int, Mapping[str, Any]] = {}
    for item in packs:
        if not isinstance(item, Mapping):
            raise ScaleTrainingError("v3 pack entry is not an object")
        index = int(item.get("index", -1))
        row_ids = item.get("row_ids")
        if index in by_index or not isinstance(row_ids, list) or not row_ids:
            raise ScaleTrainingError(f"invalid or duplicate v3 pack entry: {index}")
        by_index[index] = item
    for raw_index in global_indices:
        index = int(raw_index)
        if index not in by_index:
            raise ScaleTrainingError(f"global schedule references unknown pack {index}")
    return by_index


def expected_first_two_updates(
    packet: Mapping[str, Any], *, rank: int
) -> list[dict[str, Any]]:
    """Return the two local micro-step bindings expected for each update."""

    schedule = _packet_schedule(packet)
    if rank < 0 or rank >= schedule["world_size"]:
        raise ScaleTrainingError(f"rank is outside the v3 world: {rank}")
    packs = _validate_packing_plan(packet)
    global_indices = packet["global_pack_indices"]
    result: list[dict[str, Any]] = []
    for step in range(1, OBSERVED_UPDATES + 1):
        for local_index in range(schedule["grad_accum"]):
            global_position = (
                (step - 1) * schedule["effective_batch"]
                + local_index * schedule["world_size"]
                + rank
            )
            pack_index = int(global_indices[global_position])
            pack = packs[pack_index]
            result.append(
                {
                    "planned_step_id": step,
                    "local_micro_step_index": local_index,
                    "pack_index": pack_index,
                    "pack_id": int(pack["index"]),
                    "example_ids": [str(value) for value in pack["row_ids"]],
                }
            )
    return result


def _micro_step_identity(micro_step: Any) -> dict[str, Any]:
    metadata = getattr(micro_step, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ScaleTrainingError("production micro-step has no metadata binding")
    example_ids = metadata.get("example_ids")
    if "pack_id" not in metadata or (
        not isinstance(example_ids, Sequence) or isinstance(example_ids, (str, bytes))
    ):
        raise ScaleTrainingError("production micro-step metadata lacks pack/example IDs")
    return {
        "pack_id": int(metadata["pack_id"]),
        "example_ids": [str(value) for value in example_ids],
    }


def _category_gradient_summary(model: Any) -> dict[str, Any]:
    gradients = _parameter_gradient_summary(model)
    parameter_map = dict(model.named_parameters())
    input_ids, output_ids = _embedding_delta_ids(model)
    categories = {name: [] for name in ("language", "vision", "aligner", "input", "output", "codebook")}
    for name, value in gradients.items():
        category = _parameter_category(
            name,
            parameter_map[name],
            input_ids=input_ids,
            output_ids=output_ids,
        )
        if category in categories:
            categories[category].append((name, value))
    result: dict[str, Any] = {}
    unclassified: list[str] = []
    for name, value in gradients.items():
        category = _parameter_category(
            name,
            parameter_map[name],
            input_ids=input_ids,
            output_ids=output_ids,
        )
        if category is None:
            unclassified.append(name)
    unwrapped = getattr(model, "module", model)
    if getattr(unwrapped, "coordinate_codebook", None) is None:
        if categories["codebook"]:
            raise ScaleTrainingError("codebook parameters exist without an installed codebook")
        del categories["codebook"]
    for category, values in categories.items():
        nonzero = [name for name, value in values if value["nonzero"]]
        result[category] = {
            "parameter_count": len(values),
            "gradient_present_count": sum(bool(value["present"]) for _, value in values),
            "finite_count": sum(bool(value["finite"]) for _, value in values),
            "nonzero_count": len(nonzero),
            "nonzero_parameter_names": nonzero,
            "max_norm": max(
                (float(value["norm"]) for _, value in values if value["norm"] is not None),
                default=0.0,
            ),
        }
    result["_unclassified"] = {"parameter_names": unclassified}
    return result


def _parameter_deltas(
    model: Any, before: Mapping[str, torch.Tensor]
) -> tuple[dict[str, str], dict[str, float]]:
    hashes: dict[str, str] = {}
    maxima: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in before:
            continue
        delta = parameter.detach().cpu().float() - before[name].float()
        hashes[name] = _tensor_hash(delta)
        maxima[name] = float(delta.abs().max())
    return hashes, maxima


def _validate_parameter_delta(
    *,
    planned_step_id: int,
    learning_rates: Sequence[float],
    maxima: Mapping[str, float],
    require_change: bool = True,
) -> None:
    """Apply scheduler-aware update evidence without changing optimizer behavior."""

    rates = [float(value) for value in learning_rates]
    if not rates or any(not math.isfinite(value) or value < 0.0 for value in rates):
        raise ScaleTrainingError("pre-update optimizer learning rates are invalid")
    changed = any(float(value) > 0.0 for value in maxima.values())
    positive_rate = any(value > 0.0 for value in rates)
    if positive_rate and require_change and not changed:
        raise ScaleTrainingError("positive-LR production update produced no parameter delta")
    if not positive_rate and planned_step_id != 1:
        raise ScaleTrainingError(
            "only the first production update may have an all-zero pre-update LR"
        )


class _ScaleProbe:
    def __init__(self, output: Path, packet: Mapping[str, Any]) -> None:
        self.output = _rank_output(output)
        self.packet = packet
        self.expected_schedule = _packet_schedule(packet)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.expected_bindings = expected_first_two_updates(packet, rank=self.rank)
        self.plan_records: list[dict[str, Any]] = []
        self.update_records: list[dict[str, Any]] = []
        self.frozen_before: dict[str, str] = {}
        self.frozen_after: dict[str, str] = {}
        self.optimizer_coverage: dict[str, Any] | None = None
        self.runtime: Any | None = None
        self.receipt_written = False
        self.category_nonzero_seen = {category: False for category in EXPECTED_CATEGORIES}
        self.in_flight_update: dict[str, Any] | None = None

    def bind(self, *, schedule: Any, runtime: Any, model: Any) -> None:
        actual = {
            "updates": int(schedule.resolved_max_steps),
            "world_size": int(schedule.runtime_batch.world_size),
            "grad_accum": int(schedule.runtime_batch.resolved_grad_accum_steps),
            "effective_batch": int(schedule.runtime_batch.effective_batch_size),
        }
        expected = {
            "updates": EXPECTED_UPDATES,
            "world_size": EXPECTED_WORLD_SIZE,
            "grad_accum": EXPECTED_GRAD_ACCUM,
            "effective_batch": EXPECTED_EFFECTIVE_BATCH,
        }
        if actual != expected or self.world_size != EXPECTED_WORLD_SIZE:
            raise ScaleTrainingError(
                f"production schedule mismatch: observed={actual}, env_world_size={self.world_size}, expected={expected}"
            )
        if self.rank < 0 or self.rank >= self.world_size:
            raise ScaleTrainingError(f"invalid production rank: {self.rank}")
        self.runtime = runtime
        self.frozen_before = _parameter_hashes(model)
        self.optimizer_coverage = _group_coverage(runtime)
        if not self.optimizer_coverage["exact_name_coverage"]:
            raise ScaleTrainingError("optimizer trainable surface coverage is not exact")
        installed = getattr(getattr(model, "module", model), "coordinate_codebook", None)
        expects_codebook = "codebook" in EXPECTED_CATEGORIES
        if (installed is not None) != expects_codebook:
            raise ScaleTrainingError("codebook installation and expected trainable categories disagree")
        if not expects_codebook and self.optimizer_coverage["trainable_parameter_count"] != 902:
            raise ScaleTrainingError("injection-off assembly does not have exactly 902 common trainable tensors")

    def capture_plan(self, micro_steps: Sequence[Any], plan: Any) -> None:
        if len(self.plan_records) >= OBSERVED_UPDATES:
            return
        step = len(self.plan_records) + 1
        if len(micro_steps) != EXPECTED_GRAD_ACCUM:
            raise ScaleTrainingError(
                f"step {step} has {len(micro_steps)} local micro-steps, expected {EXPECTED_GRAD_ACCUM}"
            )
        expected = self.expected_bindings[(step - 1) * EXPECTED_GRAD_ACCUM : step * EXPECTED_GRAD_ACCUM]
        actual_records = []
        for local_index, (micro_step, expected_identity) in enumerate(zip(micro_steps, expected, strict=True)):
            identity = _micro_step_identity(micro_step)
            actual = {
                "planned_step_id": step,
                "local_micro_step_index": local_index,
                "rank": self.rank,
                **identity,
            }
            if actual["pack_id"] != expected_identity["pack_id"] or actual["example_ids"] != expected_identity["example_ids"]:
                raise ScaleTrainingError(
                    f"v3 pack binding mismatch at rank {self.rank}, step {step}, micro-step {local_index}: "
                    f"observed={actual} expected={expected_identity}"
                )
            actual_records.append(actual)
        global_records = self._check_global_pack_bindings(step, actual_records)
        denominators = getattr(plan, "denominators", None)
        denominator = denominators.get("base_ce") if isinstance(denominators, Mapping) else None
        if denominator is None or not callable(getattr(denominator, "to_artifact_dict", None)):
            raise ScaleTrainingError("production loss plan has no base-ce denominator")
        denominator_artifact = denominator.to_artifact_dict()
        if denominator_artifact.get("denominator_scope") != "planned_step_global":
            raise ScaleTrainingError("first-production loss denominator is not global")
        if int(getattr(plan, "world_size", -1)) != EXPECTED_WORLD_SIZE:
            raise ScaleTrainingError("first-production loss plan has the wrong world size")
        backend_scale = float(getattr(plan, "backend_gradient_scale", math.nan))
        if not math.isfinite(backend_scale) or backend_scale != float(EXPECTED_WORLD_SIZE):
            raise ScaleTrainingError(f"unexpected global backend gradient scale: {backend_scale}")
        expected_segment_count = sum(
            len(item["example_ids"]) for item in global_records
        )
        if int(denominator_artifact.get("eligible_segment_count", -1)) != expected_segment_count:
            raise ScaleTrainingError(
                f"global denominator segment count mismatch at step {step}: "
                f"observed={denominator_artifact.get('eligible_segment_count')} "
                f"expected={expected_segment_count}"
            )
        self.plan_records.append(
            {
                "planned_step_id": step,
                "pack_bindings": actual_records,
                "global_pack_binding_count": len(global_records),
                "global_segment_count_from_pack_rows": expected_segment_count,
                "eligible_segment_count": int(denominator_artifact["eligible_segment_count"]),
                "global_denominator": denominator_artifact,
                "backend_gradient_scale": backend_scale,
                "plan_counts": dict(getattr(plan, "counts", {})),
            }
        )

    def _check_global_pack_bindings(
        self, step: int, local_records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        distributed_ready = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if distributed_ready:
            gathered: list[Any] = [None] * self.world_size
            torch.distributed.all_gather_object(gathered, local_records)
            if len(gathered) != self.world_size or any(
                not isinstance(item, list) for item in gathered
            ):
                raise ScaleTrainingError("global first-production pack gather is malformed")
            global_records = [record for records in gathered for record in records]
        elif self.world_size == 1:
            # This branch is only for the CPU helper test; a production
            # four-rank run must use the global gather path below.
            global_records = local_records
        else:
            raise ScaleTrainingError(
                "four-rank first-production pack check lacks a distributed gather"
            )
        expected_global: list[dict[str, Any]] = []
        ranks = range(EXPECTED_WORLD_SIZE) if distributed_ready else (self.rank,)
        for local_index in range(EXPECTED_GRAD_ACCUM):
            for rank in ranks:
                expected = expected_first_two_updates(self.packet, rank=rank)[
                    (step - 1) * EXPECTED_GRAD_ACCUM + local_index
                ]
                expected_global.append({"rank": rank, **expected})
        observed_keys = sorted(
            (int(item["rank"]), int(item["local_micro_step_index"]), int(item["pack_id"]), tuple(item["example_ids"]))
            for item in global_records
        )
        expected_keys = sorted(
            (int(item["rank"]), int(item["local_micro_step_index"]), int(item["pack_id"]), tuple(item["example_ids"]))
            for item in expected_global
        )
        if observed_keys != expected_keys:
            raise ScaleTrainingError(
                f"global first-production pack bindings mismatch at step {step}: "
                f"observed={observed_keys} expected={expected_keys}"
            )
        return global_records

    def before_update(self, *, planned_step_id: int, runtime: Any) -> dict[str, Any]:
        expected_step = len(self.update_records) + 1
        if planned_step_id != expected_step or len(self.plan_records) < expected_step:
            raise ScaleTrainingError(
                f"unexpected genuine update boundary: step={planned_step_id}, expected={expected_step}"
            )
        gradients = _category_gradient_summary(runtime.model)
        if set(gradients) != set(EXPECTED_CATEGORIES) | {"_unclassified"}:
            raise ScaleTrainingError("intended gradient category set is incomplete")
        if gradients["_unclassified"]["parameter_names"]:
            raise ScaleTrainingError(
                "trainable parameters are not covered by the category mapper: "
                f"{gradients['_unclassified']['parameter_names'][:8]}"
            )
        if any(not gradients[category]["parameter_count"] for category in EXPECTED_CATEGORIES):
            raise ScaleTrainingError("an intended trainable gradient category is empty")
        active_categories = {category: gradients[category] for category in EXPECTED_CATEGORIES}
        if any(item["finite_count"] == 0 for item in active_categories.values()):
            raise ScaleTrainingError("an intended category has no finite gradient")
        for category, item in active_categories.items():
            self.category_nonzero_seen[category] |= item["nonzero_count"] > 0
        optimizer = getattr(runtime, "optimizer", None)
        groups = getattr(optimizer, "param_groups", None)
        if not isinstance(groups, Sequence):
            raise ScaleTrainingError("runtime optimizer has no parameter groups")
        learning_rates = [float(group.get("lr", math.nan)) for group in groups]
        _validate_parameter_delta(
            planned_step_id=planned_step_id,
            learning_rates=learning_rates,
            maxima={},
            require_change=False,
        )
        before_values = {
            name: parameter.detach().cpu().clone()
            for name, parameter in runtime.model.named_parameters()
            if parameter.requires_grad
        }
        receipt = {
            "planned_step_id": planned_step_id,
            "category_gradients": gradients,
            "optimizer_learning_rates_before": learning_rates,
            "trainable_parameter_hashes_before": {
                name: _tensor_hash(parameter)
                for name, parameter in runtime.model.named_parameters()
                if parameter.requires_grad
            },
        }
        self.in_flight_update = receipt
        return {**receipt, "before_values": before_values}

    def after_update(
        self, *, planned_step_id: int, runtime: Any, before: Mapping[str, Any]
    ) -> None:
        delta_hashes, delta_maxima = _parameter_deltas(runtime.model, before["before_values"])
        _validate_parameter_delta(
            planned_step_id=planned_step_id,
            learning_rates=before["optimizer_learning_rates_before"],
            maxima=delta_maxima,
        )
        self.frozen_after = _parameter_hashes(runtime.model)
        if self.frozen_after != self.frozen_before:
            raise ScaleTrainingError("frozen parameters changed during first-production updates")
        if not delta_hashes:
            raise ScaleTrainingError("first-production optimizer update has no trainable parameters")
        self.update_records.append(
            {
                "planned_step_id": planned_step_id,
                "category_gradients": before["category_gradients"],
                "optimizer_learning_rates_before": before["optimizer_learning_rates_before"],
                "trainable_parameter_hashes_before": before["trainable_parameter_hashes_before"],
                "parameter_delta_hashes": delta_hashes,
                "parameter_max_abs_deltas": delta_maxima,
            }
        )
        self.in_flight_update = None
        if len(self.update_records) == OBSERVED_UPDATES:
            self.write_candidate(runtime=runtime)

    def write_candidate(self, *, runtime: Any) -> None:
        if self.receipt_written:
            return
        if not all(self.category_nonzero_seen.values()):
            missing = [name for name, seen in self.category_nonzero_seen.items() if not seen]
            raise ScaleTrainingError(
                f"intended categories lacked a nonzero gradient across first two updates: {missing}"
            )
        model = getattr(runtime.model, "module", runtime.model)
        codebook = getattr(model, "coordinate_codebook", None)
        if getattr(codebook, "mode", None) == "early_patch_edges":
            projection = "coordinate_codebook.projection.weight"
            gradient_seen = any(
                any(name.endswith(projection) for name in item["category_gradients"]["codebook"]["nonzero_parameter_names"])
                for item in self.update_records
            )
            update_seen = any(
                value > 0
                for item in self.update_records
                for name, value in item["parameter_max_abs_deltas"].items()
                if name.endswith(projection)
            )
            if not gradient_seen or not update_seen:
                raise ScaleTrainingError("early projection lacked a gradient or positive-LR update")
        _write_once(self.output, self.artifact(status="candidate", error=None))
        self.receipt_written = True

    def write_failure(self, error: BaseException) -> None:
        if self.receipt_written:
            return
        _write_once(self.output, self.artifact(status="blocked", error=repr(error)))
        self.receipt_written = True

    def artifact(self, *, status: str, error: str | None) -> dict[str, Any]:
        return {
            "schema": "coordinate_codebook_scale.first_production.v1",
            "status": status,
            "error": error,
            "rank": self.rank,
            "world_size": self.world_size,
            "expected_schedule": self.expected_schedule,
            "first_two_updates": self.update_records,
            "loss_plans": self.plan_records,
            "frozen_parameter_hashes_before": self.frozen_before,
            "frozen_parameter_hashes_after": self.frozen_after,
            "frozen_parameters_unchanged": self.frozen_before == self.frozen_after,
            "optimizer_coverage": self.optimizer_coverage,
            "in_flight_update": self.in_flight_update,
        }


class _ScaleLossRunner:
    def __init__(self, inner: Any, probe: _ScaleProbe) -> None:
        self._inner = inner
        self._probe = probe

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def prepare_planned_step(self, micro_steps: Sequence[Any], **kwargs: Any) -> Any:
        plan = self._inner.prepare_planned_step(micro_steps, **kwargs)
        self._probe.capture_plan(micro_steps, plan)
        return plan


class _ScaleTrainer(SupervisedTrainer):
    def __init__(self, *, scale_probe: _ScaleProbe, **kwargs: Any) -> None:
        self.scale_probe = scale_probe
        runtime = kwargs["runtime"]
        self.scale_probe.bind(schedule=kwargs["schedule"], runtime=runtime, model=kwargs["model"])
        kwargs["loss_runner"] = _ScaleLossRunner(kwargs["loss_runner"], scale_probe)
        original_optimizer_step = runtime.optimizer_step

        def optimizer_step(*, planned_step_id: int) -> None:
            if len(scale_probe.update_records) >= OBSERVED_UPDATES:
                original_optimizer_step(planned_step_id=planned_step_id)
                return
            before = scale_probe.before_update(
                planned_step_id=planned_step_id, runtime=runtime
            )
            original_optimizer_step(planned_step_id=planned_step_id)
            scale_probe.after_update(
                planned_step_id=planned_step_id, runtime=runtime, before=before
            )

        runtime.optimizer_step = optimizer_step
        self._restore_optimizer_step = lambda: setattr(runtime, "optimizer_step", original_optimizer_step)
        try:
            super().__init__(**kwargs)
        except BaseException:
            self._restore_optimizer_step()
            raise

    def run(self) -> Any:
        try:
            return super().run()
        finally:
            self._restore_optimizer_step()


@contextmanager
def _patch_trainer(probe: _ScaleProbe):
    import src.training.pipeline as pipeline

    original = pipeline.SupervisedTrainer
    pipeline.SupervisedTrainer = lambda **kwargs: _ScaleTrainer(
        scale_probe=probe, **kwargs
    )
    try:
        yield
    finally:
        pipeline.SupervisedTrainer = original


def run(config: Path, output: Path, packing_plan: Path) -> dict[str, Any]:
    packet = _load_json(packing_plan)
    probe = _ScaleProbe(output, packet)
    error: BaseException | None = None
    try:
        from src import train as train_entry

        def runner(config_path: Path) -> Mapping[str, Any]:
            import src.training.pipeline as pipeline

            return pipeline.run_training_pipeline(config_path)

        with _patch_trainer(probe):
            train_entry.main(["--config", str(config)], runner=runner)
        if len(probe.update_records) != OBSERVED_UPDATES:
            raise ScaleTrainingError("training ended before two genuine updates")
    except BaseException as exc:
        error = exc
        probe.write_failure(exc)
    if error is not None:
        raise error
    return probe.artifact(status="candidate", error=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture first two scale-training updates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packing-plan", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.config, args.output, args.packing_plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
