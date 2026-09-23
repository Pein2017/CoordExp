"""Eight-rank early-edge entry through the maintained scale trainer."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

from probes.training_set_completion.coordinate_codebook_alignment import scale_train as scale
from probes.training_set_completion.coordinate_codebook_alignment.three_loss_checks import (
    ThreeLossQualificationProbe,
    ThreeLossRunnerHook,
)


class _FirstTwoLossHook(ThreeLossRunnerHook):
    def prepare_planned_step(self, *args, **kwargs):
        if len(self._probe.plan_records) >= 2:
            return self._inner.prepare_planned_step(*args, **kwargs)
        return super().prepare_planned_step(*args, **kwargs)

    def compute_micro_step(self, *args, **kwargs):
        if len(self._probe.records) >= 2:
            return self._inner.compute_micro_step(*args, **kwargs)
        result = super().compute_micro_step(*args, **kwargs)
        if len(self._probe.records) == 2:
            scale._write_once(scale._rank_output(self.output), self._probe.validate())
        return result


@contextmanager
def _patched_trainer(probe, output):
    import src.training.pipeline as pipeline

    original = pipeline.SupervisedTrainer

    def trainer(**kwargs):
        hook = _FirstTwoLossHook(kwargs["loss_runner"], ThreeLossQualificationProbe())
        hook.output = output
        kwargs["loss_runner"] = hook
        return scale._ScaleTrainer(scale_probe=probe, **kwargs)

    pipeline.SupervisedTrainer = trainer
    try:
        yield
    finally:
        pipeline.SupervisedTrainer = original


def run(config: Path, output: Path, packing_plan: Path):
    old = (scale.EXPECTED_WORLD_SIZE, scale.EXPECTED_GRAD_ACCUM,
           scale.EXPECTED_UPDATES, scale._patch_trainer)
    scale.EXPECTED_WORLD_SIZE = 8
    scale.EXPECTED_GRAD_ACCUM = 1
    scale.EXPECTED_UPDATES = 984
    scale._patch_trainer = lambda probe: _patched_trainer(
        probe, output.with_name("three-loss-objective.json")
    )
    try:
        return scale.run(config, output, packing_plan)
    finally:
        (scale.EXPECTED_WORLD_SIZE, scale.EXPECTED_GRAD_ACCUM,
         scale.EXPECTED_UPDATES, scale._patch_trainer) = old


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packing-plan", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output, args.packing_plan)
