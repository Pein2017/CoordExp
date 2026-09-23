"""Three-loss observations around the existing maintained scale caller."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from pathlib import Path
from probes.training_set_completion.coordinate_codebook_alignment import scale_train as scale
from probes.training_set_completion.coordinate_codebook_alignment.three_loss_checks import ThreeLossQualificationProbe, ThreeLossRunnerHook

class BoundedLossHook(ThreeLossRunnerHook):
    def prepare_planned_step(self, *args, **kwargs):
        if len(self._probe.plan_records) >= 2:
            return self._inner.prepare_planned_step(*args, **kwargs)
        return super().prepare_planned_step(*args, **kwargs)

    def compute_micro_step(self, *args, **kwargs):
        if len(self._probe.records) >= 4:
            return self._inner.compute_micro_step(*args, **kwargs)
        result = super().compute_micro_step(*args, **kwargs)
        if len(self._probe.records) == 4:
            scale._write_once(scale._rank_output(self.output), self._probe.validate())
        return result

@contextmanager
def patched(probe, output, *, expected_weights=None):
    import src.training.pipeline as pipeline
    original = pipeline.SupervisedTrainer
    def trainer(**kwargs):
        qualification = (
            ThreeLossQualificationProbe()
            if expected_weights is None else ThreeLossQualificationProbe(expected_weights=expected_weights)
        )
        hook = BoundedLossHook(kwargs['loss_runner'], qualification)
        hook.output = output
        kwargs['loss_runner'] = hook
        return scale._ScaleTrainer(scale_probe=probe, **kwargs)
    pipeline.SupervisedTrainer = trainer
    try: yield
    finally: pipeline.SupervisedTrainer = original

def run(config, output, packing_plan):
    # This process-local wrapper fixes the amended horizon; the historical caller
    # and its default remain unchanged for existing users.
    old_updates, old_patch = scale.EXPECTED_UPDATES, scale._patch_trainer
    scale.EXPECTED_UPDATES = 984
    scale._patch_trainer = lambda probe: patched(probe, output.with_name('three-loss-objective.json'))
    try: return scale.run(config, output, packing_plan)
    finally: scale.EXPECTED_UPDATES, scale._patch_trainer = old_updates, old_patch

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--packing-plan',type=Path,required=True)
    a=p.parse_args();run(a.config,a.output,a.packing_plan)
