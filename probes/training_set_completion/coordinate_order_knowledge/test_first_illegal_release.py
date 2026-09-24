from types import SimpleNamespace

import pytest
import torch

from probes.training_set_completion.coordinate_order_knowledge import first_illegal_release as release
from probes.training_set_completion.coordinate_order_knowledge import probe


def test_only_first_x2_slot_can_be_forced():
    prefix = [15, probe.BOX_START, probe.COORD0, probe.COORD0]
    release.validate_boundary(prefix, 4, probe.COORD0 + 47)
    with pytest.raises(ValueError, match='boundary'):
        release.validate_boundary(prefix, 3, probe.COORD0 + 47)
    with pytest.raises(ValueError, match='boundary'):
        release.validate_boundary([15, probe.BOX_START, probe.COORD0 + 1, probe.COORD0], 4, probe.COORD0 + 47)
    with pytest.raises(ValueError, match='boundary'):
        release.validate_boundary(prefix, 4, probe.COORD0 + 2)


def test_actual_continuation_wrapper_rejects_persistent_override(monkeypatch):
    class Model:
        def generate(self, **kwargs):
            ids = kwargs['input_ids']
            # Emitted second token disagrees with unmodified-logit argmax.
            logits = (torch.tensor([[0., 2., 0.]]), torch.tensor([[0., 0., 2.]]))
            return SimpleNamespace(sequences=torch.cat([ids, torch.tensor([[1, 1]])], dim=1), logits=logits)

    model = Model()
    qwen = SimpleNamespace(model=model, tokenizer=SimpleNamespace(pad_token_id=0))

    def caller(model, batch, **kwargs):
        model.generate(input_ids=torch.tensor([[11, 12]]))
        raise AssertionError('persistent override was accepted')

    monkeypatch.setattr(release, 'generate_continuations', caller)
    with pytest.raises(ValueError, match='override persisted'):
        release._continue(qwen, object(), [11, 12], 2)
    assert model.generate.__func__ is Model.generate
