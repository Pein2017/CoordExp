from types import SimpleNamespace

import pytest
import torch

from probes.training_set_completion.coordinate_order_knowledge import probe
from probes.training_set_completion.coordinate_order_knowledge import sustained_legality as legal


C = list(range(probe.COORD0, probe.COORD0 + 1000))
START = [probe.REF_START, 8987, probe.REF_END, probe.BOX_START]


def mask(prefix, *, enabled=True):
    p = legal.CoordinatePolicy(1, C, enabled=enabled)
    scores = torch.zeros((1, C[-1] + 3))
    scores[0, C[0]] = 4
    out = p(torch.tensor([[10, *prefix]]), scores)
    return out, scores, p.records[-1]


def test_strict_coordinate_slots_and_midbox_prefix():
    for prefix, stage, lo, hi in (
        (START, 'x1', 0, 998),
        (START + [C[600]], 'y1', 0, 998),
        (START + [C[600], C[500]], 'x2', 601, 999),
        (START + [C[600], C[500], C[999]], 'y2', 501, 999),
        (START + [C[998], C[0]], 'x2', 999, 999),
    ):
        out, _, record = mask(prefix)
        assert record['stage'] == stage and record['legal_range'] == [lo, hi]
        assert torch.isfinite(out[0, C[lo]:C[hi] + 1]).all()
        assert not torch.isfinite(out[0, C[lo - 1]]).item() if lo else True
        assert not torch.isfinite(out[0, C[hi + 1]]).item() if hi < 999 else True
        assert not torch.isfinite(out[0, 8987]).item()  # family gate
    # Equality and reversal are forbidden, including the current x2=0 boundary.
    out, _, _ = mask(START + [C[0], C[0]])
    assert not torch.isfinite(out[0, C[0]]) and torch.isfinite(out[0, C[1]])
    with pytest.raises(legal.MalformedHistory, match='no legal successor'):
        mask(START + [C[999], C[0]])


def test_new_row_reset_outside_identity_and_fail_closed():
    complete = START + [C[0], C[0], C[1], C[1], probe.BOX_END]
    outside, original, record = mask(complete)
    assert record['stage'] == 'outside' and torch.equal(outside, original)
    after = complete + START
    out, _, record = mask(after)
    assert record['stage'] == 'x1' and record['legal_range'] == [0, 998]
    assert not torch.isfinite(out[0, C[999]])
    off, original, record = mask(START + [C[600], C[0]], enabled=False)
    assert record['stage'] == 'x2' and torch.equal(off, original)
    with pytest.raises(legal.MalformedHistory, match='outside'):
        mask([8987])
    with pytest.raises(legal.MalformedHistory, match='missing box start'):
        mask([probe.REF_START, 8987, probe.REF_END, 8987])
    with pytest.raises(legal.MalformedHistory, match='token after EOS'):
        mask([legal.release.EOS, probe.REF_START])


def test_generation_caller_inserts_only_probe_policy(monkeypatch):
    class Model:
        def generate(self, **kwargs):
            scores = torch.zeros((1, C[-1] + 3))
            scores[0, C[0]] = 3
            scores[0, C[1]] = 2
            adjusted = kwargs['logits_processor'](kwargs['input_ids'], scores)
            assert not torch.isfinite(adjusted[0, C[0]])
            assert adjusted[0, C[1]] == 2
            chosen = adjusted.argmax(dim=-1).view(1, 1)
            return SimpleNamespace(sequences=torch.cat((kwargs['input_ids'], chosen), dim=1),
                                   logits=(scores,), scores=(adjusted,))

    model = Model()
    qwen = SimpleNamespace(model=model, tokenizer=SimpleNamespace(pad_token_id=0))
    p = legal.CoordinatePolicy(1, C)

    def caller(model, batch, **kwargs):
        result = model.generate(input_ids=torch.tensor([[10, *START, C[0], C[0]]]))
        return (SimpleNamespace(token_ids=(int(result.sequences[0, -1]),)),)

    monkeypatch.setattr(legal, 'generate_continuations', caller)
    result, trace = legal._generate(qwen, object(), START + [C[0], C[0]], p, 1)
    assert result.token_ids == (C[1],) and trace['steps'][0]['stage'] == 'x2'
    assert model.generate.__func__ is Model.generate
