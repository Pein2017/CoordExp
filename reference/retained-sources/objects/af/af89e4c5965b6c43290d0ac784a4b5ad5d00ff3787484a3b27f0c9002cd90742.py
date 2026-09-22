import math

import pytest
import torch

from probes.dora_owner_learning.route_access import (
    EOS, PAD, branch_support, checked_ids, first_fork, indexed, prepare,
    publish, score_logits, section_labels, validate_score,
)
from src.qwen.native import ExactReplay


def test_full_vocab_math_ties_causal_publication(tmp_path):
    # The target's best competitor is not restricted to a selected vocabulary.
    logits = torch.tensor([[1., 1., -2., 0.], [-3., 2., 4., 1.]])
    targets = torch.tensor([1, 1])
    score = score_logits(logits, targets, prompt_length=7)
    assert score['first_non_argmax'] == 0
    assert score['positions'][0]['target_in_top_tie']
    assert score['positions'][0]['top1_tie_count'] == 2
    assert score['positions'][0]['top1_id'] == 0
    assert score['positions'][1]['best_other_id'] == 2
    assert score['positions'][1]['target_best_other_margin'] == -2
    for i, p in enumerate(score['positions']):
        independent = float(logits[i, 1]) - math.log(sum(math.exp(float(v)) for v in logits[i]))
        assert p['logprob'] == pytest.approx(independent, abs=1e-6)
    validate_score(score, {'ids': [1, 1]}, 7)
    publish(tmp_path / 'score.json', score)
    original = (tmp_path / 'score.json').read_bytes()
    with pytest.raises(Exception):
        publish(tmp_path / 'score.json', {'corruption': True})
    assert (tmp_path / 'score.json').read_bytes() == original


@pytest.mark.parametrize('ids,stop', [([1, EOS, 2], 'im_end'), ([1, PAD, EOS], 'im_end'),
                                      ([1, 2], 'im_end'), ([1, EOS], 'length'), ([1, True, EOS], 'im_end'),
                                      ([], 'im_end')])
def test_terminal_corruption(ids, stop):
    with pytest.raises(ValueError):
        checked_ids(ids, stop)


def test_alignment_corruption():
    logits = torch.tensor([[1., 2.], [3., 4.]])
    targets = torch.tensor([0, 1])
    score = score_logits(logits, targets, prompt_length=3)
    with pytest.raises(ValueError):
        validate_score(score, {'ids': [1, 0]}, 3)
    with pytest.raises(ValueError):
        validate_score(score, {'ids': [0, 1]}, 4)
    score['positions'][1]['position'] = 0
    with pytest.raises(ValueError):
        validate_score(score, {'ids': [0, 1]}, 3)
    with pytest.raises(ValueError):
        score_logits(logits, targets[:1], prompt_length=3)
    with pytest.raises(ValueError):
        score_logits(logits * float('nan'), targets, prompt_length=3)


def test_native_alignment_has_teeth():
    replay = ExactReplay({'logits_to_keep': 3}, torch.tensor([0, 1]), 7)
    full = torch.arange(12.).reshape(1, 3, 4)
    assert torch.equal(replay.aligned_logits(full), full[0, :2])
    with pytest.raises(Exception):
        replay.aligned_logits(full[:, :2])


def test_duplicate_missing_and_collision(tmp_path):
    with pytest.raises(ValueError):
        indexed([{'id': 1}, {'id': 1}], 'id')
    with pytest.raises(ValueError):
        prepare(tmp_path)
    with pytest.raises(ValueError):
        branch_support([1], [{'seed': 1, 'advantage': 0, 'action_token_ids': [1]}])


def test_branch_support_prefix_cancellation():
    actions = [dict(seed=i, advantage=a, action_token_ids=ids) for i, (a, ids) in enumerate([
        (1., [1, 2, EOS]), (2., [1, 2, EOS]), (-1., [1, 3, EOS]), (-2., [1, 4, EOS])])]
    support = branch_support([1, 2, EOS], actions)
    assert support[0]['target_branch_advantage_sum'] == 0
    assert support[1]['target_branch_advantage_sum'] == 3
    assert support[1]['prefix_advantage_sum'] == 0
    assert support[2]['shared_prefix_siblings'] == 2
    assert support[2]['prefix_advantage_sum'] == 3
    assert first_fork([1, 2, EOS], [1, 3, EOS]) == 1
    assert first_fork([1, EOS], [1, EOS]) is None


def test_sections_overlap_explicitly_and_reject_wrong_span():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return ''.join({1: '<row>', 2: 'cat', 3: '</row>', EOS: '<eos>'}[i] for i in ids)
    sample = dict(comparison={'gained': ['owner']}, invalid_predictions=0, parsed_prediction_count=1,
                  **{'50': {'matches': [{'pred_index': 0, 'owner': 'owner'}]}})
    pred = dict(char_start=0, char_end=14, raw_span_text='<row>cat</row>')
    pred['char_end'] = len(pred['raw_span_text'])
    labels = section_labels([1, 2, 3, EOS], Tokenizer(), [pred], sample, [1, 2, EOS])
    assert labels['partition'] == ['gained_target_row'] * 3 + ['wrapper_eos_or_unparsed']
    assert labels['common_prefix_length'] == 2
    pred['char_start'] = 1
    with pytest.raises(ValueError):
        section_labels([1, 2, 3, EOS], Tokenizer(), [pred], sample, [1, EOS])
