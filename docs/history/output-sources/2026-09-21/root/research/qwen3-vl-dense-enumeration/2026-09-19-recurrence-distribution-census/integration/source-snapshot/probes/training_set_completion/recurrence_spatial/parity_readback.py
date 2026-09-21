"""Parent-owned single-position parity consumer; no model execution."""
import argparse
import ast
import json
from pathlib import Path

import torch

TOL = 2e-4


def compare(source, target, *, target_index, consumed_actions, trace):
    assert consumed_actions == 119 and trace['action_index'] == 119
    for capture in (source, target):
        z = capture['logits']
        assert z.dtype == torch.float32 and z.ndim == 3
        assert z.shape[1:] == (1, 152670)
        assert capture['consumed_actions'] == consumed_actions
        assert capture['position_ids'] is not None
    assert target['logits'].shape[0] == 1
    sm = source['attention_mask'][target_index].bool()
    tm = target['attention_mask'][0].bool()
    assert torch.equal(source['input_ids'][target_index, sm], target['input_ids'][0, tm])
    assert len(trace['prefix_token_ids']) == 119
    assert target['input_ids'][0, tm].tolist() == target['prompt_token_ids'][0] + trace['prefix_token_ids']
    assert torch.equal(source['position_ids'][:, target_index, sm], target['position_ids'][:, 0, tm])
    assert source['prompt_token_ids'][target_index] == target['prompt_token_ids'][0]
    assert source['media_sha256'][target_index] == target['media_sha256'][0]
    assert source['image_grids'][target_index] == target['image_grids'][0]
    a, b = source['logits'][target_index, 0], target['logits'][0, 0]
    delta = float((a-b).abs().max())
    token = int(trace['token_id'])
    trace_delta = abs(float(a[token])-float(trace['chosen_raw_logit']))
    winner = int(a.argmax())
    passed = delta <= TOL and winner == int(b.argmax()) == token and trace_delta <= TOL
    return dict(status='passed' if passed else 'failed', full_vocab_max_abs_delta=delta,
                source_winner=winner, target_winner=int(b.argmax()),
                saved_trace_delta=trace_delta, tolerance=TOL,
                consumed_action_tokens=119, next_predicted_action_index=119,
                score_selection='final consumed input position, logits[:,0,:]',
                exact_unpadded_target_tokens_positions=True)


def self_test(archived):
    # Execute the actual archived nested consumer on position-coded logits.
    tree = ast.parse(Path(archived).read_text())
    stage = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'stage')
    ns = dict(torch=torch, Any=object, target_index=0, TOL=TOL,
              trace_step=lambda i: {}, top_summary=lambda *args: {'saved_trace': {'passed': True}})
    exec(compile(ast.Module(body=[stage], type_ignores=[]), str(archived), 'exec'), ns)
    old = torch.zeros(1, 6, 152670)
    old[0, 0, 1] = 10
    old[0, -1, 2] = 20
    assert not ns['stage']('row_boundary', old, old.clone(), 119)['passed']
    z = old[:, -1:, :].clone()
    c = dict(logits=z, consumed_actions=119, input_ids=torch.tensor([[4]+[5]*119]),
             attention_mask=torch.ones(1, 120, dtype=torch.long),
             position_ids=torch.arange(120).reshape(1,1,120).repeat(3,1,1), prompt_token_ids=[[4]],
             media_sha256=['image'], image_grids=[[1, 2, 2]])
    trace = dict(action_index=119, token_id=2, chosen_raw_logit=20., prefix_token_ids=[5]*119)
    assert compare(c, c, target_index=0, consumed_actions=119, trace=trace)['status'] == 'passed'
    try:
        compare(c, c, target_index=0, consumed_actions=119, trace={**trace, 'action_index': 118})
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong saved-trace offset accepted')
    return dict(status='passed', archived_wrong_position_falsified=True,
                single_position_consumer_passed=True, wrong_trace_offset_rejected=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path)
    p.add_argument('--target', type=Path)
    p.add_argument('--trace', type=Path)
    p.add_argument('--target-index', type=int, default=3)
    p.add_argument('--self-test', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.self_test:
        result = self_test(args.self_test)
    else:
        result = compare(torch.load(args.source, map_location='cpu', weights_only=False),
                         torch.load(args.target, map_location='cpu', weights_only=False),
                         target_index=args.target_index, consumed_actions=119,
                         trace=json.loads(args.trace.read_text()))
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    if result['status'] != 'passed':
        raise SystemExit(1)
