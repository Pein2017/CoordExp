"""Small real-entry prefix/score gate before the release matrix."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from probes.training_set_completion.coordinate_continuity.runtime import _trace_expectation
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.recurrence_phase_decision import common
from probes.training_set_completion.recurrence_phase_decision.prepare import PANEL, binding, write_new
from probes.training_set_completion.untied_shared import load_model
from src.qwen.native import derive_position_ids, exact_history_inputs

ATOL = 2e-4


def families(plan: dict, model: str) -> list[dict]:
    sources = {s['boundary_id']: s for s in plan['source_summaries']}
    chosen = {}
    for cell in plan['cells']:
        if cell['model'] != model or cell['release_mode'] != 'immediate' or cell['delta'] != 1 or cell['status'] != 'ready':
            continue
        source = sources[cell['boundary_id']]
        key = source['source_panel']['path']
        chosen.setdefault(key, cell)
    if len(chosen) != 3:
        raise ValueError(f'expected three source families for {model}, got {len(chosen)}')
    return [chosen[k] for k in sorted(chosen)]


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    plan = common.read_plan(args.plan)
    panel = json.loads(PANEL.read_text())
    cells = families(plan, args.model)
    write_new(args.output / 'launch.json', {
        'schema': 'recurrence_phase_decision.gate_launch.v1', 'status': 'frozen_before_model',
        'model': args.model, 'device': args.device, 'cells': cells,
        'plan': binding(args.plan), 'panel': binding(PANEL),
        'code': {name: binding(Path(__file__).with_name(name)) for name in ('qualify.py', 'common.py', 'prepare.py')}})
    start = time.monotonic()
    q, identity = load_model(args.model, torch.device(args.device))
    model = q.model
    tokenizer = q.tokenizer
    for expected in (151645, 151646, 151647, 151648, 151649, 151670, 152669):
        text = tokenizer.convert_ids_to_tokens(expected)
        if tokenizer.convert_tokens_to_ids(text) != expected:
            raise ValueError(f'special token roundtrip changed at {expected}: {text}')
    if q.model.get_output_embeddings().selected_token_ids[4:].tolist() != list(range(151670, 152670)):
        raise ValueError('coordinate token ID family changed')
    head_inputs = []
    def head_hook(_module, values):
        head_inputs.append(values[0].detach().float().cpu())
    handle = model.get_output_embeddings().register_forward_pre_hook(head_hook)
    calls = 0
    comparisons = []
    try:
        for family_index, cell in enumerate(cells):
            batch, raw, trace, source = common.bound_source(plan, panel, cell['boundary_id'], q, torch.device(args.device))
            boundary = source['boundary']
            target = int(boundary['batch_index'])
            end = int(cell['prefix_end'])
            saved = {}
            for mode in ('native', 'changed', 'exact_native' if family_index == 0 else None):
                if mode is None:
                    continue
                site = None if mode != 'changed' else (int(cell['site_offset']), int(cell['old_token_id']), int(cell['new_token_id']))
                if mode == 'exact_native':
                    suffixes = _prefix_tokens(raw, end, int(tokenizer.pad_token_id))
                    histories = [list(prompt) + suffix for prompt, suffix in zip(batch.prompt_token_ids, suffixes, strict=True)]
                    inputs = exact_history_inputs(model, batch.inputs, histories,
                                                  pad_token_id=int(tokenizer.pad_token_id), logits_to_keep=1)
                else:
                    inputs = common.prefix(batch, raw, boundary, end, q, torch.device(args.device), site)
                    inputs['logits_to_keep'] = 1
                    inputs['use_cache'] = False
                    inputs['return_dict'] = True
                positions = inputs.get('position_ids')
                if positions is None:
                    positions = derive_position_ids(model=model, input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'], image_grid_thw=inputs['image_grid_thw'],
                        video_grid_thw=inputs.get('video_grid_thw'))
                head_inputs.clear()
                with torch.inference_mode():
                    logits = model(**inputs).logits[target, -1].detach().float().cpu()
                calls += 1
                if calls > args.max_forwards:
                    raise RuntimeError('gate forward budget exceeded')
                if not head_inputs:
                    raise RuntimeError('LM head input missing')
                payload = {'schema': 'recurrence_phase_decision.gate_vector.v1', 'cell_id': cell['id'],
                           'mode': mode, 'consumed_source_tokens': end, 'next_action_offset': end,
                           'input_ids': inputs['input_ids'][target].detach().cpu(),
                           'attention_mask': inputs['attention_mask'][target].detach().cpu(),
                           'position_ids': positions[:, target].detach().cpu(),
                           'logits': logits, 'head_input': head_inputs[-1][target, -1].detach().cpu()}
                path = args.output / f'{family_index}-{mode}.pt'
                torch.save(payload, path)
                saved[mode] = {'path': path, 'logits': logits, 'binding': binding(path)}
            native = saved['native']['logits']
            top = native.topk(2)
            expected = _trace_expectation(trace, target, end)
            if int(top.indices[0]) != expected['winner'] or any(abs(float(a) - b[1]) > ATOL for a, b in zip(top.values, expected['top2'], strict=True)):
                raise RuntimeError(f'source trace gate failed for {cell["boundary_id"]} offset {end}')
            comparison = {'cell_id': cell['id'], 'family': source['source_panel'],
                          'native': saved['native']['binding'], 'changed': saved['changed']['binding'],
                          'source_trace_winner': expected['winner'], 'native_winner': int(top.indices[0]),
                          'source_trace_top2_max_abs': max(abs(float(a) - b[1]) for a, b in zip(top.values, expected['top2'], strict=True))}
            if 'exact_native' in saved:
                diff = float((native - saved['exact_native']['logits']).abs().max())
                if diff > ATOL or int(saved['exact_native']['logits'].argmax()) != int(native.argmax()):
                    raise RuntimeError(f'full vocabulary prefix path gate failed: {diff}')
                comparison['exact_native'] = saved['exact_native']['binding']
                comparison['full_vocab_max_abs'] = diff
            comparisons.append(comparison)
    finally:
        handle.remove()
        write_new(args.output / 'terminal.json', {'schema': 'recurrence_phase_decision.gate_terminal.v1',
            'status': 'candidate_complete' if len(comparisons) == len(cells) else 'technical_invalid',
            'model': args.model, 'model_forwards': calls, 'elapsed_seconds': time.monotonic() - start,
            'identity': identity, 'comparisons': comparisons})
    print(json.dumps({'model': args.model, 'status': 'candidate_complete', 'forwards': calls,
                      'families': len(comparisons)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--model', choices=('tied', 'untied'), required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-forwards', type=int, default=8)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
