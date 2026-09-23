"""Source-bound selection and independent readback for the three-image assay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from probes.training_set_completion.artifacts import literal_binding as binding
from probes.training_set_completion.recurrence_census.prepare import _rows_from_tokens
from probes.training_set_completion.recurrence_transition_readback import aligned, first_fork, verified

SOURCE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural')
UNIT = Path('research/experiments/2026-09-23-recurrence-local-prefix-content-transfer')


def select():
    panel = json.loads((SOURCE / 'panel.json').read_text())
    order = json.loads((SOURCE / 'event-selection-order.json').read_text())['image_ids']
    lookup = {}
    for group in panel['groups']:
        for batch, case in enumerate(group['cases']):
            image_id = int(Path(case['image_path']).stem)
            assert image_id not in lookup, image_id
            lookup[image_id] = group, batch
    loaded, examined, selected = {}, [], []
    for rank, image_id in enumerate(order):
        if image_id in (7511, 269858):
            continue
        group, batch = lookup[image_id]
        key = group['key']
        folder = SOURCE / 'runtime/untied-original' / key
        if key not in loaded:
            receipt = json.loads((folder / 'receipt.json').read_text())
            loaded[key] = receipt, json.loads(verified(receipt['raw']).read_text())
        receipt, raw = loaded[key]
        sample = raw['rows'][batch]
        assert int(sample['image_id']) == image_id
        tokens = sample['token_ids']
        rows = _rows_from_tokens(tokens)
        runs = []
        for row in rows:
            seq = tokens[row['start']:row['end']]
            if runs and runs[-1]['tokens'] == seq and rows[runs[-1]['last']]['end'] == row['start']:
                runs[-1]['last'] = row['row_index']
            else:
                runs.append(dict(first=row['row_index'], last=row['row_index'], tokens=seq))
        eligible = []
        for run in runs:
            if run['last'] - run['first'] + 1 < 3 or run['last'] + 1 >= len(rows):
                continue
            previous, successor = rows[run['last']], rows[run['last'] + 1]
            fork = first_fork(previous, successor, tokens)
            if previous['end'] == successor['start'] and fork and fork['role'] in ('x1', 'y1', 'x2', 'y2'):
                eligible.append((run, successor, fork))
        examined.append(dict(image_id=image_id, source_order_rank=rank, group=key, batch_index=batch,
                             raw=receipt['raw'], eligible_run_count=len(eligible)))
        if not eligible:
            continue
        run, successor, fork = eligible[0]
        donor = rows[run['first'] + 1]
        count = fork['offset'] - 1
        action = successor['start'] + fork['offset']
        widths = list(map(len, receipt['input_identity']['prompt_token_ids']))
        width = max(widths)
        src = [donor['start'], donor['start'] + count]
        dst = [successor['start'], successor['start'] + count]
        assert count > 0 and src[1] <= dst[0] and dst[1] == action - 1
        assert tokens[slice(*src)] == tokens[slice(*dst)]
        trace = json.loads(verified(receipt['trace']).read_text())
        _, steps = aligned(raw, trace, batch, image_id)
        step = steps[action]
        assert step['raw_winners'][batch] == fork['new_token']
        selected.append(dict(
            image_id=image_id, source_order_rank=rank, group=key, batch_index=batch, row_id=sample['row_id'],
            run_rows=[run['first'], run['last']], donor_row=donor['row_index'], recipient_row=successor['row_index'],
            role=fork['role'], raw_action=action, row_difference_offset=fork['offset'], prefix_count=count,
            donor_prefix_raw=src, destination_prefix_raw=dst, prompt_width=width, prompt_lengths=widths,
            query=width + action - 1, donor_prefix_physical=[x + width for x in src],
            destination_prefix_physical=[x + width for x in dst], prefix_tokens=tokens[slice(*src)],
            repeat_token=fork['old_token'], native_token=fork['new_token'], repeat_bin=fork['old_token'] - 151670,
            native_bin=fork['new_token'] - 151670, stop=sample['stop'], raw=receipt['raw'], trace=receipt['trace'],
            runtime_receipt=binding(folder / 'receipt.json'), image=binding(Path(group['cases'][batch]['image_path'])),
            saved_trace=dict(winner=step['raw_winners'][batch], runnerup=step['raw_runnerups'][batch],
                             top2=step['raw_top2'][batch], logsumexp=step['logsumexp'][batch]),
            model_identity_sha256=hashlib.sha256(json.dumps(receipt['identity'], sort_keys=True).encode()).hexdigest()))
        if len(selected) == 3:
            break
    assert [x['image_id'] for x in selected] == [14038, 351017, 417044]
    assert [x['prefix_count'] for x in selected] == [3, 6, 5]
    return dict(schema_version=1, unit_id=UNIT.name, condition='untied-original', panel=binding(SOURCE / 'panel.json'),
                event_selection_order=binding(SOURCE / 'event-selection-order.json'),
                selection_rule='First three source-order images excluding7511/269858; earliest adjacent coordinate difference after>=3exactrows; second-row donor; exclude query self-key.',
                examined=examined, cases=selected,
                budget=dict(model_forwards=9, vision_forwards=9, gpu=4, wall_seconds_from_setup=900, tensor_bytes=96*1024*1024),
                output_root='/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/' + UNIT.name)


def analyze(attempt):
    import torch
    from src.qwen.input_identity import tensor_hash

    selection = json.loads((UNIT / 'selection.json').read_text())
    assert select() == selection, 'source selection no longer reproduces'
    read = lambda p: json.loads(p.read_text())
    receipt = read(attempt / 'receipt.json')
    manifest = read(attempt / 'source-to-cell.json')
    producer_result = read(attempt / 'result.json')
    assert receipt['status'] == 'candidate-complete'
    assert receipt['model_forwards'] == receipt['vision_forwards'] == 9
    assert receipt['full_sdpa_calls'] == 252 and receipt['selected_sdpa_calls'] == 168
    assert receipt['completed_cells'] == [f"{c['image_id']}:{name}" for c in selection['cases']
                                         for name in ('native', 'identity', 'donor_content')]
    assert receipt['elapsed_seconds'] <= 900 and receipt['tensor_bytes'] <= 96 * 1024 * 1024
    assert not Path(f"/proc/{receipt['pid']}").exists(), 'producer still running'
    assert manifest['status'] == 'frozen-before-forward'
    paths = set()
    historical_loader = manifest['model']['source_receipt_identity']['loader_source']
    current_loader = manifest['model']['loaded_identity']['loader_source']
    loader_capture = next(x['capture'] for x in manifest['source_captures'] if x['source'] == current_loader)
    assert all(historical_loader[k] == current_loader[k] == loader_capture[k] for k in ('sha256', 'size_bytes'))
    verified(current_loader)
    verified(loader_capture)
    relocated = []

    def verify_tree(obj):
        if isinstance(obj, dict):
            if {'path', 'sha256', 'size_bytes'} <= obj.keys():
                if obj == historical_loader and not Path(obj['path']).exists():
                    # The frozen loader contract permits only this same-byte path relocation.
                    relocated.append(dict(historical=obj, current=current_loader, retained=loader_capture))
                else:
                    verified(obj)
                paths.add(obj['path'])
            else:
                for child in obj.values():
                    verify_tree(child)
        elif isinstance(obj, list):
            for child in obj:
                verify_tree(child)

    for obj in (selection, receipt, manifest, producer_result):
        verify_tree(obj)

    def rotation(pre, cos, sin):
        # Independent complex multiplication; pre is [prefix,KV,dim].
        z = pre.permute(1, 0, 2).double()
        half = z.shape[-1] // 2
        phase = torch.complex(cos[:, :half].double(), sin[:, :half].double())
        value = torch.complex(z[..., :half], z[..., half:]) * phase
        return torch.cat((value.real, value.imag), -1).repeat_interleave(2, 0)

    summaries = []
    all_errors = []
    for case in selection['cases']:
        b, query = case['batch_index'], case['query']
        folder = attempt / 'cases' / str(case['image_id'])
        order = ('native', 'identity', 'donor_content')
        cells = {name: torch.load(folder / f'{name}.pt', map_location='cpu', weights_only=True) for name in order}
        checks = {name: read(folder / f'{name}-checks.json') for name in order}
        case_result = read(folder / 'result.json')
        verify_tree(case_result)
        native = cells['native']['full_logits']
        assert native.shape == (4, 152670) and native.dtype == torch.float32
        original = native[b].double()
        values, ids = original.topk(2)
        saved = case['saved_trace']
        assert ids.tolist() == [saved['winner'], saved['runnerup']]
        native_error = max(float((values - torch.tensor(saved['top2'], dtype=torch.float64)).abs().max()),
                           abs(float(original.logsumexp(0)) - saved['logsumexp']))
        identity_error = float((cells['identity']['full_logits'] - native).abs().max())
        assert cells['identity']['full_logits'][b].argmax() == native[b].argmax()
        errors = [native_error, identity_error]
        per_cell = {}
        expected_mask = torch.ones(1, query + 1, dtype=torch.bool)
        expected_mask[:, :case['prompt_width'] - case['prompt_lengths'][b]] = False
        for name in order:
            payload, check = cells[name], checks[name]
            verify_tree(check)
            assert payload['image_id'] == case['image_id'] and payload['cell'] == name
            assert len(payload['layers']) == len(check['layers']) == 28
            logits = payload['full_logits']
            assert logits.shape == native.shape and logits.dtype == torch.float32 and torch.isfinite(logits).all()
            companions = [index for index in range(4) if index != b]
            companion_error = float((logits[companions] - native[companions]).abs().max())
            errors.append(companion_error)
            assert check['model_call_count'] == check['vision_call_count'] == 1
            assert check['text_sdpa_call_count'] == 28 and check['input_unchanged']
            assert check['fresh_cache_exact'] and check['cache_sequence_length'] == query + 1
            assert check['lm_head_output_equals_returned_logits'] and check['lm_head_output_shape'] == [4, 1, 152670]
            for i in range(28):
                layer = payload['layers'][str(i)]
                reference = cells['native']['layers'][str(i)]
                gates = check['layers'][i]
                assert gates['sdpa_calls'] == 1 and gates['o_proj_consumed']
                assert all(gates[k] for k in ('full_inputs_unchanged', 'off_target_output_exact',
                           'selected_nonprefix_kv_exact', 'selected_q_exact', 'selected_mask_exact',
                           'layer_attention_consumed', 'cache_empty_at_entry'))
                assert gates['cache_position'] == list(range(query + 1))
                assert gates['full_input_hashes_before'] == gates['full_input_hashes_after']
                assert gates['selected_call'] == (name != 'native')
                assert torch.equal(layer['mask_row'], expected_mask)
                for key in ('donor_pre_k', 'donor_v', 'destination_pre_k', 'destination_v',
                            'donor_cos', 'donor_sin', 'destination_cos', 'destination_sin',
                            'position_ids_donor', 'position_ids_destination', 'position_ids_query'):
                    assert torch.equal(layer[key], reference[key]), (case['image_id'], name, i, key)
                before_k = rotation(layer['destination_pre_k'], layer['destination_cos'], layer['destination_sin'])
                before_v = layer['destination_v'].permute(1, 0, 2).repeat_interleave(2, 0)
                errors.append(float((layer['selected_k_before'] - before_k).abs().max()))
                assert torch.equal(layer['selected_v_before'], before_v)
                if name == 'native':
                    assert torch.equal(layer['selected_k_before'], layer['selected_k_used'])
                    assert torch.equal(layer['selected_v_before'], layer['selected_v_used'])
                    consumed = layer['full_output_target']
                else:
                    consumed = layer['selected_output_target']
                    if name == 'identity':
                        errors.append(float((consumed - layer['full_output_target']).abs().max()))
                        assert gates['selected_k_before_hash'] == gates['selected_k_used_hash']
                        assert gates['selected_v_before_hash'] == gates['selected_v_used_hash']
                        assert torch.equal(layer['selected_k_before'], layer['selected_k_used'])
                        assert torch.equal(layer['selected_v_before'], layer['selected_v_used'])
                    else:
                        used_k = rotation(layer['donor_pre_k'], layer['destination_cos'], layer['destination_sin'])
                        used_v = layer['donor_v'].permute(1, 0, 2).repeat_interleave(2, 0)
                        errors.extend([float((layer['selected_k_before'] - before_k).abs().max()),
                                       float((layer['selected_k_used'] - used_k).abs().max())])
                        assert torch.equal(layer['selected_v_before'], before_v)
                        assert torch.equal(layer['selected_v_used'], used_v)
                assert torch.equal(layer['o_proj_input'], consumed.flatten())
                assert tensor_hash(layer['o_proj_input']) == gates['o_proj_input_hash']
                if i == 0:
                    assert torch.equal(layer['donor_pre_k'], layer['destination_pre_k'])
                    assert torch.equal(layer['donor_v'], layer['destination_v'])
                if i == 1:
                    errors.append(float((layer['incoming_layer1_target'] - reference['incoming_layer1_target']).abs().max()))
            scores = logits[b].double()
            top, ids = scores.topk(10)
            prob = scores.softmax(0)
            gap = float(top[0] - top[1])
            winner = int(ids[0])
            per_cell[name] = dict(winner_token=winner, winner_bin=winner - 151670, gap=gap,
                repeat_probability=float(prob[case['repeat_token']]), native_probability=float(prob[case['native_token']]),
                repeat_rank=int((scores > scores[case['repeat_token']]).sum()) + 1,
                repeat_minus_native_logit=float(scores[case['repeat_token']] - scores[case['native_token']]),
                top10=[dict(token=int(token), logit=float(value)) for token, value in zip(ids, top)],
                classification='inconclusive' if gap <= .001 else 'repeat' if winner == case['repeat_token'] else
                    'native' if winner == case['native_token'] else 'third_coordinate' if 151670 <= winner < 152670 else 'noncoordinate')
        assert max(errors) <= 2e-4, (case['image_id'], max(errors))
        restored = per_cell['donor_content']['classification'] == 'repeat'
        assert restored == case_result['restores_repeated_coordinate']
        all_errors.extend(errors)
        changed_layers = [i for i in range(28) if any(
            not torch.equal(cells['donor_content']['layers'][str(i)][f'selected_{kind}_before'],
                            cells['donor_content']['layers'][str(i)][f'selected_{kind}_used'])
            for kind in ('k', 'v'))]
        assert changed_layers == list(range(1, 28))
        summaries.append(dict(image_id=case['image_id'], role=case['role'], repeat_bin=case['repeat_bin'],
                              cells=per_cell, max_checked_error=max(errors), restored=restored,
                              changed_prefix_layers=changed_layers))
    count = sum(x['restored'] for x in summaries)
    assert count == producer_result['restoration_count'] == receipt['restoration_count']
    return dict(status='independently-qualified', selection=binding(UNIT / 'selection.json'),
                receipt=binding(attempt / 'receipt.json'), manifest=binding(attempt / 'source-to-cell.json'),
                verified_binding_count=len(paths), model_forwards=9, vision_forwards=9,
                historical_loader_locator_substitutions=relocated,
                max_checked_error=max(all_errors), restoration_count=count, case_count=3,
                prediction_passed=count >= 2, cases=summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection-out', type=Path)
    parser.add_argument('--attempt', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.selection_out and not args.attempt:
        result = select()
        args.selection_out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(selection=binding(args.selection_out), cases=[{k: c[k] for k in
            ('image_id', 'query', 'donor_prefix_physical', 'destination_prefix_physical', 'prefix_count')}
            for c in result['cases']], examined=len(result['examined'])), indent=2))
    elif args.attempt and args.out and not args.selection_out:
        result = analyze(args.attempt)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
    else:
        parser.error('choose --selection-out or --attempt plus --out')
