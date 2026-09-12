"""Frozen coordinate-branch escape/return probe; no training or decode processor."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

from probes.dora_owner_learning.candidate_opportunity import score
from probes.dora_owner_learning.entrance_ce_eval import owner_change
from probes.dora_owner_learning.route_access import checkpoint_config
from probes.source_rweak_row_cross.run import native_record
from src.data.geometry import coord_bins_to_pixel_xyxy, iou_xyxy

PRIOR = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-branch-vs-repeat-event')
OUTPUT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/dynamics')
EOS, ROW_START, BOX_START, ROW_END, COORD, CAP = 151645, 151646, 151648, 151649, 151670, 3084
IDS = ('351017', '417044', '477415')
SOURCE_HASHES = {
    PRIOR / 'endpoint-preparation/packet.json': '560006e73f3f0fc416e7d58751fe96c936aba9478fb6b320ab6118db2bcd5053',
    PRIOR / 'input-preparation/candidate_manifest.json': '2870e777965007b5b06487fbb33b8408d1a992f4c90e2b4bd1f0f564c1e4aa3b',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded(value))
    require(json.loads(path.read_text()) == value, 'publication cold readback')


def rows(ids):
    """Strict pre-intervention row split, not a substitute for native output parsing."""
    out, cursor = [], 0
    while cursor < len(ids) and ids[cursor] != EOS:
        require(ids[cursor] == ROW_START, 'prefix row opener')
        try:
            end = ids.index(ROW_END, cursor) + 1
        except ValueError:
            break
        row = ids[cursor:end]
        require(BOX_START in row, 'prefix box opener')
        offset = row.index(BOX_START) + 1
        require(len(row[offset:-1]) == 4 and all(COORD <= x < COORD + 1000 for x in row[offset:-1]), 'prefix coordinate arity/range')
        out.append(dict(start=cursor, end=end, ids=row, x1_position=cursor + offset,
                        coord_bins=[x - COORD for x in row[offset:-1]]))
        cursor = end
    return out


def pixel_box(row, case):
    return coord_bins_to_pixel_xyxy(row['coord_bins'], image_width=case['image_width'],
                                    image_height=case['image_height'], field='frozen_row')


def choose_boundaries(stable_ids, h_ids, case):
    require(stable_ids[:len(h_ids)] == h_ids, 'original h not native prefix')
    hrows = rows(h_ids)
    require(hrows and hrows[-1]['end'] == len(h_ids), 'h is not complete rows')
    # Only consume the first successor; a capped later tail may contain invalid syntax.
    successor_end = stable_ids.index(ROW_END, len(h_ids)) + 1
    allrows = rows(stable_ids[:successor_end])
    repeated = allrows[len(hrows)]
    prior = [pixel_box(row, case) for row in hrows]
    require(any(iou_xyxy(pixel_box(repeated, case), b) > .95 for b in prior), 'selected h has no immediate strict repeat')
    eligible = [i for i, row in enumerate(hrows) if not any(
        iou_xyxy(pixel_box(row, case), b) > .95 for b in prior[:i])]
    require(eligible, 'no saved-native nonrepeat control')
    result = []
    for kind, row in (('repeat', repeated), ('control', hrows[eligible[-1]])):
        result.append(dict(boundary_id=kind, history_ids=stable_ids[:row['start']],
                           history_sha256=digest(stable_ids[:row['start']]), target_row=row,
                           intervention_position=row['x1_position'], expected_first_row_repeat=kind == 'repeat'))
    return result


def select_alternatives(coordinate_logits, original):
    """Predeclared deterministic ranking, preserving overlapping family denominators."""
    require(len(coordinate_logits) == 1000 and COORD <= original < COORD + 1000, 'coordinate selection schema')
    import math
    require(all(math.isfinite(x) for x in coordinate_logits), 'nonfinite coordinate logits')
    native_order = sorted(range(1000), key=lambda i: (-coordinate_logits[i], i))
    high = [COORD + i for i in native_order if COORD + i != original][:3]
    near = sorted((COORD + i for i in range(1000) if COORD + i != original),
                  key=lambda token: (abs(token - original), token))[:3]
    tokens = [original] + sorted(set(high + near))
    return [dict(token_id=token, coordinate_bin=token - COORD,
                 families=(['native_self'] if token == original else []) +
                          (['high_probability'] if token in high else []) +
                          (['geometric_neighbor'] if token in near else []),
                 coordinate_rank=native_order.index(token - COORD) + 1,
                 displacement_bins=token - original,
                 native_minus_alternative_logit=coordinate_logits[original - COORD] - coordinate_logits[token - COORD])
            for token in tokens]


def build_packet():
    for path, expected in SOURCE_HASHES.items():
        require(file_hash(path) == expected, f'source changed: {path}')
    endpoint = json.loads((PRIOR / 'endpoint-preparation/packet.json').read_text())
    manifest = json.loads((PRIOR / 'input-preparation/candidate_manifest.json').read_text())
    source_files = {str(p): h for p, h in SOURCE_HASHES.items()}
    source_files[str(Path(__file__).resolve())] = file_hash(__file__)
    cases = []
    for cid in IDS:
        p = next(x for x in manifest['positives'] if x['case_id'] == cid)
        e = next(x for x in endpoint['eval_records'] if str(x['image_id']) == cid)
        require(e['prompt_token_ids'] == p['prompt']['token_ids'], 'saved prompt differs')
        source_files[p['image']['image_path']] = p['image']['image_sha256']
        cases.append(dict(case_id=cid, example_id=e['example_id'], source_case=e['case'], golden=e['golden'],
                          prompt_token_ids=e['prompt_token_ids'], stable_ids=e['stable_ids'],
                          stable_stop=e['stable_score']['stop_reason'], stable_score=e['stable_score'],
                          boundaries=choose_boundaries(e['stable_ids'], p['h']['token_ids'], e['case'])))
    for identity in (endpoint['model']['current_adapter'], endpoint['model']['source_embedding']):
        for item in identity['files']:
            source_files[str(Path(identity['root']) / item['relative_path'])] = item['sha256']
    packet = dict(schema='parallel_owner.dynamics.v1', status='frozen_cpu_prepared',
                  model=endpoint['model'], config=endpoint['config'], source_files=source_files, cases=cases,
                  policy=dict(temperature=0., top_p=1., top_k=0, repetition_penalty=1., use_model_defaults=False),
                  design=dict(coordinate_slot='x1', high_probability_count=3, geometric_neighbor_count=3,
                              coordinate_legal_range=[0, 999], continuation_total_cap=CAP,
                              full_suffix=True, maximum_unique_cells=42, maximum_natural_anchors=3,
                              maximum_score_replays=6, cell_family_memberships=42,
                              duplicate_definition='class-blind native-pixel IoU > .95; each later valid row once',
                              control_rule='latest valid nonrepeat complete row before original h, selected only from saved Stable50',
                              credit='history and mixed intervention-containing row earn no autonomous owner credit'))
    validate_packet(packet)
    return packet


def validate_packet(packet):
    require(packet['schema'] == 'parallel_owner.dynamics.v1' and packet['status'] == 'frozen_cpu_prepared', 'packet schema/status')
    require([c['case_id'] for c in packet['cases']] == list(IDS), 'frozen case identities')
    require(packet['design']['continuation_total_cap'] == CAP and packet['design']['full_suffix'], 'decode cap changed')
    require(packet['policy'] == dict(temperature=0., top_p=1., top_k=0, repetition_penalty=1., use_model_defaults=False), 'native policy changed')
    for path, expected in packet['source_files'].items():
        require(file_hash(path) == expected, f'source changed: {path}')
    for case in packet['cases']:
        for boundary in case['boundaries']:
            pos = boundary['intervention_position']
            require(pos == len(boundary['history_ids']) + boundary['target_row']['ids'].index(BOX_START) + 1, 'intervention position')
            require(case['stable_ids'][:len(boundary['history_ids'])] == boundary['history_ids'], 'history identity')
            require(digest(boundary['history_ids']) == boundary['history_sha256'], 'history hash')
            require(COORD <= case['stable_ids'][pos] < COORD + 1000, 'target is not coordinate')


def behavior(text, parsed, *, history_text, intervention_char_end, stop):
    """Partition native parser output without hiding invalid/intervention-bearing rows."""
    ordered = sorted(parsed['pred'] + parsed['dropped_predictions'], key=lambda r: (r['char_start'], r['char_end']))
    prior, outcomes = [], []
    for row in ordered:
        supplied = row['char_end'] <= len(history_text)
        valid = 'bbox' in row
        if supplied:
            if valid:
                prior.append(row['bbox'])
            continue
        mixed = row['char_start'] < intervention_char_end
        overlap = max((iou_xyxy(row['bbox'], b) for b in prior), default=0.) if valid else None
        outcomes.append(dict(generated_order=row.get('generated_order'), char_start=row['char_start'],
                             char_end=row['char_end'], mixed_intervened_row=mixed, valid=valid,
                             bbox=row.get('bbox'), description=row.get('description'),
                             reason=row.get('reason'), strict_repeat=bool(valid and overlap > .95),
                             maximum_iou_to_earlier_valid=overlap,
                             complete_object_row=row['raw_span_text'].startswith('<|object_ref_start|>') and
                                                 row['raw_span_text'].endswith('<|box_end|>')))
        if valid:
            prior.append(row['bbox'])
    target = next((r for r in outcomes if r['mixed_intervened_row']), None)
    autonomous = [r for r in outcomes if not r['mixed_intervened_row']]
    successor = next((r for r in autonomous if r['complete_object_row']), None)
    drops = [r for r in outcomes if not r['valid']]
    complete = [r for r in outcomes if r['complete_object_row']]
    return dict(first_intervened_row=target, first_autonomous_complete_row=successor,
                first_two_complete_rows=complete[:2], row_ledger=outcomes,
                posthistory_valid_rows=sum(r['valid'] for r in outcomes),
                autonomous_valid_rows=sum(r['valid'] for r in autonomous),
                posthistory_strict_repeats=sum(r['strict_repeat'] for r in outcomes),
                autonomous_strict_repeats=sum(r['strict_repeat'] for r in autonomous),
                geometry_invalid_rows=sum(r['reason'] == 'geometry_invalid' for r in drops),
                other_parser_drops=sum(r['reason'] != 'geometry_invalid' for r in drops),
                raw_starts_after_history=text[len(history_text):].count('<|object_ref_start|>'),
                native_stop=stop, cap=stop == 'length')


def parse_cell(qwen, case, boundary, prefix, free, stop):
    ids = prefix + free
    text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
    parsed = native_record(text, case['source_case'], case['golden'], stop)
    history_text = qwen.tokenizer.decode(boundary['history_ids'], skip_special_tokens=False)
    intervention_text = qwen.tokenizer.decode(prefix, skip_special_tokens=False)
    ledger = behavior(text, parsed, history_text=history_text, intervention_char_end=len(intervention_text), stop=stop)
    # Full matching is descriptive. Autonomous matching excludes all supplied or mixed rows.
    free_parsed = dict(parsed, pred=[p for p in parsed['pred'] if p['char_start'] >= len(intervention_text)],
                       dropped_predictions=[p for p in parsed['dropped_predictions'] if p['char_start'] >= len(intervention_text)])
    free_parsed['dropped_prediction_count'] = len(free_parsed['dropped_predictions'])
    free_parsed['valid_prediction_count'] = len(free_parsed['pred'])
    return dict(action_ids=ids, free_ids=free, text=text, parsed_full=parsed, behavior=ledger,
                full_score_descriptive=score(parsed, seed=None, length=len(ids), stop=stop),
                autonomous_score=score(free_parsed, seed=None, length=len(free), stop=stop))


def run(packet_path, out_dir, case_id, slice_only=False, reuse_slice=None):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '4', 'exclusive physical GPU4 required')
    packet = json.loads(packet_path.read_text())
    validate_packet(packet)
    case = next(c for c in packet['cases'] if c['case_id'] == case_id)
    require(not out_dir.exists(), 'run output exists; do not overwrite or relaunch')
    out_dir.mkdir(parents=True)
    counters = dict(model_forwards=0, image_forwards=0, generation_calls=0, score_replays=0, generated_tokens=0)
    terminal = dict(schema='parallel_owner.dynamics.terminal.v1', status='running', case_id=case_id,
                    packet_sha256=file_hash(packet_path), physical_gpu=4, slice_only=slice_only)
    reused = []
    if reuse_slice is not None:
        prior_terminal = json.loads((reuse_slice / 'terminal.json').read_text())
        require(not slice_only and prior_terminal['status'] == 'complete' and prior_terminal['slice_only'] and
                prior_terminal['case_id'] == case_id and prior_terminal['packet_sha256'] == file_hash(packet_path), 'reused slice identity')
        require(file_hash(reuse_slice / 'records.jsonl') == prior_terminal['records_sha256'], 'reused slice records changed')
        reused = [json.loads(line) for line in (reuse_slice / 'records.jsonl').read_text().splitlines()]
        require(len(reused) == 4 and sum(r['kind'] == 'cell' for r in reused) == 2, 'reused slice counts')
        terminal['reused_slice'] = dict(path=str(reuse_slice / 'terminal.json'), sha256=file_hash(reuse_slice / 'terminal.json'),
                                       records_sha256=prior_terminal['records_sha256'], counters=prior_terminal)
    publish(out_dir / 'launch.json', dict(terminal, pid=os.getpid(), started_unix=time.time()))
    handles, records = [], []
    torch = None
    started = time.monotonic()
    error = None
    try:
        import torch as torch_module
        torch = torch_module
        from probes.dora_owner_learning.runtime import load_policy
        from probes.source_rweak_row_cross.run import build_requests
        from src.config.inference import InferConfig
        from src.qwen.generation import NativeGenerationPolicy, generate_continuations
        from src.qwen.native import prepare_native_inputs, prepare_replay
        config = checkpoint_config(InferConfig.model_validate(packet['config']), packet['model']['current_adapter']['root'])
        qwen, identity = load_policy(config, device=torch.device('cuda:0'))
        require(identity['effective_settings']['observed_attn_implementation'] == 'sdpa', 'attention mismatch')
        require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32'], 'dtype mismatch')
        require(identity['model_identity']['adapter']['adapter_path'] == packet['model']['current_adapter']['root'], 'adapter mismatch')
        publish(out_dir / 'model.json', identity)
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)
        visual = [v for k, v in qwen.model.named_modules() if k.endswith('visual')]
        require(len(visual) == 1, 'visual owner ambiguity')
        handles.append(qwen.model.register_forward_pre_hook(lambda *_: counters.__setitem__('model_forwards', counters['model_forwards'] + 1)))
        handles.append(visual[0].register_forward_pre_hook(lambda *_: counters.__setitem__('image_forwards', counters['image_forwards'] + 1)))
        requests, _ = build_requests(qwen, packet['config'], [case['source_case']])
        batch = prepare_native_inputs(qwen.processor, requests, device=torch.device('cuda:0'), record_media_identity=True)
        require(list(batch.prompt_token_ids[0]) == case['prompt_token_ids'], 'live prompt identity')
        publish(out_dir / 'batch.json', dict(prompt_token_ids=list(batch.prompt_token_ids[0]), image_grid=list(batch.image_grids[0]), request_id=batch.request_ids[0]))
        policy = NativeGenerationPolicy(**packet['policy'])

        def generate(prefix):
            budget = CAP - len(prefix)
            value = generate_continuations(qwen.model, batch, extensions=[prefix], budgets=[budget],
                                           eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id,
                                           policy=policy, trace='none')[0]
            tokens = list(value.token_ids)
            require((value.stop_reason == 'length' and len(tokens) == budget and EOS not in tokens) or
                    (value.stop_reason == 'im_end' and tokens and tokens[-1] == EOS and EOS not in tokens[:-1]), 'stop/token mismatch')
            counters['generation_calls'] += 1
            counters['generated_tokens'] += len(tokens)
            return tokens, value.stop_reason

        with (out_dir / 'records.jsonl').open('x') as stream, torch.inference_mode():
            def record(value):
                stream.write(encoded(value).decode()); stream.flush(); os.fsync(stream.fileno())
                records.append(value)
            if reused:
                natural_record_saved = next(r for r in reused if r['kind'] == 'natural_anchor')
                natural, natural_stop = natural_record_saved['action_ids'], natural_record_saved['stop_reason']
                record(natural_record_saved)
            else:
                natural, natural_stop = generate([])
                record(dict(kind='natural_anchor', case_id=case_id, action_ids=natural, stop_reason=natural_stop,
                            exact_saved_match=natural == case['stable_ids']))
            require(natural == case['stable_ids'] and natural_stop == case['stable_stop'], 'fresh natural Stable50 mismatch')
            for boundary in case['boundaries'][:1] if slice_only else case['boundaries']:
                pos = boundary['intervention_position']
                original = case['stable_ids'][pos]
                saved_selection = next((r for r in reused if r['kind'] == 'selection' and r['boundary_id'] == boundary['boundary_id']), None)
                if saved_selection is not None:
                    choices = saved_selection['choices']; record(saved_selection)
                else:
                    continuation = case['stable_ids'][:pos + 1]
                    replay = prepare_replay(qwen.model, batch.inputs, prompt_token_ids=case['prompt_token_ids'], continuation_token_ids=continuation)
                    logits = replay.aligned_logits(qwen.model(**replay.inputs).logits)[-1].float()
                    counters['score_replays'] += 1
                    require(int(logits.argmax()) == original, 'replay/greedy earliest coordinate mismatch')
                    choices = select_alternatives(logits[COORD:COORD + 1000].cpu().tolist(), original)
                    lognorm = float(torch.logsumexp(logits, dim=0))
                    for choice in choices:
                        choice.update(logprob=float(logits[choice['token_id']]) - lognorm,
                                      native_logprob=float(logits[original]) - lognorm,
                                      native_token_id=original)
                    record(dict(kind='selection', boundary_id=boundary['boundary_id'], intervention_position=pos,
                                choices=choices, family_counts=dict(Counter(f for c in choices for f in c['families']))))
                    del logits, replay
                selected = choices if not slice_only else [choices[0], min(
                    (c for c in choices if 'high_probability' in c['families']), key=lambda c: c['coordinate_rank'])]
                for choice in selected:
                    saved_cell = next((r for r in reused if r['kind'] == 'cell' and r['boundary_id'] == boundary['boundary_id'] and
                                       r['choice']['token_id'] == choice['token_id']), None)
                    if saved_cell is not None:
                        record(saved_cell)
                        continue
                    prefix = case['stable_ids'][:pos] + [choice['token_id']]
                    free, stop = generate(prefix)
                    result = parse_cell(qwen, case, boundary, prefix, free, stop)
                    value = dict(kind='cell', case_id=case_id, boundary_id=boundary['boundary_id'],
                                 choice=choice, prefix_ids=prefix, budget=CAP - len(prefix), stop_reason=stop, **result)
                    record(value)
                    if 'native_self' in choice['families']:
                        require(result['action_ids'] == natural, 'self intervention changed natural continuation')
                        require(result['behavior']['first_intervened_row']['strict_repeat'] == boundary['expected_first_row_repeat'], 'repeat/control admission differs')
        durable = [json.loads(line) for line in (out_dir / 'records.jsonl').read_text().splitlines()]
        require(durable == records, 'durable record readback')
        terminal.update(status='complete', exit_code=0, records_sha256=file_hash(out_dir / 'records.jsonl'),
                        record_count=len(records), cells=sum(r['kind'] == 'cell' for r in records))
    except BaseException as exc:
        error = exc
        terminal.update(status='failed', exit_code=1, error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    finally:
        for handle in handles:
            handle.remove()
        terminal.update(counters, elapsed_seconds=time.monotonic() - started,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if torch is not None and torch.cuda.is_available():
            terminal.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        publish(out_dir / 'terminal.json', terminal)
    if error:
        raise error
    return terminal


def reduce_runs(packet_path, run_dirs, output):
    packet = json.loads(packet_path.read_text()); validate_packet(packet)
    summaries, receipts = [], []
    for directory in run_dirs:
        terminal = json.loads((directory / 'terminal.json').read_text())
        require(terminal['status'] == 'complete' and not terminal['slice_only'], 'full case incomplete')
        require(terminal['packet_sha256'] == file_hash(packet_path), 'run packet differs')
        require(terminal['records_sha256'] == file_hash(directory / 'records.jsonl'), 'run records changed')
        records = [json.loads(line) for line in (directory / 'records.jsonl').read_text().splitlines()]
        cells = [r for r in records if r['kind'] == 'cell']
        for bid in ('repeat', 'control'):
            selected = [r for r in cells if r['boundary_id'] == bid]
            baseline = next(r for r in selected if 'native_self' in r['choice']['families'])
            for r in selected:
                b = r['behavior']
                summaries.append(dict(case_id=r['case_id'], boundary_id=bid, choice=r['choice'],
                    first_intervened_row=b['first_intervened_row'], first_autonomous_complete_row=b['first_autonomous_complete_row'],
                    autonomous_strict_repeats=b['autonomous_strict_repeats'], posthistory_strict_repeats=b['posthistory_strict_repeats'],
                    geometry_invalid_rows=b['geometry_invalid_rows'], other_parser_drops=b['other_parser_drops'],
                    raw_starts_after_history=b['raw_starts_after_history'], autonomous_valid_rows=b['autonomous_valid_rows'],
                    stop_reason=r['stop_reason'], autonomous_score=r['autonomous_score'],
                    owner_changes={k: owner_change(r['autonomous_score'][k]['owners'], baseline['autonomous_score'][k]['owners']) for k in ('50', '60', '80')}))
        receipts.append(dict(path=str(directory / 'terminal.json'), sha256=file_hash(directory / 'terminal.json'), **terminal))
    require(sorted(r['case_id'] for r in receipts) == sorted(IDS), 'full three-case denominator')
    family = Counter(f for s in summaries for f in s['choice']['families'])
    require(family == dict(native_self=6, high_probability=18, geometric_neighbor=18), 'family denominator differs')
    result = dict(schema='parallel_owner.dynamics.reduction.v1', packet_sha256=file_hash(packet_path),
                  unique_cells=len(summaries), family_memberships=dict(family), cases=summaries, receipts=receipts)
    publish(output, result)
    return dict(unique_cells=len(summaries), family_memberships=dict(family), output=str(output))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--packet', type=Path, default=OUTPUT / 'packet.json')
    execute = sub.add_parser('run'); execute.add_argument('--packet', type=Path, default=OUTPUT / 'packet.json')
    execute.add_argument('--case-id', choices=IDS, required=True); execute.add_argument('--out-dir', type=Path, required=True)
    execute.add_argument('--slice-only', action='store_true')
    execute.add_argument('--reuse-slice', type=Path)
    reduce = sub.add_parser('reduce'); reduce.add_argument('--packet', type=Path, default=OUTPUT / 'packet.json')
    reduce.add_argument('--run-dirs', type=Path, nargs=3, required=True); reduce.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        packet = build_packet(); publish(args.packet, packet)
        result = dict(packet=str(args.packet), sha256=file_hash(args.packet), boundaries=[
            dict(case_id=c['case_id'], boundaries=[dict(id=b['boundary_id'], history_tokens=len(b['history_ids']),
                 intervention_position=b['intervention_position']) for b in c['boundaries']]) for c in packet['cases']])
    elif args.command == 'run':
        result = run(args.packet, args.out_dir, args.case_id, args.slice_only, args.reuse_slice)
    else:
        result = reduce_runs(args.packet, args.run_dirs, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
