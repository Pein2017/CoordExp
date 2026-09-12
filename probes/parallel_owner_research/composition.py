"""Finite, CPU-only trustworthy-repair supply for the composition lane.

Mechanical selection never certifies physical omission.  Visual admission and
Stable50 conditional releases are separate, explicit later inputs.
"""
from __future__ import annotations

import argparse
import copy
from itertools import combinations
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from probes.dora_owner_learning.candidate_opportunity import (
    digest, file_hash, require, score, validate_parser,
)
from probes.dora_owner_learning.reward_rows import _gt_objects, _pred_objects
from src.artifacts import publish_json_exclusive
from src.data.geometry import iou_xyxy

BASE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
CASES = BASE / '2026-09-09-natural-candidate-opportunity/full-v2/cases.json'
PLAN = BASE / '2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/rloo/round-1/plan.json'
STABLE = BASE / '2026-09-10-selective-owner-learning-autonomous/positive7-support50-81/evaluation'
ACQUISITION_SELECTION = {351017: ('478719', '499060'), 417044: ('1082918', '1080038')}
RESIDUAL_SELECTION = {351017: ('478719', '667769'), 417044: ('1082918', '1083042')}
RESIDUAL_CANDIDATE_SHA256 = 'a39233b68b2fc40011c697559c12fb958421a4bb982d4682368217f18a0bd114'


def complete_row_span(action_ids, prediction, tokenizer):
    """Recover a literal complete sampled row, never a GT-authored replacement."""
    text = tokenizer.decode(action_ids, skip_special_tokens=False)
    start, end = prediction['char_start'], prediction['char_end']
    require(text[start:end] == prediction['raw_span_text'], 'sample row text boundary')
    prefix = tokenizer.encode(text[:start], add_special_tokens=False).ids
    target = tokenizer.encode(text[start:end], add_special_tokens=False).ids
    require(action_ids[:len(prefix)] == prefix, 'sample prefix token boundary')
    require(action_ids[len(prefix):len(prefix) + len(target)] == target,
            'sample target token boundary')
    require(target and target[0] == 151646 and target[-1] == 151649,
            'positive must be one complete object row')
    require(target.count(151646) == target.count(151649) == 1,
            'positive contains more than one row')
    return {'prefix_token_ids': prefix, 'target_token_ids': target,
            'prefix_token_ids_sha256': digest(prefix),
            'target_token_ids_sha256': digest(target)}


def qualify_sample(sample, stable_score):
    """A finite supply filter, not a training-label or usefulness certificate."""
    old = set(stable_score['50']['owners'])
    available = set(sample['50']['owners'])
    missing = sorted(available - old, key=int)
    return {
        'at_least_two_stable_missed': len(missing) >= 2,
        'stable_missed_owner_ids': missing,
        'retains_all_stable_owners': old <= available,
        'stable_owner_losses': sorted(old - available, key=int),
        'sample_burden_not_worse': all(
            sample[k] <= stable_score[k]
            for k in ('strict_repeats', 'parser_drops', 'invalid_predictions', 'cap')),
        'sample_fp_not_worse': sample['50']['fp'] <= stable_score['50']['fp'],
        'physical_omission_admitted': False,
    }


def geometry_record(parsed, owner):
    ids = [str(o['object_id']) for o in parsed['gt']]
    gt = _gt_objects(parsed, row_id=parsed['row_id'])
    category, box = gt[ids.index(str(owner))]
    predictions, _ = _pred_objects(parsed)
    ranked = sorted(({'pred_index': i, 'category': c, 'box': list(b),
                      'iou': iou_xyxy(box, b)}
                     for i, (c, b) in enumerate(predictions)),
                    key=lambda v: (-v['iou'], v['pred_index']))
    same = [p for p in ranked if p['category'] == category]
    return {'owner_id': str(owner), 'category': category, 'gt_box_pixels': list(box),
            'stable_best_any': ranked[0] if ranked else None,
            'stable_best_same': same[0] if same else None,
            'geometry_is_not_visual_admission': True}


def prepare_inventory(output_dir: Path):
    from tokenizers import Tokenizer

    require(not output_dir.exists(), 'occupied composition inventory output')
    sources = {}

    def read(path, expected=None):
        path = Path(path).resolve(strict=True)
        sha = file_hash(path)
        require(expected is None or sha == expected, f'source changed: {path}')
        sources[str(path)] = sha
        return json.loads(path.read_text())

    cases = read(CASES, '62a072ab9d8d56c638936666bfc76ec094b89047465e24e9f8b5338ba684a1fd')
    stable_rows = read(STABLE / 'consumer.json',
                       '27850345b25c2e1eb26e11fae9e58e4fe2be35e8294889a6f615f3a518c1cde3')
    stable_manifest = read(STABLE / 'manifest.json',
                           'c44f0673fc53f58403b446b93913f031b3cafc38d2a0641e64dfc1357d05a954')
    stable = {r['example_id']: r for r in stable_rows}
    images = {r['example_id']: r for r in stable_manifest['records']}
    plan = read(PLAN)
    require(digest({k: v for k, v in plan.items() if k != 'content_sha256'})
            == plan['content_sha256'], 'retained plan content identity')
    groups = {r['example_id']: r for r in plan['population']['groups']}
    tokenizer_path = Path(plan['model']['base_model_path']) / 'tokenizer.json'
    sources[str(tokenizer_path)] = file_hash(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    raw = {}
    for shard_source in plan['sources']['rollout_artifacts']:
        shard = read(shard_source['path'], shard_source['sha256'])
        raw.update({(r['example_id'], int(r['seed'])): r for r in shard['rollouts']})
    rows, pairs = [], []
    for case in sorted(cases, key=lambda c: int(c['image_id'])):
        key = case['example_id']
        anchor = stable[key]
        for sample in sorted(case['samples'], key=lambda s: s['seed']):
            # Start with exactly the old strong bank, not a new best-of-K selector.
            if not sample['comparison']['strong_joint_witness']:
                continue
            qualification = qualify_sample(sample, anchor['score'])
            if not qualification['at_least_two_stable_missed']:
                continue
            source = raw[key, int(sample['seed'])]
            action = next(a for a in groups[key]['actions'] if a['seed'] == sample['seed'])
            require(source['generated_token_ids'] == action['generated_token_ids'],
                    'retained raw/plan action identity')
            require(digest(source['predictions']) == action['parser_evidence_sha256'],
                    'retained parser evidence identity')
            require(images[key]['prompt_token_ids'] == source['prompt_token_ids'],
                    'Source/Stable50 prompt identity')
            parsed = anchor['parsed']
            validate_parser(source['generated_text'], source['predictions'],
                            parsed['image_width'], parsed['image_height'])
            sample_parsed = dict(parsed, pred=source['predictions']['predictions'],
                                 dropped_prediction_count=source['predictions']['dropped_prediction_count'])
            replay = score(sample_parsed, seed=sample['seed'],
                           length=sample['complete_token_length'], stop=sample['stop_reason'])
            require(replay['50'] == sample['50'], 'sample score replay')
            targets = []
            for owner in qualification['stable_missed_owner_ids']:
                match = next(m for m in sample['50']['matches'] if m['owner'] == owner)
                pred = source['predictions']['predictions'][match['pred_index']]
                row = geometry_record(parsed, owner)
                row.update(complete_row_span(action['action_token_ids'], pred, tokenizer))
                row.update(sample_row_index=match['pred_index'], sample_iou=match['iou'],
                           sample_category=pred['description'], sample_box_pixels=pred['bbox'],
                           target_text=pred['raw_span_text'])
                require(row['category'] == row['sample_category'], 'sample target class conflict')
                targets.append(row)
            record = {'example_id': key, 'image_id': int(case['image_id']),
                      'seed': sample['seed'], 'image_path': case['image_path'],
                      'stable_score': anchor['score'], 'sample_score': replay,
                      'qualification': qualification, 'targets': targets,
                      'image': images[key]['case'],
                      'prompt_token_ids': images[key]['prompt_token_ids'],
                      'stable_action_ids': anchor['action_ids'],
                      'sample_action_ids': action['action_token_ids'],
                      'sample_parsed': sample_parsed, 'stable_parsed': parsed}
            rows.append(record)
            for a, b in combinations(targets, 2):
                pairs.append({'image_id': record['image_id'], 'seed': sample['seed'],
                              'owners': [a['owner_id'], b['owner_id']],
                              'category_agreement': a['category'] == b['category'],
                              'target_pair_iou': iou_xyxy(a['gt_box_pixels'], b['gt_box_pixels'])})
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'parallel_owner_composition.inventory.v1',
              'status': 'cpu_supply_not_visual_or_conditional_admission',
              'sources': sources, 'source_case_count': len(cases),
              'image_ids': sorted({r['image_id'] for r in rows}),
              'sample_count': len(rows), 'pair_count': len(pairs),
              'records': rows, 'pairs': pairs}
    publish_json_exclusive(output_dir / 'inventory.json', result)
    summary = {k: result[k] for k in ('schema', 'status', 'source_case_count', 'image_ids',
                                     'sample_count', 'pair_count')}
    summary['inventory_sha256'] = file_hash(output_dir / 'inventory.json')
    publish_json_exclusive(output_dir / 'summary.json', summary)
    return summary


def prepare_cards(inventory_path: Path, selection_path: Path, output_dir: Path):
    from src.vis.api import render_prediction_comparison

    require(not output_dir.exists(), 'occupied composition cards output')
    inventory = json.loads(inventory_path.read_text())
    selection = json.loads(selection_path.read_text())
    selected = []
    left, right = [], []
    for item in selection['cases']:
        candidates = [r for r in inventory['records']
                      if r['image_id'] == item['image_id'] and r['seed'] == item['seed']]
        require(len(candidates) == 1, 'selection image/seed missing or ambiguous')
        row = candidates[0]
        owners = [item['A'], item['B']]
        require(len(set(owners)) == 2, 'A and B must be distinct owners')
        targets = [next(t for t in row['targets'] if t['owner_id'] == owner) for owner in owners]
        target_gt = [g for g in row['stable_parsed']['gt'] if str(g['object_id']) in owners]
        require(len(target_gt) == 2, 'target GT identity')
        pred_indices = {t[field]['pred_index'] for t in targets
                        for field in ('stable_best_any', 'stable_best_same') if t[field]}
        focus_left = dict(row['stable_parsed'], gt=target_gt,
                          pred=[p for i, p in enumerate(row['stable_parsed']['pred']) if i in pred_indices])
        focus_right = dict(row['sample_parsed'], gt=target_gt,
                           pred=[row['sample_parsed']['pred'][t['sample_row_index']] for t in targets])
        left.append(focus_left)
        right.append(focus_right)
        selected.append({**item, 'example_id': row['example_id'], 'image_path': row['image_path'],
                         'targets': targets, 'qualification': row['qualification'],
                         'stable_pred_count': len(row['stable_parsed']['pred']),
                         'displayed_stable_pred_indices': sorted(pred_indices),
                         'image': row['image'], 'prompt_token_ids': row['prompt_token_ids'],
                         'stable_action_ids': row['stable_action_ids'],
                         'stable_score': row['stable_score'], 'stable_parsed': row['stable_parsed']})
    output_dir.mkdir(parents=True, exist_ok=False)
    for side, rows_ in [('stable', left), ('sample', right)]:
        (output_dir / side).mkdir(exist_ok=False)
        # The renderer requires paired raw/scored views. These are deliberately
        # identical focused geometry excerpts, not recomputed model scores.
        for filename in ('gt_vs_pred.jsonl', 'gt_vs_pred_scored.jsonl'):
            with (output_dir / side / filename).open('x') as stream:
                for row in rows_:
                    stream.write(json.dumps(row) + '\n')
    result = render_prediction_comparison(
        output_dir / 'stable/gt_vs_pred_scored.jsonl',
        output_dir / 'sample/gt_vs_pred_scored.jsonl',
        output_dir / 'cards', left_label='Stable50 nearest boxes ONLY (not full output)',
        right_label='Two sampled witness rows ONLY (not full output)')
    packet = {'schema': 'parallel_owner_composition.visual_candidates.v1',
              'status': 'candidate_awaiting_lead_admission',
              'inventory': {'path': str(inventory_path.resolve()), 'sha256': file_hash(inventory_path)},
              'selection': {'path': str(selection_path.resolve()), 'sha256': file_hash(selection_path)},
              'visual_metric_boundary': 'Focused subsets only; renderer matching is not official metrics.',
              'visualization_manifest': str(result.manifest_path), 'cases': selected}
    publish_json_exclusive(output_dir / 'candidates.json', packet)
    return {'candidate_count': len(selected), 'candidates': str(output_dir / 'candidates.json'),
            'cards': [str(p) for p in result.image_paths]}


def prepare_acquisition(candidates_path: Path, output_dir: Path):
    from tokenizers import Tokenizer

    require(not output_dir.exists(), 'occupied acquisition preparation')
    candidates = json.loads(candidates_path.read_text())
    old_path = BASE / '2026-09-11-recursive-owner-composition/inputs.json'
    old = json.loads(old_path.read_text())
    base_model = old['source_model']['base_model_path']
    tokenizer_path = Path(base_model) / 'tokenizer.json'
    tok = Tokenizer.from_file(str(tokenizer_path))
    records = []
    for image_id, owners in ACQUISITION_SELECTION.items():
        c = next(c for c in candidates['cases'] if c['image_id'] == image_id)
        first = complete_row_span(c['stable_action_ids'], c['stable_parsed']['pred'][0], tok)
        require(first['prefix_token_ids'] == [], 'retained first row must start the action')
        prefix = first['target_token_ids']
        targets = [next(t for t in c['targets'] if t['owner_id'] == owner) for owner in owners]
        require(targets[0]['sample_row_index'] < targets[1]['sample_row_index'],
                'joint order must follow sampled witness order')
        require(not (set(owners) & set(c['stable_score']['50']['owners'])), 'target is already covered')
        records.append({**c, 'A': owners[0], 'B': owners[1], 'targets': targets,
                        'common_prefix_token_ids': prefix,
                        'common_prefix_token_ids_sha256': digest(prefix)})
    sources = {str(candidates_path.resolve()): file_hash(candidates_path),
               str(old_path): file_hash(old_path), str(tokenizer_path): file_hash(tokenizer_path),
               str(Path(__file__).resolve()): file_hash(__file__)}
    for source in old['adapter']['files']:
        p = str(Path(old['adapter']['root']) / source['relative_path'])
        require(file_hash(p) == source['sha256'], 'Stable50 adapter bytes changed')
        sources[p] = source['sha256']
    for c in records:
        sources[c['image_path']] = file_hash(c['image_path'])
    packet = {'schema': 'parallel_owner_composition.acquisition.v1',
              'status': 'prepared_no_model_execution', 'source_files': sources,
              'role_mapping_authority': 'This packet supersedes display A/B aliases: 417044 A=1082918, B=1080038; same admitted owner/row identities, original sample order.',
              'base_model_path': base_model, 'adapter_path': old['adapter']['root'],
              'embedding_path': old['source_model']['source_embedding']['root'],
              'branches': ['natural', 'A', 'B', 'AB'], 'action_cap': 3084,
              'physical_gpu': '0', 'records': records,
              'limits': {'continuations': 8, 'new_tokens': 24672, 'model_forwards': 24680,
                         'image_forwards': 8, 'rank_seconds': 3600,
                         'cuda_peak_allocated_bytes': 32 * 1024**3}}
    output_dir.mkdir(parents=True, exist_ok=False)
    publish_json_exclusive(output_dir / 'input.json', packet)
    return {'input': str(output_dir / 'input.json'), 'sha256': file_hash(output_dir / 'input.json'),
            'images': list(ACQUISITION_SELECTION), 'branches': packet['branches']}


def acquisition_job(case, stage):
    require(stage in ('natural', 'A', 'B', 'AB'), 'unknown acquisition stage')
    prefix = [] if stage == 'natural' else list(case['common_prefix_token_ids'])
    targets = [] if stage == 'natural' else (
        [case['targets'][0]] if stage == 'A' else
        [case['targets'][1]] if stage == 'B' else case['targets'])
    forced = [token for target in targets for token in target['target_token_ids']]
    require(151645 not in prefix + forced and len(prefix + forced) < 3084,
            'nonterminal acquisition extension')
    return {'prefix_ids': prefix, 'forced_ids': forced,
            'forced_owners': [t['owner_id'] for t in targets],
            'remaining_budget': 3084 - len(prefix + forced)}


def reduce_acquisition(records, packet, tok):
    from probes.source_rweak_row_cross.run import native_record

    expected = {(c['image_id'], s) for c in packet['records'] for s in packet['branches']}
    require({(r['image_id'], r['stage']) for r in records} == expected
            and len(records) == len(expected), 'acquisition exact branch denominator')
    result = []
    for c in packet['records']:
        old = set(c['stable_score']['50']['owners'])
        for r in [r for r in records if r['image_id'] == c['image_id']]:
            job = acquisition_job(c, r['stage'])
            require(r['prefix_ids'] == job['prefix_ids'] and r['forced_ids'] == job['forced_ids'],
                    'literal intervention identity')
            require(r['action_ids'] == job['prefix_ids'] + job['forced_ids'] + r['free_ids'],
                    'prefix/forced/free partition')
            require(len(r['free_ids']) <= job['remaining_budget'], 'full action cap')
            require(tok.decode(r['action_ids'], skip_special_tokens=False) == r['text'], 'raw token text')
            parsed = native_record(r['text'], c['image'], c['stable_parsed'], r['stop_reason'])
            require(parsed == r['parsed'], 'cold native parser equality')
            card = score(parsed, seed=None, length=len(r['action_ids']), stop=r['stop_reason'])
            now = set(card['50']['owners'])
            forced_end_chars = len(tok.decode(job['prefix_ids'] + job['forced_ids'], skip_special_tokens=False))
            old_after = [m['owner'] for m in card['50']['matches']
                         if m['owner'] in old and parsed['pred'][m['pred_index']]['char_start'] >= forced_end_chars]
            reasons = []
            if not old <= now:
                reasons.append('old_owner_loss')
            if not set(job['forced_owners']) <= now:
                reasons.append('forced_target_not_in_final_matching')
            cursor = list(job['prefix_ids'])
            assigned_forced = []
            for owner in job['forced_owners']:
                target = next(t for t in c['targets'] if t['owner_id'] == owner)
                begin = len(tok.decode(cursor, skip_special_tokens=False))
                indices = [i for i, p in enumerate(parsed['pred'])
                           if p['char_start'] == begin and p['raw_span_text'] == target['target_text']]
                assigned = len(indices) == 1 and any(
                    m['owner'] == owner and m['pred_index'] == indices[0] for m in card['50']['matches'])
                assigned_forced.append({'owner_id': owner, 'forced_row_assigned': assigned})
                if not assigned:
                    reasons.append('forced_row_not_assigned_to_its_owner:' + owner)
                cursor.extend(target['target_token_ids'])
            if card['50']['fp'] > c['stable_score']['50']['fp']:
                reasons.append('annotation_relative_FP_increase')
            for key in ('strict_repeats', 'parser_drops', 'invalid_predictions'):
                if card[key] > c['stable_score'][key]:
                    reasons.append(key + '_increase')
            if r['stop_reason'] != 'im_end':
                reasons.append('not_native_EOS')
            result.append({'image_id': c['image_id'], 'stage': r['stage'], 'score': card,
                           'A_present': c['A'] in now, 'B_present': c['B'] in now,
                           'old_owner_losses': sorted(old - now), 'old_owners_freely_after_forced': old_after,
                           'forced_owners': job['forced_owners'], 'forced_assignments': assigned_forced, 'reasons': reasons,
                           'conditional_useful': r['stage'] != 'natural' and not reasons})
    return {'schema': 'parallel_owner_composition.acquisition_reduction.v1',
            'claim': 'Supplied row interventions, not learned or autonomous target discovery.',
            'images': len(packet['records']), 'continuations': len(records), 'records': result}


def residual_job(case, stage):
    require(stage == 'residual_B', 'exact residual branch only; no repeated diagonal')
    prefix = list(case['common_prefix_token_ids']) + list(case['targets'][0]['target_token_ids'])
    target = case['targets'][1]
    forced = list(target['target_token_ids'])
    require(151645 not in prefix + forced and len(prefix + forced) < 3084,
            'nonterminal residual extension')
    return {'prefix_ids': prefix, 'forced_ids': forced,
            'forced_owners': [target['owner_id']], 'remaining_budget': 3084 - len(prefix + forced)}


def output_burden(parsed, ids, card):
    geometry_drops = sum(p.get('reason') == 'geometry_invalid' for p in parsed['dropped_predictions'])
    return {'geometry_invalid': geometry_drops + card['invalid_predictions'],
            'other_malformed': len(parsed['dropped_predictions']) - geometry_drops,
            'raw_row_starts': ids.count(151646)}


def residual_outcome(baseline, current, *, new_owner, forced_assigned,
                     baseline_burden, current_burden, stop):
    before, after = set(baseline['50']['owners']), set(current['50']['owners'])
    require(new_owner not in before, 'residual B is already present in accepted A baseline')
    reasons = []
    if not before <= after:
        reasons.append('A_path_owner_loss')
    if new_owner not in after or not forced_assigned:
        reasons.append('residual_forced_row_not_assigned')
    if current['50']['fp'] > baseline['50']['fp']:
        reasons.append('annotation_relative_FP_increase')
    for key in ('strict_repeats', 'parser_drops', 'invalid_predictions'):
        if current[key] > baseline[key]:
            reasons.append(key + '_increase')
    for key in ('geometry_invalid', 'other_malformed'):
        if current_burden[key] > baseline_burden[key]:
            reasons.append(key + '_increase')
    if stop != 'im_end':
        reasons.append('not_native_EOS')
    return {'incrementally_useful': not reasons, 'reasons': reasons,
            'lost_A_path_owners': sorted(before - after), 'gained_vs_A': sorted(after - before)}


def validate_residual_packet(packet, *, verify_sources=True):
    from tokenizers import Tokenizer
    from probes.source_rweak_row_cross.run import native_record

    require(packet['schema'] == 'parallel_owner_composition.residual_admission.v1', 'residual schema')
    require(packet['candidate_source']['sha256'] == RESIDUAL_CANDIDATE_SHA256, 'frozen residual candidate identity')
    require(packet['branches'] == ['residual_B'] and packet['action_cap'] == 3084,
            'exactly two new residual calls; no diagonal/branch extension')
    require(packet['generation_policy'] == {'temperature': 0., 'top_p': 1., 'repetition_penalty': 1.,
            'top_k': 0, 'use_model_defaults': False, 'eos_token_id': 151645, 'action_cap': 3084},
            'retained baseline decoder identity')
    require(len(packet['records']) == 2 and {c['image_id'] for c in packet['records']} == set(RESIDUAL_SELECTION),
            'fixed two residual images')
    if verify_sources:
        for path, sha in packet['source_files'].items():
            require(file_hash(path) == sha, f'changed frozen residual source: {path}')
    candidate = json.loads(Path(packet['candidate_source']['path']).read_text())
    require(file_hash(packet['candidate_source']['path']) == RESIDUAL_CANDIDATE_SHA256,
            'residual candidate bytes changed')
    tok = Tokenizer.from_file(packet['base_model_path'] + '/tokenizer.json')
    accepted_input = json.loads(Path(packet['baseline_sources']['input']).read_text())
    accepted_rows = [json.loads(line) for line in Path(packet['baseline_sources']['rows']).read_text().splitlines()]
    require(packet['accepted_loaded_identity'] == json.loads(Path(packet['baseline_sources']['loaded_identity']).read_text()),
            'accepted loaded baseline identity binding')
    for c in packet['records']:
        accepted_case = next(r for r in accepted_input['records'] if r['image_id'] == c['image_id'])
        for key in ('image', 'example_id', 'prompt_token_ids', 'common_prefix_token_ids',
                    'stable_parsed', 'stable_action_ids', 'stable_score'):
            require(c[key] == accepted_case[key], 'accepted baseline input identity:' + key)
        source = next(s for s in candidate['selected_candidates'] if s['image_id'] == c['image_id'])
        require((c['A'], c['B']) == RESIDUAL_SELECTION[c['image_id']], 'frozen residual owner roles')
        require(c['targets'] == [source['A_record'], source['new_B_record']]
                and c['common_prefix_token_ids'] == source['common_P'], 'literal residual target/prefix identity')
        job = residual_job(c, 'residual_B')
        require(job['prefix_ids'] == source['B_training_prefix'], 'recursive B conditioning identity')
        require(tok.decode(job['forced_ids'], skip_special_tokens=False) == c['targets'][1]['target_text'],
                'literal complete residual B text')
        baseline = c['residual_baseline']['raw']
        accepted_row = next(r for r in accepted_rows if r['image_id'] == c['image_id'] and r['stage'] == 'A')
        require(baseline == accepted_row and digest(baseline) == c['residual_baseline']['raw_row_sha256'],
                'actual retained P+A baseline row binding')
        require(baseline['stage'] == 'A' and baseline['image_id'] == c['image_id']
                and baseline['example_id'] == c['example_id'], 'accepted A baseline association')
        require(baseline['prefix_ids'] == c['common_prefix_token_ids']
                and baseline['forced_ids'] == c['targets'][0]['target_token_ids'], 'baseline exact P+A intervention')
        require(baseline['action_ids'] == job['prefix_ids'] + baseline['free_ids'], 'baseline exact P+A/free partition')
        require(tok.decode(baseline['action_ids'], skip_special_tokens=False) == baseline['text'], 'baseline token/text')
        parsed = native_record(baseline['text'], c['image'], c['stable_parsed'], baseline['stop_reason'])
        require(parsed == baseline['parsed'], 'accepted A baseline native replay')
        baseline_score = score(parsed, seed=None, length=len(baseline['action_ids']), stop=baseline['stop_reason'])
        require(baseline_score == c['residual_baseline']['score'], 'accepted A baseline score replay')
        require(c['A'] in baseline_score['50']['owners'] and c['B'] not in baseline_score['50']['owners'],
                'B must remain absent after accepted A')
        require(c['residual_baseline']['burden'] == output_burden(parsed, baseline['action_ids'], baseline_score),
                'accepted A baseline complete burden ledger')
    remaining = sum(residual_job(c, 'residual_B')['remaining_budget'] for c in packet['records'])
    require(packet['limits']['continuations'] == packet['limits']['image_forwards'] == 2
            and packet['limits']['new_tokens'] == remaining
            and packet['limits']['model_forwards'] == remaining + 2, 'exact residual model-work bound')
    return packet


def prepare_residual_admission(candidates_path: Path, output_dir: Path):
    require(not output_dir.exists(), 'occupied residual preparation')
    require(file_hash(candidates_path) == RESIDUAL_CANDIDATE_SHA256, 'frozen residual candidates changed')
    candidate = json.loads(candidates_path.read_text())
    root = candidates_path.parent.parent
    old_input_path = root / 'acquisition-preparation-v1/input.json'
    old = json.loads(old_input_path.read_text())
    raw_path, reduced_path = root / 'acquisition-v1/rows.jsonl', root / 'acquisition-v1/reduction.json'
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    reduced = json.loads(reduced_path.read_text())
    loaded_path = root / 'acquisition-v1/loaded-model.json'
    packet = {k: copy.deepcopy(old[k]) for k in ('base_model_path', 'adapter_path', 'embedding_path', 'action_cap', 'limits')}
    records = []
    for selected in candidate['selected_candidates']:
        c = copy.deepcopy(next(c for c in old['records'] if c['image_id'] == selected['image_id']))
        c['old_redundant_B'] = c['B']
        c['B'] = selected['new_B_owner_id']
        c['targets'] = [selected['A_record'], selected['new_B_record']]
        baseline = next(r for r in raw if r['image_id'] == c['image_id'] and r['stage'] == 'A')
        baseline_score = next(r['score'] for r in reduced['records'] if r['image_id'] == c['image_id'] and r['stage'] == 'A')
        c['residual_baseline'] = {'raw': baseline, 'score': baseline_score,
                                  'burden': output_burden(baseline['parsed'], baseline['action_ids'], baseline_score),
                                  'raw_row_sha256': digest(baseline)}
        c['backward_x_transition'] = selected['B_is_before_A_in_x_order']
        records.append(c)
    # Accepted first-wave source is preserved before this bounded extension.
    old_snapshot = root / 'acquisition-v1/code-snapshot/composition.py'
    require(file_hash(old_snapshot) == old['source_files'][str(Path(__file__).resolve())],
            'accepted first-wave source snapshot identity')
    sources = {path: sha for path, sha in old['source_files'].items() if path != str(Path(__file__).resolve())}
    sources[str(old_snapshot)] = file_hash(old_snapshot)
    for path in [candidates_path, old_input_path, raw_path, reduced_path, loaded_path,
                 root / 'acquisition-v1/terminal.json', Path(__file__).resolve(),
                 Path(__file__).parent / 'tests/test_composition.py',
                 Path(__file__).parents[1] / 'dora_owner_learning/runtime.py',
                 Path(__file__).parents[1] / 'dora_owner_learning/route_access.py',
                 Path(__file__).parents[1] / 'source_rweak_row_cross/run.py',
                 Path(__file__).parents[2] / 'src/qwen/generation.py',
                 Path(__file__).parents[2] / 'src/qwen/native.py',
                 Path(__file__).parents[2] / 'src/inference/parsing.py']:
        sources[str(path.resolve())] = file_hash(path)
    remaining = sum(residual_job(c, 'residual_B')['remaining_budget'] for c in records)
    packet.update(schema='parallel_owner_composition.residual_admission.v1',
                  status='READY_no_GPU_assignment_or_execution_grant', branches=['residual_B'],
                  source_files=sources, records=records,
                  candidate_source={'path': str(candidates_path.resolve()), 'sha256': RESIDUAL_CANDIDATE_SHA256},
                  baseline_sources={'input': str(old_input_path), 'rows': str(raw_path),
                                    'reduction': str(reduced_path), 'loaded_identity': str(loaded_path)},
                  accepted_loaded_identity=json.loads(loaded_path.read_text()),
                  runtime_device_policy='one explicitly root-assigned physical GPU passed at launch',
                  generation_policy={'temperature': 0., 'top_p': 1., 'repetition_penalty': 1., 'top_k': 0,
                                     'use_model_defaults': False, 'eos_token_id': 151645, 'action_cap': 3084},
                  baseline_reuse='Two accepted P+A rows, exact model/prompt/history/decoder; no repeated model cells.',
                  question='Does literal residual B at P+A add its owner while preserving the actual A owner set and burden?',
                  stop='Exactly two new residual continuations; no alternate owner, order, seed, prefix, image, or fitting.')
    packet['limits'].update(continuations=2, image_forwards=2, new_tokens=remaining, model_forwards=remaining + 2)
    validate_residual_packet(packet)
    output_dir.mkdir(parents=True, exist_ok=False)
    publish_json_exclusive(output_dir / 'input.json', packet)
    summary = {'input': str(output_dir / 'input.json'), 'sha256': file_hash(output_dir / 'input.json'),
               'new_model_calls': 2, 'reused_baseline_calls': 2, 'limits': packet['limits'],
               'cases': [{'image_id': c['image_id'], 'A': c['A'], 'B': c['B'],
                          'A_baseline_owners': c['residual_baseline']['score']['50']['owners'],
                          'B_prefix_tokens': len(residual_job(c, 'residual_B')['prefix_ids']),
                          'B_tokens': len(c['targets'][1]['target_token_ids']),
                          'backward_x_transition': c['backward_x_transition']} for c in records]}
    publish_json_exclusive(output_dir / 'summary.json', summary)
    return summary


def reduce_residual_admission(records, packet, tok):
    from probes.source_rweak_row_cross.run import native_record

    expected = {(c['image_id'], 'residual_B') for c in packet['records']}
    require(len(records) == len(expected) and {(r['image_id'], r['stage']) for r in records} == expected,
            'exact residual two-call denominator')
    results = []
    for c in packet['records']:
        row = next(r for r in records if r['image_id'] == c['image_id'])
        job = residual_job(c, row['stage'])
        require(row['example_id'] == c['example_id'], 'residual raw case identity')
        require(row['prefix_ids'] == job['prefix_ids'] and row['forced_ids'] == job['forced_ids'],
                'residual exact P+A and full B intervention')
        require(row['action_ids'] == job['prefix_ids'] + job['forced_ids'] + row['free_ids']
                and row['remaining_budget'] == job['remaining_budget']
                and len(row['free_ids']) <= job['remaining_budget'], 'residual literal partition/budget')
        require(tok.decode(row['action_ids'], skip_special_tokens=False) == row['text'], 'residual token/text')
        parsed = native_record(row['text'], c['image'], c['stable_parsed'], row['stop_reason'])
        require(parsed == row['parsed'], 'residual cold native parser equality')
        card = score(parsed, seed=None, length=len(row['action_ids']), stop=row['stop_reason'])
        begin = len(tok.decode(job['prefix_ids'], skip_special_tokens=False))
        forced_indices = [i for i, pred in enumerate(parsed['pred'])
                          if pred['char_start'] == begin and pred['raw_span_text'] == c['targets'][1]['target_text']]
        assigned = len(forced_indices) == 1 and any(
            m['owner'] == c['B'] and m['pred_index'] == forced_indices[0] for m in card['50']['matches'])
        burden = output_burden(parsed, row['action_ids'], card)
        base = c['residual_baseline']
        outcome = residual_outcome(base['score'], card, new_owner=c['B'], forced_assigned=assigned,
                                   baseline_burden=base['burden'], current_burden=burden, stop=row['stop_reason'])
        results.append({'image_id': c['image_id'], 'A': c['A'], 'residual_B': c['B'],
                        'backward_x_transition': c['backward_x_transition'], 'score': card, 'burden': burden,
                        'baseline_score': base['score'], 'baseline_burden': base['burden'],
                        'forced_B_assigned': assigned, **outcome,
                        'changes_by_threshold': {str(t): {'gained': sorted(set(card[str(t)]['owners']) - set(base['score'][str(t)]['owners'])),
                                                         'lost': sorted(set(base['score'][str(t)]['owners']) - set(card[str(t)]['owners']))}
                                                 for t in (50, 60, 80)}})
    return {'schema': 'parallel_owner_composition.residual_reduction.v1', 'new_continuations': len(records),
            'reused_A_baselines': len(packet['records']), 'results': results,
            'claim': 'Supplied residual-row admission versus actual P+A owner set; no learning or ordering-cause verdict.'}


def execute_acquisition(input_path: Path, output_dir: Path, *, physical_gpu=None):
    import torch
    from tokenizers import Tokenizer
    from probes.dora_owner_learning.route_access import CONFIG, checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from probes.source_rweak_row_cross.run import build_requests, native_record
    from src.config.inference import load_research_infer_config
    from src.qwen.generation import generate_continuations, NativeGenerationPolicy
    from src.qwen.native import prepare_native_inputs

    packet = json.loads(input_path.read_text())
    residual = packet['schema'] == 'parallel_owner_composition.residual_admission.v1'
    require(residual or packet['schema'] == 'parallel_owner_composition.acquisition.v1', 'acquisition schema')
    if residual:
        validate_residual_packet(packet)
        require(isinstance(physical_gpu, str) and physical_gpu.isdecimal(), 'explicit root-assigned physical GPU required')
    else:
        physical_gpu = packet['physical_gpu']
        require(physical_gpu == '0', 'original acquisition reserved GPU0')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == physical_gpu
            and torch.cuda.device_count() == 1, 'exactly one assigned physical GPU')
    require(not output_dir.exists(), 'occupied acquisition execution')
    for path, sha in packet['source_files'].items():
        require(file_hash(path) == sha, f'changed frozen source: {path}')
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {'status': 'running', 'input_sha256': file_hash(input_path), 'pid': os.getpid(),
               'physical_gpu': physical_gpu,
               'model_loads': 0, 'model_forwards': 0, 'image_forwards': 0,
               'continuations': 0, 'new_tokens': 0}
    publish_json_exclusive(output_dir / 'launch.json', receipt)
    def expired(*_):
        raise TimeoutError('composition acquisition one-hour invocation bound')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(packet['limits']['rank_seconds'])
    try:
        cfg = checkpoint_config(load_research_infer_config(CONFIG).config, packet['adapter_path'])
        require(str(cfg.model.base_model) == packet['base_model_path']
                and str(cfg.embedding_delta.path) == packet['embedding_path'], 'base/embedding identity')
        qwen, identity = load_policy(cfg, device=torch.device('cuda:0'))
        receipt['model_loads'] = 1
        require(identity['model_identity']['adapter']['adapter_path'] == packet['adapter_path']
                and not identity['model_identity']['adapter']['merged_adapters'], 'Stable50 unmerged identity')
        if residual:
            require(digest(identity) == digest(packet['accepted_loaded_identity']),
                    'accepted baseline model/numerics identity changed; stop before generation')
        publish_json_exclusive(output_dir / 'loaded-model.json', identity)
        tok = Tokenizer.from_file(packet['base_model_path'] + '/tokenizer.json')
        policy = NativeGenerationPolicy(temperature=0., top_p=1., repetition_penalty=1.,
                                        top_k=0, use_model_defaults=False)
        def count_model(*_):
            receipt['model_forwards'] += 1
            require(receipt['model_forwards'] <= packet['limits']['model_forwards'], 'model forward bound')
        def count_image(*_):
            receipt['image_forwards'] += 1
            require(receipt['image_forwards'] <= packet['limits']['image_forwards'], 'image forward bound')
        qwen.model.register_forward_pre_hook(count_model)
        visual = [m for n, m in qwen.model.named_modules() if n.endswith('visual')]
        require(len(visual) == 1, 'unique visual counter')
        visual[0].register_forward_pre_hook(count_image)
        torch.cuda.reset_peak_memory_stats()
        records = []
        with (output_dir / 'rows.jsonl').open('x') as stream:
            for c in packet['records']:
                requests, _ = build_requests(qwen, cfg.model_dump(mode='json'), [c['image']])
                batch = prepare_native_inputs(qwen.processor, requests, device=torch.device('cuda:0'),
                                              record_media_identity=True)
                require(list(batch.prompt_token_ids[0]) == c['prompt_token_ids'], 'exact native prompt')
                plan = c['image']['image_plan']
                require(batch.media_sha256[0] == plan['executed_media_sha256']
                        and list(batch.image_grids[0]) == plan['observed_image_grid_thw'], 'exact image/grid')
                for stage in packet['branches']:
                    job = residual_job(c, stage) if residual else acquisition_job(c, stage)
                    tick = time.monotonic()
                    with torch.inference_mode():
                        generated = generate_continuations(qwen.model, batch,
                            extensions=[job['prefix_ids'] + job['forced_ids']], budgets=[job['remaining_budget']],
                            eos_token_id=151645, pad_token_id=qwen.tokenizer.pad_token_id,
                            policy=policy, trace='none')[0]
                    require(generated.request_id == c['example_id'], 'request association')
                    free = list(generated.token_ids)
                    ids = job['prefix_ids'] + job['forced_ids'] + free
                    text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
                    parsed = native_record(text, c['image'], c['stable_parsed'], generated.stop_reason)
                    row = {'image_id': c['image_id'], 'example_id': c['example_id'], 'stage': stage,
                           **job, 'free_ids': free, 'action_ids': ids, 'text': text,
                           'stop_reason': generated.stop_reason, 'parsed': parsed,
                           'seconds': time.monotonic() - tick}
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    records.append(row)
                    receipt['continuations'] += 1
                    receipt['new_tokens'] += len(free)
                    if stage == 'natural':
                        require(ids == c['stable_action_ids'], 'cold Stable50 natural token diagonal')
                        observed = score(parsed, seed=-1, length=len(ids), stop=generated.stop_reason)
                        require(observed == c['stable_score'], 'cold Stable50 natural score diagonal')
                    require(torch.cuda.max_memory_allocated() <= packet['limits']['cuda_peak_allocated_bytes'],
                            'CUDA allocation bound')
        reduction = reduce_residual_admission(records, packet, tok) if residual else reduce_acquisition(records, packet, tok)
        publish_json_exclusive(output_dir / 'reduction.json', reduction)
        require(receipt['continuations'] == packet['limits']['continuations']
                and receipt['image_forwards'] == receipt['continuations'], 'exact acquisition counts')
        receipt['status'] = 'completed'
    except BaseException as exc:
        receipt.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        receipt.update(elapsed_seconds=time.monotonic() - started,
                       rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0)
        publish_json_exclusive(output_dir / 'terminal.json', receipt)
    return {k: receipt[k] for k in ('status', 'continuations', 'new_tokens', 'elapsed_seconds')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'cards', 'prepare-acquisition', 'acquire',
                                           'prepare-residual', 'verify-residual'])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--physical-gpu')
    args = parser.parse_args()
    require(args.command == 'verify-residual' or args.output_dir, 'output directory required')
    if args.command == 'inventory':
        result = prepare_inventory(args.output_dir)
    elif args.command == 'cards':
        require(args.inventory and args.selection, 'cards require inventory and selection')
        result = prepare_cards(args.inventory, args.selection, args.output_dir)
    elif args.command == 'prepare-acquisition':
        require(args.candidates, 'preparation requires candidates')
        result = prepare_acquisition(args.candidates, args.output_dir)
    elif args.command == 'prepare-residual':
        require(args.candidates, 'residual preparation requires candidates')
        result = prepare_residual_admission(args.candidates, args.output_dir)
    elif args.command == 'verify-residual':
        require(args.input, 'verification requires frozen residual input')
        packet = validate_residual_packet(json.loads(args.input.read_text()))
        result = {'status': 'verified_CPU_only', 'input_sha256': file_hash(args.input),
                  'new_continuations': packet['limits']['continuations']}
    else:
        require(args.input, 'acquisition requires frozen input')
        result = execute_acquisition(args.input, args.output_dir, physical_gpu=args.physical_gpu)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
