"""Frozen Source-to-round1 witness replay; no generation, update, or new selection."""
from __future__ import annotations

from src.artifacts.source_provenance import preserve_source

import argparse
from collections import Counter, defaultdict
import gc
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

from src.artifacts import load_canonical_json, publish_json_exclusive
from .candidate_opportunity import digest, file_hash, indexed, require, rows
from .round1_realization import (
    ROOT, SOURCE_ROOT, CANDIDATE_ROOT, UPDATE_ROOT, trace_card, validate_identity,
)

OUTPUT = ROOT / '2026-09-10-fixed-witness-route-access'
POST = ROOT / '2026-09-09-round1-greedy-realization/cold/source256-rloo-round1-train256-natural-v1'
REALIZATION = ROOT / '2026-09-09-round1-greedy-realization/analysis-v1'
CONFIG = Path(__file__).with_name('configs') / 'source256.yaml'
EOS, PAD, CAP = 151645, 151643, 3084


def checked_ids(ids, stop):
    require(isinstance(ids, list) and 0 < len(ids) <= CAP, 'action length')
    require(all(type(t) is int and t >= 0 for t in ids), 'action token type')
    require(PAD not in ids and EOS not in ids[:-1], 'action EOS/pad corruption')
    require(stop in ('im_end', 'length') and (ids[-1] == EOS) == (stop == 'im_end')
            and (stop != 'length' or len(ids) == CAP), 'action terminal corruption')
    return ids


def publish(path, payload):
    publish_json_exclusive(path, payload)
    require(load_canonical_json(path) == payload, 'publication reload differs')


def first_fork(ids, reference):
    return next((i for i, (a, b) in enumerate(zip(ids, reference)) if a != b),
                min(len(ids), len(reference)) if ids != reference else None)


def section_labels(ids, tokenizer, predictions, sample, reference):
    """Exact character-to-token map; prefix is an overlapping annotation, not a partition."""
    pieces = [tokenizer.decode([t], skip_special_tokens=False) for t in ids]
    require(''.join(pieces) == tokenizer.decode(ids, skip_special_tokens=False), 'token/character alignment')
    ends, n = [], 0
    for piece in pieces:
        ends.append((n, n + len(piece)))
        n += len(piece)
    gained = set(sample['comparison']['gained'])
    target_indices = {m['pred_index'] for m in sample['50']['matches'] if m['owner'] in gained}
    # Matching indices refer to valid parsed predictions. Frozen candidates have no invalid predictions.
    require(sample['invalid_predictions'] == 0, 'target-row mapping needs valid prediction indices')
    require(len(predictions) == sample['parsed_prediction_count'], 'prediction span coverage')
    spans = []
    for i, pred in enumerate(predictions):
        a, b = pred['char_start'], pred['char_end']
        require(''.join(pieces)[a:b] == pred['raw_span_text'], 'prediction character span corruption')
        require(a in {x for x, _ in ends} and b in {y for _, y in ends}, 'row/token boundary misalignment')
        spans.append((a, b, 'gained_target_row' if i in target_indices else 'other_row'))
    labels = []
    for a, b in ends:
        matched = [label for x, y, label in spans if x <= a and b <= y]
        require(len(matched) <= 1, 'overlapping parsed rows')
        labels.append(matched[0] if matched else 'wrapper_eos_or_unparsed')
    require('gained_target_row' in labels, 'missing gained target row')
    fork = first_fork(ids, reference)
    return dict(partition=labels, common_prefix_length=len(ids) if fork is None else fork,
                source_greedy_first_fork=fork,
                target_prediction_indices=sorted(target_indices),
                definition='Parsed gained-owner rows / other parsed rows / wrapper EOS or unparsed spans; common prefix overlaps this partition.')


def branch_support(ids, actions):
    require(len(actions) == 4 and len({a['seed'] for a in actions}) == 4, 'K4 sibling identity')
    require(abs(sum(a['advantage'] for a in actions)) < 1e-12, 'RLOO group cancellation')
    support = []
    active = list(actions)
    for i, target in enumerate(ids):
        live = [a for a in active if len(a['action_token_ids']) > i]
        by_token = defaultdict(float)
        for a in live:
            by_token[a['action_token_ids'][i]] += a['advantage']
        support.append(dict(position=i, shared_prefix_siblings=len(live),
                            prefix_advantage_sum=sum(a['advantage'] for a in live),
                            target_branch_advantage_sum=by_token.get(target, 0.),
                            next_token_advantage_sums={str(k): v for k, v in sorted(by_token.items())}))
        active = [a for a in live if a['action_token_ids'][i] == target]
    return support


def score_logits(logits, target_ids, *, prompt_length):
    import torch
    from src.losses import aligned_token_logprobs
    require(logits.ndim == 2 and target_ids.ndim == 1 and logits.shape[0] == target_ids.numel(),
            'token score alignment')
    require(target_ids.dtype == torch.long and logits.shape[1] > 1 and prompt_length > 0,
            'token score schema')
    require(bool(torch.isfinite(logits).all()), 'nonfinite logits')
    require(bool(((target_ids >= 0) & (target_ids < logits.shape[1])).all()), 'target vocabulary')
    ll = aligned_token_logprobs(logits, target_ids)
    chosen = logits.gather(1, target_ids[:, None]).squeeze(1)
    topval, topid = logits.max(dim=1)  # torch.max uses the lowest vocabulary ID for exact ties.
    ties = (logits == topval[:, None]).sum(dim=1)
    other = logits.clone()
    other.scatter_(1, target_ids[:, None], -float('inf'))
    otherval, otherid = other.max(dim=1)
    values = torch.stack((ll, chosen, topval, otherval, chosen - topval, chosen - otherval), 1).cpu().tolist()
    targets, winners, others, tiecounts = [x.cpu().tolist() for x in (target_ids, topid, otherid, ties)]
    positions = [dict(position=i, causal_logit_position=prompt_length - 1 + i, target_id=targets[i],
                      top1_id=winners[i], best_other_id=others[i], top1_tie_count=tiecounts[i],
                      target_is_argmax=targets[i] == winners[i], target_in_top_tie=values[i][4] == 0.,
                      logprob=values[i][0], target_logit=values[i][1], top1_logit=values[i][2],
                      best_other_logit=values[i][3], target_top1_margin=values[i][4],
                      target_best_other_margin=values[i][5]) for i in range(len(targets))]
    return dict(positions=positions, length=len(targets), sum_logprob=sum(v[0] for v in values),
                mean_logprob=sum(v[0] for v in values) / len(values),
                first_non_argmax=next((p['position'] for p in positions if not p['target_is_argmax']), None),
                tie_handling='Exact FP32 comparison; torch.max selects lowest vocabulary ID. No tolerance used for winner parity.')


def validate_score(score, route, prompt_length):
    pos = score['positions']
    require(score['length'] == len(route['ids']) == len(pos), 'score missing positions')
    require([p['position'] for p in pos] == list(range(len(pos))), 'score position alignment')
    require([p['target_id'] for p in pos] == route['ids'], 'score target alignment')
    require([p['causal_logit_position'] for p in pos] == list(range(prompt_length - 1, prompt_length - 1 + len(pos))),
            'score causal alignment')
    require(all(math.isfinite(p[k]) for p in pos for k in ('logprob', 'target_best_other_margin', 'target_logit')),
            'nonfinite saved score')
    require(abs(sum(p['logprob'] for p in pos) - score['sum_logprob']) < 1e-10, 'score reduction corruption')


def prepare(output):
    from tokenizers import Tokenizer
    require(not output.exists(), 'occupied output root')
    sources = {}
    def read(path, expected=None, jsonl=False):
        path = Path(path)
        observed = file_hash(path)
        require(expected is None or observed == expected, f'input hash changed: {path}')
        sources[str(path)] = observed
        return rows(path) if jsonl else json.loads(path.read_text())
    cs = read(CANDIDATE_ROOT / 'summary.json')
    cases = indexed(read(CANDIDATE_ROOT / 'cases.json', cs['cases_sha256']), 'example_id')
    rs = read(REALIZATION / 'summary.json')
    outcomes = indexed(read(REALIZATION / 'cases.json', rs['cases_sha256']), 'example_id')
    for path, sha in {**cs['source_files'], **rs['source_files']}.items():
        require(file_hash(path) == sha, f'accepted input changed: {path}')
        sources[path] = sha
    plan = read(UPDATE_ROOT / 'round-1/plan.json')
    require(digest({k: v for k, v in plan.items() if k != 'content_sha256'}) == plan['content_sha256'], 'plan integrity')
    receipt = read(UPDATE_ROOT / 'round-1/update/receipt.json')
    require(receipt['mechanical_status'] == 'MECHANICALLY_VALID' and receipt['round'] == 1 and receipt['arm'] == 'rloo'
            and receipt['plan']['sha256'] == file_hash(UPDATE_ROOT / 'round-1/plan.json'), 'update lineage')
    sm, pm = read(SOURCE_ROOT / 'run_manifest.json'), read(POST / 'run_manifest.json')
    validate_identity(pm, sm, receipt)
    groups = indexed(plan['population']['groups'], 'example_id')
    selected = [(eid, s) for eid, c in cases.items() for s in c['samples'] if s['comparison']['strong_joint_witness']]
    require(len(selected) == 43 and len({eid for eid, _ in selected}) == 32, 'frozen strong population')
    eids = {eid for eid, _ in selected}
    tokenizer = Tokenizer.from_file(str(Path(plan['model']['base_model_path']) / 'tokenizer.json'))
    cells = {}
    for spec in plan['sources']['rollout_artifacts']:
        shard = read(spec['path'], spec['sha256'])
        for row in shard['rollouts']:
            key = (row['example_id'], row['seed'])
            require(key not in cells, 'duplicate acquisition cell')
            cells[key] = row
    routes, greedy, witnesses = {}, {}, []
    def add(eid, ids, stop, role):
        checked_ids(ids, stop)
        key = digest([eid, ids])
        if key not in routes:
            routes[key] = dict(route_id=key, example_id=eid, ids=ids, stop_reason=stop, roles=[])
        require(role not in routes[key]['roles'], 'duplicate route role')
        routes[key]['roles'].append(role)
        return key
    for name, root in [('source_greedy', SOURCE_ROOT), ('post_greedy', POST)]:
        raw = indexed(read(root / 'gt_vs_pred.jsonl', jsonl=True), 'row_id')
        trace = defaultdict(list)
        for t in read(root / 'pred_token_trace.jsonl', jsonl=True):
            if t['trace_type'] == 'generated_token' and t['row_id'] in eids:
                trace[t['row_id']].append(t)
        require(set(trace) == eids, 'missing greedy cells')
        for eid in sorted(eids):
            trace_card(raw[eid], trace[eid], tokenizer)
            ts = sorted(trace[eid], key=lambda t: t['generated_step_index'])
            ids = [t['token_id'] for t in ts if not t['is_pad']]
            greedy[eid, name] = add(eid, ids, raw[eid]['decode_stop_reason'], name)
    for eid, sample in selected:
        group = groups[eid]
        actions = {a['seed']: a for a in group['actions']}
        require(len(actions) == 4, 'missing K4 actions')
        action = actions[sample['seed']]
        row = cells[eid, sample['seed']]
        ids = action['action_token_ids']
        require(digest(ids) == action['action_token_ids_sha256'] and
                row['generated_token_ids'] == action['generated_token_ids'] and
                digest(row['predictions']) == action['parser_evidence_sha256'] and
                ids == row['generated_token_ids'] + [EOS] and row['stop_reason'] == 'im_end', 'candidate binding')
        require(sample['50']['owners'] == action['matching']['matched_owner_refs'], 'candidate owner binding')
        require(tokenizer.decode(ids[:-1], skip_special_tokens=False) == row['generated_text'], 'candidate text identity')
        key = add(eid, ids, action['stop_reason'], f"witness:{sample['seed']}")
        source_ids = routes[greedy[eid, 'source_greedy']]['ids']
        post_ids = routes[greedy[eid, 'post_greedy']]['ids']
        labels = section_labels(ids, tokenizer, row['predictions']['predictions'], sample, source_ids)
        witness = dict(witness_id=f"{eid}:{sample['seed']}", example_id=eid, seed=sample['seed'], route_id=key,
                       source_greedy_route=greedy[eid, 'source_greedy'], post_greedy_route=greedy[eid, 'post_greedy'],
                       advantage=action['advantage'], advantage_sign='positive' if action['advantage'] > 0 else 'negative' if action['advantage'] < 0 else 'zero',
                       retained_sample=sample, sections=labels, post_greedy_first_fork=first_fork(ids, post_ids),
                       owner_changes=outcomes[eid]['owner_changes'],
                       retained_witness_realization=next(w for w in outcomes[eid]['witness_realization'] if w['seed'] == sample['seed']),
                       cpu_branch_support=branch_support(ids, group['actions']))
        witnesses.append(witness)
    require(Counter(w['advantage_sign'] for w in witnesses) == {'positive': 37, 'zero': 2, 'negative': 4}, 'advantage strata')
    require(2 * len(routes) <= 214, 'forward budget preparation')
    packet = dict(schema='fixed_witness_route_access.v1', sources=sources, plan=plan, update_receipt=receipt,
                  routes=list(routes.values()), witnesses=witnesses,
                  groups=[groups[eid] for eid in sorted(eids)],
                  observed_greedy_outcomes=[outcomes[eid] for eid in sorted(eids)],
                  limits=dict(gpu=0, seconds=3600, score_forwards=220, primary_loads=2),
                  cpu_credit_definition='Maximize sum_image sum_K advantage * sum_token logprob / (256*4). Prefix support is raw advantage sum; shared-prefix cancellation does not imply unchanged shared-parameter logits.')
    output.mkdir(parents=True, exist_ok=False)
    publish(output / 'inputs.json', packet)
    staged = []
    for name, identity in [('source_adapter', plan['model']['current_adapter']),
                           ('post_adapter', receipt['saved_adapter']), ('embedding', plan['model']['source_embedding'])]:
        for f in identity['files']:
            original = Path(identity['root']) / f['relative_path']
            require(file_hash(original) == f['sha256'], f'model bytes changed: {original}')
            target = output / 'staged' / name / f['relative_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            require(file_hash(target) == f['sha256'], 'model staging copy changed')
            staged.append(dict(source=str(original), staged=str(target), sha256=f['sha256'], size_bytes=target.stat().st_size))
    # Preserve effective local dependencies, not just Git's nominal revision.
    code = sorted(set([Path(__file__), Path(__file__).with_name('runtime.py'), Path(__file__).with_name('train.py'),
                       Path(__file__).with_name('candidate_opportunity.py'), Path(__file__).with_name('round1_realization.py')]
                      + list(Path('src/qwen').glob('*.py')) + list(Path('src/losses').glob('*.py'))))
    for path in [*code + [CONFIG, Path(__file__).with_name('tests') / 'test_route_access.py'], Path(preserve_source.__code__.co_filename)]:
        target = preserve_source(path, run_root=output, relative_name=Path('effective_code') / str(path.resolve()).lstrip('/'))
        staged.append(dict(source=str(path.resolve()), staged=str(target), sha256=file_hash(target), size_bytes=target.stat().st_size))
    base = Path(plan['model']['base_model_path'])
    base_files = [dict(path=str(p), sha256=file_hash(p), size_bytes=p.stat().st_size)
                  for p in sorted(base.iterdir()) if p.is_file()]
    publish(output / 'identities.json', dict(staged=staged, base_files=base_files,
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            git_status=subprocess.check_output(['git', 'status', '--short'], text=True)))
    return packet


def reduce(output):
    packet = load_canonical_json(output / 'inputs.json')
    routes = indexed(packet['routes'], 'route_id')
    groups = indexed(packet['groups'], 'example_id')
    require(len(packet['witnesses']) == 43 and len(groups) == 32, 'reducer population')
    scores = {}
    parity = {}
    for model in ('source', 'post'):
        expected = {f'{rid}.json' for rid in routes}
        require({p.name for p in (output / 'scores' / model).glob('*.json')} == expected, 'missing/extra score identities')
        failures = []
        for rid, route in routes.items():
            s = load_canonical_json(output / 'scores' / model / f'{rid}.json')
            require(s['route_id'] == rid and s['model'] == model, 'score identity mismatch')
            validate_score(s, route, len(groups[route['example_id']]['prompt_token_ids']))
            scores[model, rid] = s
            if f'{model}_greedy' in route['roles']:
                failures.extend(dict(example_id=route['example_id'], **p) for p in s['positions'] if not p['target_is_argmax'])
        parity[model] = dict(images=32, mismatch_positions=failures, passed=not failures)
    comparisons = []
    for w in packet['witnesses']:
        before, after = [scores[m, w['route_id']] for m in ('source', 'post')]
        delta = [b['logprob'] - a['logprob'] for a, b in zip(before['positions'], after['positions'])]
        margin_delta = [b['target_best_other_margin'] - a['target_best_other_margin'] for a, b in zip(before['positions'], after['positions'])]
        sections = {}
        for label in ('gained_target_row', 'other_row', 'wrapper_eos_or_unparsed', 'common_prefix'):
            inds = list(range(w['sections']['common_prefix_length'])) if label == 'common_prefix' else [i for i, x in enumerate(w['sections']['partition']) if x == label]
            sections[label] = dict(token_count=len(inds), sum_delta_logprob=sum(delta[i] for i in inds),
                                   mean_delta_logprob=sum(delta[i] for i in inds) / len(inds) if inds else None,
                                   mean_delta_margin=sum(margin_delta[i] for i in inds) / len(inds) if inds else None)
        forks = {}
        for ref in ('source_greedy', 'post_greedy'):
            i = w['sections']['source_greedy_first_fork'] if ref == 'source_greedy' else w['post_greedy_first_fork']
            forks[ref] = dict(position=i, source=before['positions'][i] if i is not None and i < before['length'] else None,
                             post=after['positions'][i] if i is not None and i < after['length'] else None,
                             cpu_support=w['cpu_branch_support'][i] if i is not None and i < len(delta) else None)
        comparisons.append(dict(witness_id=w['witness_id'], example_id=w['example_id'], seed=w['seed'],
                                advantage=w['advantage'], advantage_sign=w['advantage_sign'], length=before['length'],
                                sum_delta_logprob=sum(delta), mean_delta_logprob=sum(delta) / len(delta),
                                source_sum_logprob=before['sum_logprob'], post_sum_logprob=after['sum_logprob'],
                                source_first_non_argmax=before['first_non_argmax'], post_first_non_argmax=after['first_non_argmax'],
                                first_forks=forks, sections=sections,
                                owner_changes=w['owner_changes'], retained_witness_realization=w['retained_witness_realization']))
    strata = {}
    for sign in ('all', 'positive', 'zero', 'negative'):
        selected = [w for w in comparisons if sign == 'all' or w['advantage_sign'] == sign]
        strata[sign] = dict(samples=len(selected), images=len({w['example_id'] for w in selected}),
                            ll_increased=sum(w['sum_delta_logprob'] > 0 for w in selected),
                            mean_sample_delta_logprob=sum(w['sum_delta_logprob'] for w in selected) / len(selected),
                            mean_sample_per_token_delta=sum(w['mean_delta_logprob'] for w in selected) / len(selected))
    image_means = [dict(example_id=eid, sample_count=sum(w['example_id'] == eid for w in comparisons),
                        mean_sample_delta_logprob=sum(w['sum_delta_logprob'] for w in comparisons if w['example_id'] == eid) /
                        sum(w['example_id'] == eid for w in comparisons)) for eid in sorted(groups)]
    return dict(schema='fixed_witness_route_access.reduction.v1', comparisons=comparisons, strata=strata,
                image_means=image_means, mean_image_delta_logprob=sum(x['mean_sample_delta_logprob'] for x in image_means) / len(image_means),
                parity=parity, samples=43, images=32, unique_routes_per_model=len(routes),
                scientific_scope='Selected complete fixed witnesses, not expected reward, a necessary owner-recovery route, or independent owner trials.',
                section_caveat='common_prefix overlaps the three-way row partition; wrapper_eos_or_unparsed includes dropped malformed spans.')


def credit_summary(output):
    """CPU objective coefficients at observed prefixes, not parameter-gradient predictions."""
    packet = load_canonical_json(output / 'inputs.json')
    comparisons = indexed(load_canonical_json(output / 'reduction.json')['comparisons'], 'witness_id')
    candidates = []
    for w in packet['witnesses']:
        source = load_canonical_json(output / 'scores' / 'source' / f"{w['route_id']}.json")
        positions = []
        for p, support in zip(source['positions'], w['cpu_branch_support']):
            require(p['position'] == support['position'], 'credit position alignment')
            # d(sum sibling A log P(action))/d logit(target at this exact prefix).
            raw = support['target_branch_advantage_sum'] - math.exp(p['logprob']) * support['prefix_advantage_sum']
            positions.append(dict(**support, target_probability=math.exp(p['logprob']),
                                  objective_ascent_target_logit_coefficient=raw / (256 * 4)))
        fork = w['sections']['source_greedy_first_fork']
        c = comparisons[w['witness_id']]
        candidates.append(dict(witness_id=w['witness_id'], example_id=w['example_id'], advantage_sign=w['advantage_sign'],
                               positions=positions, source_first_fork_credit=positions[fork] if fork is not None else None,
                               delta_logprob=c['sum_delta_logprob'],
                               source_first_fork_margin_delta=(c['first_forks']['source_greedy']['post']['target_best_other_margin'] -
                                                               c['first_forks']['source_greedy']['source']['target_best_other_margin'])))
    greedy = []
    for route in packet['routes']:
        if not any(role in ('source_greedy', 'post_greedy') for role in route['roles']):
            continue
        a, b = [load_canonical_json(output / 'scores' / m / f"{route['route_id']}.json") for m in ('source', 'post')]
        greedy.append(dict(example_id=route['example_id'], route_id=route['route_id'], roles=route['roles'],
                           length=a['length'], source_sum_logprob=a['sum_logprob'], post_sum_logprob=b['sum_logprob'],
                           delta_logprob=b['sum_logprob'] - a['sum_logprob'],
                           per_token_delta_logprob=b['mean_logprob'] - a['mean_logprob']))
    return dict(candidates=candidates, greedy_comparisons=greedy,
                definition='CPU same-image K4 objective coefficient at exact prefix: (sum A on target branch - P_source(target|prefix)*sum A reaching prefix)/(256*4). Positive means direct logit-coordinate ascent, not a parameter-update or AdamW prediction.',
                limitations='Shared-parameter updates, optimizer state, clipping and all256 groups are not localized by this coefficient. Shared-prefix cancellation is an objective property, not unchanged-logit evidence.')


def checkpoint_config(config, adapter_path):
    """Inference configuration models are immutable; retain validated field types."""
    changed = config.model_copy(update={'adapter': config.adapter.model_copy(update={'path': str(adapter_path)})}, deep=True)
    expected = config.model_dump(mode='json')
    expected['adapter']['path'] = str(adapter_path)
    require(changed.model_dump(mode='json') == expected, 'checkpoint switch changed other config fields')
    return changed


def resume_counters(prior, route_count):
    require(prior['status'] == 'failed' and prior['model_loads'] == 1 and
            prior['model_receipts'] == ['source'] and 'frozen_instance' in prior['error'] and
            prior['actual_score_forwards'] == route_count, 'unsupported post-only continuation')
    seconds = prior['cumulative_model_execution_seconds']
    require(0 < seconds < 3600 and 2 * route_count <= 220 and prior['total_scored_positions'] > 0,
            'post-only cumulative budget')
    return dict(seconds=seconds, forwards=prior['actual_score_forwards'],
                positions=prior['total_scored_positions'], loads=prior['model_loads'])


def execute(output, *, post_only=False):
    import torch
    from src.config.inference import load_research_infer_config
    from src.config.fingerprint import sha256_json
    from src.data import load_raw_examples
    from src.inference.runtime import assemble_frontend
    from src.qwen.native import prepare_replay
    from .runtime import load_policy
    from .train import _materialize_group
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'GPU0 only')
    packet = load_canonical_json(output / 'inputs.json')
    suffix = '_post' if post_only else ''
    require(not (output / f'execution_started{suffix}.json').exists(), 'existing model invocation')
    prior = load_canonical_json(output / 'terminal.json') if post_only else None
    if prior is not None:
        resume_counters(prior, len(packet['routes']))
        require(not (output / 'scores' / 'post').exists(), 'occupied post scores')
        require(len(list((output / 'scores' / 'source').glob('*.json'))) == len(packet['routes']), 'incomplete Source continuation')
    publish(output / f'execution_started{suffix}.json', dict(pid=os.getpid(), time=time.time(), visible_devices='0'))
    plan = packet['plan']
    resolved = load_research_infer_config(CONFIG)
    config = resolved.config
    require(str(config.model.base_model) == plan['model']['base_model_path'] and
            str(config.adapter.path) == plan['model']['current_adapter']['root'] and
            str(config.embedding_delta.path) == plan['model']['source_embedding']['root'] and
            str(config.data.input_jsonl) == plan['sources']['train_jsonl']['path'] and
            config.backend.hf.patch_embed_linearization == 'enabled', 'current config semantic identity')
    require(file_hash(config.data.input_jsonl) == plan['sources']['train_jsonl']['sha256'], 'train bytes changed')
    config_relocation = dict(original=plan['sources']['infer_config'], effective_path=str(CONFIG),
            effective_sha256=file_hash(CONFIG), effective_fingerprint=resolved.fingerprint,
            limitation='Historical worktree/config absent. Maintained native config used; explicit model, data, FP32 SDPA patch-linearization checks and exact per-image prompt/media/grid parity bind execution semantics. Historical config bytes are not claimed identical.')
    if post_only:
        require(config_relocation == load_canonical_json(output / 'config_relocation.json'), 'continuation config changed')
    else:
        publish(output / 'config_relocation.json', config_relocation)
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode='json')))
    raw = {str(r.example_id): r for r in load_raw_examples(config.data.input_jsonl)}
    groups = indexed(packet['groups'], 'example_id')
    routes = packet['routes']
    prior_seconds = prior['cumulative_model_execution_seconds'] if prior else 0.
    started = time.monotonic() - prior_seconds
    forwards = prior['actual_score_forwards'] if prior else 0
    scored_positions = prior['total_scored_positions'] if prior else 0
    loads = prior['model_loads'] if prior else 0
    status, error = 'failed', None
    model_receipts = list(prior['model_receipts']) if prior else []
    def expired(_signum, _frame):
        raise TimeoutError('cumulative 3600 second model execution budget')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(3600 - prior_seconds)))
    try:
        for name in (('post',) if post_only else ('source', 'post')):
            cfg = checkpoint_config(config, packet['update_receipt']['saved_adapter']['root']) if name == 'post' else config
            for f in packet['plan']['model']['source_embedding']['files']:
                require(file_hash(Path(config.embedding_delta.path) / f['relative_path']) == f['sha256'], 'embedding bytes changed')
            adapter_id = plan['model']['current_adapter'] if name == 'source' else packet['update_receipt']['saved_adapter']
            for f in adapter_id['files']:
                require(file_hash(Path(cfg.adapter.path) / f['relative_path']) == f['sha256'], 'adapter bytes changed')
            gpu_state = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], text=True)
            publish(output / f'{name}_preload.json', dict(processes=gpu_state, time=time.time(), cumulative_elapsed=time.monotonic() - started))
            qwen, receipt = load_policy(cfg, device=torch.device('cuda:0'))
            loads += 1
            require(qwen.token_identity.im_end_token_ids == (EOS,), 'native EOS identity')
            require(receipt['effective_settings']['observed_attn_implementation'] == 'sdpa' and
                    receipt['effective_settings']['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32'] and
                    receipt['model_identity']['adapter']['merged_adapters'] == [], 'live FP32 SDPA unmerged identity')
            publish(output / f'{name}_model.json', receipt)
            publish(output / f'{name}_effective_config.json', cfg.model_dump(mode='json'))
            model_receipts.append(name)
            qwen.model.eval()
            torch.cuda.reset_peak_memory_stats()
            # One shortest retained matching greedy route traverses real score/publication/reload first.
            smoke = min((r for r in routes if f'{name}_greedy' in r['roles']), key=lambda r: (len(r['ids']), r['route_id']))
            ordered = [smoke] + sorted((r for r in routes if r is not smoke), key=lambda r: (r['example_id'], r['route_id']))
            current_eid = None
            for route in ordered:
                require(forwards < 220 and time.monotonic() - started < 3600, 'model execution bound')
                eid = route['example_id']
                if eid != current_eid:
                    prompt, inputs, grid = _materialize_group(qwen=qwen, frontend=frontend, config=config, raw=raw[eid], group=groups[eid])
                    current_eid = eid
                t0 = time.monotonic()
                with torch.inference_mode():
                    replay = prepare_replay(qwen.model, {**inputs, 'image_grid_thw': grid},
                                            prompt_token_ids=prompt, continuation_token_ids=route['ids'])
                    forwards += 1
                    logits = replay.aligned_logits(qwen.model(**replay.inputs).logits)
                    scored = score_logits(logits, replay.target_ids, prompt_length=len(prompt))
                    del logits, replay
                torch.cuda.synchronize()
                scored_positions += scored['length']
                scored.update(route_id=route['route_id'], model=name, example_id=eid,
                              elapsed_seconds=time.monotonic() - t0, prompt_length=len(prompt))
                validate_score(scored, route, len(prompt))
                path = output / 'scores' / name / f"{route['route_id']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                publish(path, scored)
                if f'{name}_greedy' in route['roles'] and scored['first_non_argmax'] is not None:
                    failures = [p for p in scored['positions'] if not p['target_is_argmax']]
                    publish(output / f'{name}_parity_failure.json', dict(route=route, mismatch_positions=failures))
                    raise ValueError(f'{name} retained greedy winner parity failure: {eid}, {len(failures)} positions')
                if route is smoke:
                    publish(output / f'{name}_real_entry_smoke.json', dict(route_id=route['route_id'], parity=True,
                            score_sha256=file_hash(path), publication_reload=True, forwards=forwards, cumulative_seconds=time.monotonic() - started))
                print(json.dumps(dict(model=name, forwards=forwards, eid=eid, length=scored['length'], elapsed=time.monotonic() - started)), flush=True)
            publish(output / f'{name}_resources.json', dict(peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved=torch.cuda.max_memory_reserved(), cumulative_elapsed=time.monotonic() - started))
            del qwen, inputs, grid
            gc.collect()
            torch.cuda.empty_cache()
        result = reduce(output)
        publish(output / 'reduction.json', result)
        status = 'completed'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        signal.alarm(0)
        publish(output / f'terminal{suffix}.json', dict(status=status, error=error, model_loads=loads, model_receipts=model_receipts,
                actual_score_forwards=forwards, total_scored_positions=scored_positions,
                cumulative_model_execution_seconds=time.monotonic() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                disk_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'execute', 'execute-post', 'reduce', 'credit'])
    p.add_argument('--output', type=Path, default=OUTPUT)
    args = p.parse_args()
    if args.command == 'prepare':
        packet = prepare(args.output)
        print(json.dumps(dict(witnesses=len(packet['witnesses']), images=len(packet['groups']), routes=len(packet['routes']))))
    elif args.command in ('execute', 'execute-post'):
        execute(args.output, post_only=args.command == 'execute-post')
    elif args.command == 'credit':
        result = credit_summary(args.output)
        require(result == load_canonical_json(args.output / 'credit.json'), 'credit replay changed')
        print(json.dumps(dict(candidates=len(result['candidates']), greedy_routes=len(result['greedy_comparisons']))))
    else:
        result = reduce(args.output)
        require(result == load_canonical_json(args.output / 'reduction.json'), 'reducer replay changed')
        print(json.dumps({k: result[k] for k in ('samples', 'images', 'unique_routes_per_model', 'strata', 'mean_image_delta_logprob')}))


if __name__ == '__main__':
    main()
