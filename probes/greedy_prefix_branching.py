"""Finite native coordinate interventions; CPU preparation never loads a model."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import time

from probes import online_row_credit as online
from probes import rollout_row_credit as rows
from probes.full_label_fit.coord_norm import load, write
from probes.full_label_fit.experiment import sha
from probes.full_label_fit.region import acceptable_bins
from src.artifacts.git_identity import capture_source_identity, verify_source_identity

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / 'research/experiments/2026-10-03-greedy-prefix-branching'
OUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01'
OLD = ROOT / 'outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/coord-norm-01'
IMAGES = [7511, 351017]
CAP = 3084
BRANCHES = ['greedy', 'sham', 'alternative1', 'alternative2']


def source_paths():
    # Clean commit/tree binds the checkout; list the decision-bearing producers/consumers.
    return [
        'probes/greedy_prefix_branching.py', 'probes/online_row_credit.py',
        'probes/rollout_row_credit.py', 'probes/iterative_positive.py',
        'probes/hidden_human_recovery.py', 'probes/full_label_fit/coord_norm.py',
        'probes/full_label_fit/experiment.py', 'probes/full_label_fit/recipe.py',
        'probes/full_label_fit/region.py', 'src/artifacts/git_identity.py',
        'src/adapters/dora.py', 'src/qwen/vllm_rollout.py', 'src/qwen/vllm_dora_model.py',
        'src/qwen/native.py', 'src/qwen/generation.py', 'src/qwen/runtime_loading.py',
        'src/qwen/images.py', 'src/qwen/tokens.py', 'src/qwen/patches.py',
        'src/qwen/untied_embeddings.py', 'src/qwen/special_token_embeddings.py',
        'src/inference/parsing.py', 'src/inference/token_text.py', 'src/inference/vllm_backend.py',
        'src/eval/saved_rows.py', 'src/eval/assignment.py',
        'tests/probes/test_greedy_prefix_branching.py', 'tests/qwen/test_vllm_rollout.py',
    ]


def legal_range(prefix):
    """Strict-positive box completion, using only already supplied coordinates."""
    assert len(prefix) <= 3 and all(type(x) is int and 0 <= x <= 999 for x in prefix)
    if len(prefix) < 2:
        return 0, 999  # x1/y1=999 cannot admit a positive-width/height box.
    if prefix[0] == 999 or (len(prefix) >= 2 and prefix[1] == 999):
        return 0, 0
    return prefix[len(prefix)-2] + 1, 1000


def select_sites(record, tokenizer):
    observations, malformed = online.observations(dict(record, arm='greedy'), tokenizer)
    selections = []
    invalid = next((r for r in observations if not r['valid']), None)
    repeated = next((r for r in observations if not r['first']), None)
    for event, row in [('invalid', invalid), ('literal_repeat', repeated)]:
        if row is None:
            selections.append(dict(image_id=record['image_id'], event=event,
                eligible=False, skipped_reason='no_complete_event'))
            continue
        slot = 0
        if event == 'invalid':
            slot = next(j for j, value in enumerate(row['bbox'])
                if not legal_range(row['bbox'][:j])[0] <= value < legal_range(row['bbox'][:j])[1])
        position = row['coordinate_positions'][slot]
        lo, hi = legal_range(row['bbox'][:slot])
        site = dict(image_id=record['image_id'], event=event, row=row, slot=slot,
            position=position, causal_logits_position=len(record['prompt_token_ids'])+position-1,
            emitted_token_id=record['token_ids'][position], emitted_bin=row['bbox'][slot],
            legal_range=[lo, hi], raw_identity=record['raw_identity'],
            prefix_identity=online.identity(record['prompt_token_ids']+record['token_ids'][:position]),
            remaining_tokens=CAP-position, supplied_remaining_tokens=CAP-position-1,
            eligible=hi-lo >= 3 and position < CAP-1)
        site['site_id'] = f"{record['image_id']}-{position}"
        site['skipped_reason'] = None if site['eligible'] else 'insufficient_legal_support_or_budget'
        selections.append(site)
    seen = set()
    for site in selections:
        if not site['eligible']:
            continue
        if site['position'] in seen:
            site.update(eligible=False, skipped_reason='same_causal_site')
        seen.add(site['position'])
    return selections, dict(complete_rows=len(observations), malformed_or_censored=len(malformed))


def branch_prefix(record, site, supplied=None):
    p = site['position']
    assert site['prefix_identity'] == online.identity(record['prompt_token_ids']+record['token_ids'][:p])
    assert site['emitted_token_id'] == record['token_ids'][p]
    extension = list(record['token_ids'][:p])
    if supplied is not None:
        assert type(supplied) is int and supplied >= 0
        extension.append(supplied)
    budget = CAP-len(extension)
    assert budget > 0 and len(extension)+budget == CAP
    return extension, budget


def owner_regions(image, description, prefix):
    """Retrospective sets only; never used by native candidate selection."""
    per_owner = {str(o['coco_ann_id']): acceptable_bins(o['bbox_2d'], list(prefix))
                 for o in image['objects'] if o['desc'] == description}
    union = sorted({v for values in per_owner.values() for v in values})
    return dict(per_owner=per_owner, union=union,
        completable_owner_ids=sorted(int(k) for k,v in per_owner.items() if v))


def native_candidates(scores, coordinate_ids, site, vocab_size):
    assert set(scores) == set(range(vocab_size)), 'native full-vocabulary scores incomplete'
    assert all(not math.isnan(v) and v != math.inf for v in scores.values()), 'NaN/+inf native scores'
    assert abs(sum(math.exp(v) for v in scores.values())-1) < 1e-4, 'not full-vocabulary normalized logprobs'
    lo, hi = site['legal_range']
    legal = coordinate_ids[lo:hi]
    candidates = sorted((t for t in legal if t != site['emitted_token_id']),
                        key=lambda t: (-scores[t], t))[:2]
    assert len(candidates) == 2 and len(set(candidates)) == 2
    return [dict(token_id=t, bin=coordinate_ids.index(t), raw_logprob=('-inf' if scores[t] == -math.inf else scores[t])) for t in candidates]


def score_region(scores, token_ids):
    selected = set(token_ids)
    gap = (max(scores[t] for t in selected)-max(v for t,v in scores.items() if t not in selected)) if selected and len(selected)<len(scores) else None
    if gap is not None and not math.isfinite(gap):
        gap = '-inf' if gap == -math.inf else '+inf' if gap == math.inf else None
    return dict(count=len(selected), mass=sum(math.exp(scores[t]) for t in selected),
        best_acceptable_minus_best_unacceptable=gap,
        denominator='complete_native_vocabulary', vocabulary_size=len(scores), backend='vllm_raw_logprobs_norm_off')


def score_sites(scores, site, image, coordinate_ids):
    lo, hi = site['legal_range']
    region = owner_regions(image, site['row']['description'], site['row']['bbox'][:site['slot']])
    return dict(legal=score_region(scores, coordinate_ids[lo:hi]),
        owner_union=score_region(scores, [coordinate_ids[b] for b in region['union']]),
        per_owner={key: score_region(scores, [coordinate_ids[b] for b in values])
                   for key,values in region['per_owner'].items()}, retrospective_regions=region)


def evaluate_branch(image, record, site, tokenizer, supplied):
    """Same annotation matcher; inherited, assisted, and later rows remain separate."""
    record = dict(record, arm='greedy')
    parsed = rows.parse(record)
    encoded = rows.aligned_tokens(record, tokenizer)
    start = site['row']['positions'][0]
    start_char = encoded['offset_mapping'][start][0]
    current = next((r for r in [*parsed.predictions, *parsed.dropped_predictions]
                    if r['char_start'] == start_char), None)
    end = (rows.row_positions(current, encoded)[-1]+1) if current else len(record['token_ids'])
    current_valid = current in parsed.predictions if current else False
    current_complete = current is not None and (current_valid or current.get('reason') == 'geometry_invalid')
    projection = lambda begin, finish: dict(record, token_ids=record['token_ids'][begin:finish],
        text=tokenizer.decode(record['token_ids'][begin:finish], skip_special_tokens=False),
        generated_tokens=finish-begin)
    whole = rows.assess_outputs([image], [], [record])[0]
    inherited = rows.assess_outputs([image], [], [projection(0,start)])[0]
    assisted = rows.assess_outputs([image], [], [projection(start,end)])[0]
    later = rows.assess_outputs([image], [], [projection(end,len(record['token_ids']))])[0]
    def owners(result):
        return {mode:sorted(result['ids'][mode]['retained']) for mode in ('raw','category')}
    coordinates = {tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>'):i for i in range(1000)}
    trace = []
    prefix = []
    for j in range(4):
        index = site['row']['coordinate_positions'][0]+j
        regions = owner_regions(image, site['row']['description'], prefix)
        token = record['token_ids'][index] if index < len(record['token_ids']) else None
        value = coordinates.get(token)
        trace.append(dict(slot=j, position=index, emitted_token_id=token, emitted_bin=value,
            prefix_bins=list(prefix), regions=regions,
            surviving_owner_ids=sorted(int(k) for k,v in regions['per_owner'].items() if value in v)))
        if value is None:
            break
        prefix.append(value)
    return dict(whole=owners(whole), inherited=owners(inherited),
        current_row=owners(assisted), later_free=owners(later), burdens=whole['burdens'],
        first_row=dict(complete=current_complete, strict_valid=current_valid, supplied=supplied,
            assisted_credit_only=supplied is not None, start=start, end=end,
            parser_reason=current.get('reason') if current else 'censored_or_unaligned'),
        owner_prefix_trace=trace, denominator_ids=whole['denominator_ids']['retained'],
        accounting='Each subset independently uses cardinality-first one-to-one geometry then exact description; subsets are not additive. Supplied-row gains are assisted, inherited rows earn no new free credit.',
        annotation_unmatched_is_physical_negative=False)


def transitions(before, after):
    return {mode:dict(gained=sorted(set(after[mode])-set(before[mode])),
        lost=sorted(set(before[mode])-set(after[mode])), retained=sorted(set(before[mode])&set(after[mode])))
        for mode in ('raw','category')}


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    historical = load(OLD/'contract.json')
    state = load(ROOT/'research/experiments/2026-10-02-full-label-self-rollout-fit/state.json')['coord_norm_followup']
    bindings = dict(historical['input_sha256'])
    bindings.update(historical['checkpoint_acceptance'])
    bindings[str(OLD/'contract.json')] = state['contract']['sha256']
    bindings[state['lead_acceptance']['path']] = state['lead_acceptance']['sha256']
    bindings[state['terminal_candidate']['path']] = state['terminal_candidate']['sha256']
    for path, expected in bindings.items():
        assert sha(Path(path)) == expected, path
    checkpoint = Path(historical['checkpoint'])
    assert load(checkpoint/'identity.json') == historical['checkpoint_files']
    for path, expected in historical['checkpoint_files'].items():
        assert sha(checkpoint/path) == expected, path
    q = rows.frontend()
    assert q.model is None and str(q.base_model_path) == historical['base_model']
    assert q.base_config_sha256 == historical['base_config_sha256'] and q.tokenizer_sha256 == historical['tokenizer_sha256']
    coordinate_ids = [q.tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)]
    assert coordinate_ids == historical['coordinate_ids']
    inputs = {r['image_id']:r for r in load(historical['input_path'])}
    labels = load(historical['label_path'])
    assert len(labels) == 18 and sum(len(r['objects']) for r in labels) == 570
    sites, selection, raw_paths, input_reports = [], {}, {}, []
    for i in IMAGES:
        paths = list((OLD/'native-01/off').glob(f'rank-*/{i}.json'))
        assert len(paths) == 1
        path = paths[0]
        complete = load(path.parent/'complete.json')
        assert sha(path) == complete['artifacts'][path.name]
        record = load(path)
        assert record['raw_identity'] == online.seal(record, record['producer'])['raw_identity']
        assert record['producer'] == dict(kind='fixed_checkpoint_coord_norm_inference', condition='off',
            weight_identity=historical['weight_identity'], source=historical['source'],
            contract_sha256=state['contract']['sha256'])
        assert record['generated_tokens'] == len(record['token_ids']) <= CAP
        for key,value in inputs[i].items():
            assert record[key] == value, key
        request = online.vllm_requests(q, inputs, [i])[0]
        batch = online.native_batch(q, inputs[i])
        # CPU-only native assembly checks expanded original prompt, media and grid.
        assert list(batch.prompt_token_ids[0]) == record['prompt_token_ids']
        assert list(batch.image_grids[0]) == record['image_grid_thw']
        assert batch.media_sha256[0] == record['media_sha256']
        input_reports.append(dict(image_id=i, prompt_tokens=len(record['prompt_token_ids']),
            visual_tokens=math.prod(record['image_grid_thw'])//4, media_sha256=batch.media_sha256[0],
            unexpanded_chat_token_ids=q.tokenizer.encode(request.chat_text, add_special_tokens=False)))
        chosen, disposition = select_sites(record, q.tokenizer)
        # Installed native token replacement, on each actual causal prefix; no text fallback.
        from vllm.multimodal.processing.processor import PromptReplacement, _apply_token_matches_with_placeholders
        image_token = q.processor.image_token_id
        visual = math.prod(record['image_grid_thw'])//q.processor.image_processor.merge_size**2
        update = {'image':[[PromptReplacement(modality='image', target=[image_token],
            replacement=[image_token]*visual).resolve(0)]]}
        chat_tokens = input_reports[-1]['unexpanded_chat_token_ids']
        for site in chosen:
            if not site['eligible']:
                continue
            prefix,_ = branch_prefix(record,site)
            expanded,matches,placeholders = _apply_token_matches_with_placeholders(chat_tokens+prefix,update)
            assert expanded == record['prompt_token_ids']+prefix
            assert matches == {'image':[0]} and placeholders['image'][0].length == visual
        input_reports[-1]['installed_token_placeholder_expansion'] = 'exact_at_all_eligible_sites'

        sites.extend(chosen); selection[str(i)] = disposition
        raw_paths[str(i)] = str(path)
        bindings[str(path)] = complete['artifacts'][path.name]
    assert len([s for s in sites if s['eligible']]) <= 4
    contract = dict(schema='greedy-prefix-branching-v1', images=IMAGES, annotation_denominator=570,
        generation=historical['generation'], norm='off', branches=BRANCHES,
        checkpoint=str(checkpoint), checkpoint_files=historical['checkpoint_files'],
        checkpoint_acceptance=historical['checkpoint_acceptance'], weight_identity=historical['weight_identity'],
        base_model=historical['base_model'], base_config_sha256=q.base_config_sha256,
        tokenizer_sha256=q.tokenizer_sha256, coordinate_ids=coordinate_ids,
        vocab_size=q.config.text_config.vocab_size, runtime=historical['runtime'],
        input_path=historical['input_path'], label_path=historical['label_path'],
        bindings=bindings, raw_paths=raw_paths, input_reports=input_reports,
        sites=sites, selection=selection, output_root=str(OUT), native_released=False,
        source=None, bounds=dict(gpus=1, ranks=1, concurrent_sequences=1,
            score_requests=4, continuation_requests=16, requests=20,
            context=max(len(inputs[i]['prompt_token_ids'])+CAP for i in IMAGES),
            generated_tokens=sum(1+4*(CAP-s['position'])-3 for s in sites if s['eligible']),
            whole_wall_seconds=1200, cleanup_seconds=30, kv_cache_bytes=2*1024**3,
            HF_forwards=0, optimizer_steps=0),
        fidelity_stop_rules=dict(compare='full_token_suffix_and_stop_reason',
            score_greedy_mismatch='HOLD_site_no_alternatives_no_retry',
            historical_greedy_or_sham_mismatch='HOLD_site_no_alternatives_no_retry',
            prompt_media_checkpoint_or_full_score_failure='stop_package_preserve_evidence_no_retry',
            success='finish_only_declared_branches_no_promotion'))
    write(output/'contract.json', contract)
    write(output/'cpu-sites.json', dict(status='candidate', sites=sites, selection=selection,
        native_launched=False, checkpoint_payload_verified=True, inputs=input_reports))
    return contract


def validate_contract(path, require_release=False):
    c = load(path)
    assert c['schema'] == 'greedy-prefix-branching-v1' and c['images'] == IMAGES
    assert c['norm'] == 'off' and c['branches'] == BRANCHES
    assert c['generation'] == dict(temperature=0, top_p=1, top_k=-1, repetition_penalty=1, min_tokens=0, cap=CAP, seed=92711)
    assert c['annotation_denominator'] == 570
    assert len(c['coordinate_ids']) == len(set(c['coordinate_ids'])) == 1000
    assert all(type(t) is int and 0 <= t < c['vocab_size'] for t in c['coordinate_ids'])
    if require_release:
        assert c['native_released'] is True, 'lead release required'
    verify_source_identity(c['source'], required_paths=source_paths(), root=ROOT)
    for file, expected in c['bindings'].items():
        assert sha(Path(file)) == expected, file
    checkpoint = Path(c['checkpoint'])
    assert load(checkpoint/'identity.json') == c['checkpoint_files']
    for file, expected in c['checkpoint_files'].items():
        assert sha(checkpoint/file) == expected, file
    for package, expected in c['runtime'].items():
        assert version(package) == expected, package
    labels = load(c['label_path'])
    assert len(labels) == 18 and sum(len(r['objects']) for r in labels) == 570
    assert len([s for s in c['sites'] if s['eligible']]) <= 4
    return c


def qualify(contract_path, destination):
    c = load(contract_path)
    assert c['source'] is None and c['native_released'] is False
    c['source'] = capture_source_identity(source_paths(), root=ROOT)
    write(destination, c)


def run(contract_path, output, expected_sha256):
    # Imported only after release; preparation has no model/GPU invocation.
    from src.qwen.vllm_rollout import VllmDoraRollout
    assert sha(contract_path) == expected_sha256, 'lead-released contract identity drift'
    c = validate_contract(contract_path, require_release=True)
    output.mkdir(parents=True, exist_ok=False)
    q = rows.frontend()
    assert q.model is None and q.base_config_sha256 == c['base_config_sha256'] and q.tokenizer_sha256 == c['tokenizer_sha256']
    inputs = {r['image_id']:r for r in load(c['input_path'])}
    by_image = {r['image_id']:r for r in load(c['label_path'])}
    requests = {i:online.vllm_requests(q,inputs,[i])[0] for i in IMAGES}
    chat_ids = {r['image_id']:r['unexpanded_chat_token_ids'] for r in c['input_reports']}
    started = time.monotonic()
    ledgers = []
    counters = dict(requests=0, score_requests=0, continuation_requests=0, generated_tokens=0)
    with VllmDoraRollout(base_model=c['base_model'], checkpoint=c['checkpoint'],
            identity=c['weight_identity'], log_path=output/'vllm.log', device=0, trainer_rank=0,
            max_model_len=c['bounds']['context'], max_num_seqs=1, max_logprobs=-1,
            seed=92711, timeout=c['bounds']['whole_wall_seconds']) as engine:
        engine.configure_coordinate_output_norm('off', c['coordinate_ids'], identity=c['weight_identity'])
        for site in c['sites']:
            if not site['eligible']:
                continue
            assert time.monotonic()-started < c['bounds']['whole_wall_seconds']
            record = load(c['raw_paths'][str(site['image_id'])])
            extension,budget = branch_prefix(record,site)
            def acquire(prefix, remaining, full_scores=False):
                result = engine.generate_exact([requests[site['image_id']]], chat_token_ids=[chat_ids[site['image_id']]],
                    extensions=[prefix], budgets=[remaining], eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),
                    pad_token_id=q.tokenizer.pad_token_id, identity=c['weight_identity'], vocab_size=c['vocab_size'], full_scores=full_scores)[0]
                counters['requests'] += 1
                counters['score_requests' if full_scores else 'continuation_requests'] += 1
                counters['generated_tokens'] += len(result['token_ids'])
                for key in counters:
                    assert counters[key] <= c['bounds'][key], key
                return result
            scored = acquire(extension,1,True)
            scores = {int(t):float(v) for t,v in scored['full_scores'].items()}
            candidates = native_candidates(scores,c['coordinate_ids'],site,c['vocab_size'])
            directory = output/site['site_id']; directory.mkdir()
            write(directory/'scores.json', dict(site=site, result=scored,
                candidates=candidates, regions=score_sites(scores,site,by_image[site['image_id']],c['coordinate_ids'])))
            measurements, fidelity, completed_tokens = {}, {}, {}
            for name, supplied in [('greedy',None),('sham',site['emitted_token_id']),
                    ('alternative1',candidates[0]['token_id']),('alternative2',candidates[1]['token_id'])]:
                if name.startswith('alternative') and not all(fidelity.values()):
                    break
                prefix,remaining = branch_prefix(record,site,supplied)
                acquired = acquire(prefix,remaining)
                full = prefix+acquired['token_ids']
                assert len(full) <= CAP
                producer = dict(kind='native_coordinate_branch', source=c['source']['commit'],
                    contract_sha256=expected_sha256, site_id=site['site_id'], branch=name,
                    norm='off', weight_identity=c['weight_identity'])
                result = online.seal(dict(inputs[site['image_id']], token_ids=full,
                    text=q.tokenizer.decode(full,skip_special_tokens=False), generated_tokens=len(full),
                    stop_reason=acquired['stop_reason']), producer)
                measure = evaluate_branch(by_image[site['image_id']],result,site,q.tokenizer,supplied)
                measurements[name] = measure
                completed_tokens[name] = full
                if name in ('greedy','sham'):
                    fidelity[name+'_historical'] = full == record['token_ids'] and acquired['stop_reason'] == record['stop_reason']
                if name == 'sham':
                    fidelity['greedy_sham'] = completed_tokens['greedy'] == full
                if name == 'greedy':
                    fidelity['score_argmax'] = acquired['token_ids'][0] == scored['token_ids'][0]
                baseline = evaluate_branch(by_image[site['image_id']],record,site,q.tokenizer,None)
                write(directory/(name+'.json'),dict(site_id=site['site_id'], branch=name,
                    input_prefix_identity=online.identity(record['prompt_token_ids']+prefix),
                    supplied_token_id=supplied, prefix_tokens=len(prefix), free_tokens=len(acquired['token_ids']),
                    budget=remaining, raw=result, acquisition=acquired, measurements=measure,
                    historical_raw_identity=record['raw_identity'],
                    whole_transitions=transitions(baseline['whole'],measure['whole']),
                    current_greedy_transitions=transitions(measurements['greedy']['whole'],measure['whole']),
                    later_free_transitions=transitions(baseline['later_free'],measure['later_free']),
                    current_row_assisted_transitions=transitions(baseline['current_row'],measure['current_row']) if supplied is not None else None))
            ledger = dict(site_id=site['site_id'], fidelity=fidelity,
                status='candidate' if all(fidelity.values()) else 'HOLD_native_fidelity',
                completed_branches=list(measurements), skipped_branches=[b for b in BRANCHES if b not in measurements])
            ledger['artifacts'] = {p.name:sha(p) for p in sorted(directory.glob('*.json'))}
            write(directory/'complete.json',ledger); ledgers.append(ledger)
        operations = list(engine.receipts)
        startup = engine.startup
    write(output/'complete.json',dict(status='candidate' if all(l['status']=='candidate' for l in ledgers) else 'HOLD_native_fidelity', contract_sha256=sha(contract_path),
        sites=ledgers, counters=counters,
        artifacts={f"{l['site_id']}/complete.json":sha(output/l['site_id']/'complete.json') for l in ledgers},
        operations=operations, startup=startup, seconds=time.monotonic()-started,
        HF_forwards=0, optimizer_steps=0, next_unit_scheduled=False))



def readback(contract_path, output, expected_sha256):
    assert sha(contract_path) == expected_sha256, 'lead-released contract identity drift'
    c = validate_contract(contract_path, require_release=True)
    complete = load(output/'complete.json')
    assert complete['contract_sha256'] == expected_sha256
    q = rows.frontend()
    inputs = {r['image_id']:r for r in load(c['input_path'])}
    images = {r['image_id']:r for r in load(c['label_path'])}
    counters = dict(requests=0, score_requests=0, continuation_requests=0, generated_tokens=0)
    for relative, expected in complete['artifacts'].items():
        assert sha(output/relative) == expected, relative
    assert [l['site_id'] for l in complete['sites']] == [s['site_id'] for s in c['sites'] if s['eligible']]
    for site, ledger in zip([s for s in c['sites'] if s['eligible']],complete['sites'],strict=True):
        directory = output/site['site_id']
        assert load(directory/'complete.json') == ledger
        for relative, expected in ledger['artifacts'].items():
            assert sha(directory/relative) == expected, relative
        expected_branches = BRANCHES if all(ledger['fidelity'].values()) else BRANCHES[:2]
        assert ledger['completed_branches'] == expected_branches
        assert ledger['skipped_branches'] == [name for name in BRANCHES if name not in expected_branches]
        assert ledger['status'] == ('candidate' if all(ledger['fidelity'].values()) else 'HOLD_native_fidelity')
        historical = load(c['raw_paths'][str(site['image_id'])])
        scoring = load(directory/'scores.json')
        assert scoring['site'] == site
        extension,_ = branch_prefix(historical,site)
        acquisition = scoring['result']
        assert acquisition['processed_prompt_token_ids'] == historical['prompt_token_ids']+extension
        assert len(acquisition['token_ids']) == 1 and acquisition['score_semantics'] == 'native_raw_logprobs_full_vocabulary'
        native = {int(t):float(v) for t,v in acquisition['full_scores'].items()}
        candidates = native_candidates(native,c['coordinate_ids'],site,c['vocab_size'])
        assert scoring['candidates'] == candidates
        assert scoring['regions'] == score_sites(native,site,images[site['image_id']],c['coordinate_ids'])
        counters['score_requests'] += 1; counters['requests'] += 1; counters['generated_tokens'] += 1
        baseline = evaluate_branch(images[site['image_id']],historical,site,q.tokenizer,None)
        tokens, stops = {}, {}
        measures = {}
        for name in ledger['completed_branches']:
            artifact = load(directory/(name+'.json'))
            supplied = None if name=='greedy' else site['emitted_token_id'] if name=='sham' else candidates[int(name[-1])-1]['token_id']
            extension,budget = branch_prefix(historical,site,supplied)
            assert artifact['supplied_token_id'] == supplied and artifact['budget'] == budget
            assert artifact['prefix_tokens'] == len(extension) and artifact['historical_raw_identity'] == historical['raw_identity']
            acquired = artifact['acquisition']
            assert acquired['full_scores'] is None
            assert acquired['processed_prompt_token_ids'] == historical['prompt_token_ids']+extension
            from src.qwen.generation import trim_suffix
            checked, reason = trim_suffix(acquired['token_ids'],budget=budget,
                eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),pad_token_id=q.tokenizer.pad_token_id)
            assert list(checked) == acquired['token_ids'] and reason == acquired['stop_reason']
            raw = artifact['raw']
            assert raw['stop_reason'] == acquired['stop_reason']
            expected_producer = dict(kind='native_coordinate_branch',source=c['source']['commit'],
                contract_sha256=expected_sha256,site_id=site['site_id'],branch=name,norm='off',weight_identity=c['weight_identity'])
            assert raw['producer'] == expected_producer
            assert raw['raw_identity'] == online.seal(dict(raw),expected_producer)['raw_identity']
            for key,value in inputs[site['image_id']].items():
                assert raw[key] == value, key
            assert artifact['input_prefix_identity'] == online.identity(historical['prompt_token_ids']+extension)
            assert raw['token_ids'] == extension+acquired['token_ids']
            assert raw['generated_tokens'] == len(raw['token_ids']) <= CAP
            assert artifact['free_tokens'] == len(acquired['token_ids']) <= budget
            measure = evaluate_branch(images[site['image_id']],raw,site,q.tokenizer,supplied)
            assert artifact['measurements'] == measure
            measures[name] = measure; tokens[name] = raw['token_ids']; stops[name] = raw['stop_reason']
            assert artifact['whole_transitions'] == transitions(baseline['whole'],measure['whole'])
            assert artifact['later_free_transitions'] == transitions(baseline['later_free'],measure['later_free'])
            assert artifact['current_greedy_transitions'] == transitions(measures['greedy']['whole'],measure['whole'])
            assisted = transitions(baseline['current_row'],measure['current_row']) if supplied is not None else None
            assert artifact['current_row_assisted_transitions'] == assisted
            counters['requests'] += 1; counters['continuation_requests'] += 1
            counters['generated_tokens'] += len(acquired['token_ids'])
        fidelity = dict(greedy_historical=tokens['greedy']==historical['token_ids'] and stops['greedy']==historical['stop_reason'],
            sham_historical=tokens['sham']==historical['token_ids'] and stops['sham']==historical['stop_reason'],
            greedy_sham=tokens['greedy']==tokens['sham'],
            score_argmax=tokens['greedy'][site['position']]==scoring['result']['token_ids'][0])
        assert ledger['fidelity'] == fidelity
    assert complete['status'] == ('candidate' if all(l['status']=='candidate' for l in complete['sites']) else 'HOLD_native_fidelity')
    assert counters == complete['counters']
    assert all(counters[k] <= c['bounds'][k] for k in counters)
    assert complete['HF_forwards'] == complete['optimizer_steps'] == 0
    from src.qwen.vllm_rollout import validate_device_receipt
    startup = complete['startup']
    assert startup['identity'] == c['weight_identity']
    validate_device_receipt(startup['device'],startup['device']['requested'])
    operations = complete['operations']
    assert len(operations) == 1+counters['requests']
    assert [op['operation'] for op in operations] == ['coordinate_output_norm']+['generate_exact']*counters['requests']
    assert all(op['identity'] == c['weight_identity'] and op['coordinate_output_norm']['mode']=='off' for op in operations)
    from probes.full_label_fit.coord_norm import validate_policy
    for op in operations[1:]:
        validate_policy(op['coordinate_output_norm'],'off',c)
    return dict(status=complete['status'], counters=counters, sites=complete['sites'],
        terminal_sha256=sha(output/'complete.json'), contract_sha256=expected_sha256,
        annotation_denominator=570, native_fidelity_measured=True,
        scope='Selected exact-prefix diagnostic; no empty-history/population efficacy or physical-negative claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','qualify','run','validate','readback'])
    parser.add_argument('--contract',type=Path)
    parser.add_argument('--contract-sha256')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
    elif args.command == 'qualify':
        qualify(args.contract,args.output)
    elif args.command == 'validate':
        validate_contract(args.contract)
    elif args.command == 'readback':
        write(args.output/'readback.json',readback(args.contract,args.output,args.contract_sha256))
    else:
        run(args.contract,args.output,args.contract_sha256)


if __name__ == '__main__':
    main()
