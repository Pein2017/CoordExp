"""One frozen completed-row crossover; preparation never loads a model."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import time

from probes import matched_coordinate_branches as m

b = m.b
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-03/completed-row-crossover-03'
PREDECESSOR = Path('/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/matched-coordinate-branches-02')
ACCEPTANCE_SHA = '5ded25c857e723fdcdf1d0fc1f3a66e222617a5bc150c9ac67e0e4eaf7bbceb6'
TERMINAL_SHA = 'd464ca98cc10cccc32a3ac5c03d2fb72a6dae6e968cc1a655139b6d99be36196'
RELEASE_SHA = '3c3d9a51b8fc4f4105d0ed149d31d17aa514435b32fa5fa58e7b93643a24e6db'
SITE = '351017-31'
ORDER = ['C', 'B', 'H', 'H', 'B', 'C']
CONDITIONS = dict(C=[0, 0, 33, 86], B=[72, 0, 999, 999], H=[0, 0, 999, 999])
BOUNDS = dict(m.BOUNDS, continuation_requests=6, requests=6, generated_tokens=18288)


def source_paths():
    return m.source_paths() + ['probes/completed_row_crossover.py',
                               'tests/probes/test_completed_row_crossover.py']


def inherited_fields(old):
    c = {k: old[k] for k in m.FIELDS}
    c.update(images=[351017], sites=[next(s for s in old['sites'] if s['site_id'] == SITE)],
             input_reports=[r for r in old['input_reports'] if r['image_id'] == 351017],
             raw_paths={'351017': old['raw_paths']['351017']})
    return c


def accepted_history():
    """Bind finalized acceptance and saved payloads, never old source to new HEAD."""
    paths = {str(PREDECESSOR / name): sha for name, sha in [
        ('lead-acceptance-01.json', ACCEPTANCE_SHA),
        ('native-terminal-candidate-02.json', TERMINAL_SHA),
        ('released-contract-01.json', RELEASE_SHA)]}
    for path, expected in paths.items():
        assert b.sha(Path(path)) == expected, 'accepted predecessor identity drift'
    accepted = b.load(PREDECESSOR / 'lead-acceptance-01.json')
    terminal = b.load(PREDECESSOR / 'native-terminal-candidate-02.json')
    old = b.load(PREDECESSOR / 'released-contract-01.json')
    assert accepted['status'] == 'lead-accepted'
    assert accepted['execution_source_commit'] == old['source']['commit'] == terminal['execution_source_commit']
    assert terminal['native_exit'] == terminal['readback_exit'] == 0
    assert terminal['cleanup']['engine_close_preceded_terminal_publish']
    assert all(terminal['cleanup']['owned_processes_absent'].values())
    for key in ['worker_candidate', 'released_contract', 'terminal', 'readback']:
        item = accepted[key]
        if item['path'] in paths:
            assert paths[item['path']] == item['sha256']
        else:
            assert terminal['artifacts'][item['path']] == item['sha256']
            paths[item['path']] = item['sha256']
    paths.update(old['bindings'])
    case = next(x for x in terminal['cases'] if x['site_id'] == SITE)
    paths[case['site_ledger']['path']] = case['site_ledger']['sha256']
    ledger = b.load(case['site_ledger']['path'])
    complete = b.load(accepted['terminal']['path'])
    readback = b.load(accepted['readback']['path'])
    assert complete['sites'] == readback['sites']
    assert next(s for s in complete['sites'] if s['site_id'] == SITE) == ledger
    assert complete['artifacts'][SITE + '/complete.json'] == case['site_ledger']['sha256']
    raw_paths = {}
    for condition in ['C', 'B']:
        raw_paths[condition] = []
        for repeat, item in enumerate(case['conditions'][condition]['artifacts'], 1):
            name = f'{condition}-{repeat}.json'
            assert item['path'] == 'native-01/' + SITE + '/' + name
            assert ledger['artifacts'][name] == item['sha256']
            path = str(PREDECESSOR / item['path'])
            paths[path] = item['sha256']; raw_paths[condition].append(path)
    for path, expected in paths.items():
        assert b.sha(Path(path)) == expected, path
    return old, paths, raw_paths


def row_boundary(c, tokenizer):
    artifacts = {k: [b.load(p) for p in paths] for k, paths in c['accepted_raw_paths'].items()}
    reference = artifacts['C'][0]['raw']['token_ids']
    site = c['sites'][0]
    assert site['site_id'] == SITE and site['row']['positions'] == list(range(27, 36))
    assert site['row']['coordinate_positions'] == [31, 32, 33, 34]
    proof = {}
    for condition, pair in artifacts.items():
        for repeat, value in enumerate(pair, 1):
            raw = value['raw']; ids = raw['token_ids']
            assert value['request']['condition'] == condition and value['request']['repeat'] == repeat
            assert raw['raw_identity'] == b.online.seal(raw, raw['producer'])['raw_identity']
            assert ids[:31] == reference[:31] and ids[35] == reference[35]
            assert ids[31:35] == [c['coordinate_ids'][v] for v in CONDITIONS[condition]]
            assert ids[35] == tokenizer.convert_tokens_to_ids('<|box_end|>')
            observed, _ = b.online.observations(dict(raw, arm='greedy'), tokenizer)
            row = next(r for r in observed if r['positions'][0] == 27)
            assert row['positions'] == list(range(27, 36)) and row['description'] == 'person'
            assert row['coordinate_positions'] == [31, 32, 33, 34] and row['bbox'] == CONDITIONS[condition] and row['valid']
            proof[f'{condition}-{repeat}'] = dict(row_token_ids=ids[27:36],
                raw_identity=raw['raw_identity'], positions=row['positions'], exclusive_end=36)
    return dict(site_id=SITE, start=27, exclusive_end=36, coordinate_positions=[31, 32, 33, 34],
                closing_token_id=reference[35], header_token_ids=reference[27:31], accepted_rows=proof)


def request_layout(c):
    raw = b.load(c['accepted_raw_paths']['C'][0])['raw']
    assert c['sites'][0]['site_id'] == SITE
    assert raw['prompt_token_ids'] == b.load(c['raw_paths']['351017'])['prompt_token_ids']
    result = []; seen = dict.fromkeys(CONDITIONS, 0)
    for condition in ORDER:
        seen[condition] += 1
        prefix = list(raw['token_ids'][:36])
        prefix[31:35] = [c['coordinate_ids'][v] for v in CONDITIONS[condition]]
        assert len(prefix) == 36 and prefix[35] == c['row_boundary']['closing_token_id']
        assert len(raw['prompt_token_ids']) + len(prefix) + 3048 <= BOUNDS['context']
        result.append(dict(request_index=len(result), site_id=SITE, condition=condition, repeat=seen[condition],
            condition_name='completed_row', supplied_token_id=prefix[31], supplied_coordinates=CONDITIONS[condition],
            supplied_row_positions=list(range(27, 36)), extension=prefix, budget=b.CAP-len(prefix),
            prefix_tokens=len(prefix), processed_prompt_token_ids=raw['prompt_token_ids']+prefix,
            input_prefix_identity=b.online.identity(raw['prompt_token_ids']+prefix)))
    assert {r['budget'] for r in result} == {3048} and sum(r['budget'] for r in result) == 18288
    return result


def validate_contract(path, require_release=False, current_source=True):
    c = b.load(path)
    assert c['schema'] == 'completed-row-crossover-v1'
    assert c['order'] == ORDER and c['conditions'] == CONDITIONS and c['bounds'] == BOUNDS
    assert type(c['native_released']) is bool
    if require_release:
        assert c['native_released'], 'lead release required'
        assert Path(c['execution_checkout']).resolve() == ROOT.resolve()
        assert Path(c['output_root']).resolve() == OUT.resolve()
    if current_source:
        m.verify_source_identity(c['source'], required_paths=source_paths(), root=ROOT)
    old, paths, raws = accepted_history()
    assert c['predecessor'] == dict(acceptance=str(PREDECESSOR/'lead-acceptance-01.json'),
        acceptance_sha256=ACCEPTANCE_SHA, historical_source_commit=old['source']['commit'])
    assert c['evidence_bindings'] == paths and c['accepted_raw_paths'] == raws
    for key, expected in inherited_fields(old).items():
        assert c[key] == expected, key
    checkpoint = Path(c['checkpoint'])
    assert b.load(checkpoint/'identity.json') == c['checkpoint_files']
    for name, expected in c['checkpoint_files'].items():
        assert b.sha(checkpoint/name) == expected, name
    for package, expected in c['runtime'].items():
        assert version(package) == expected, package
    labels = b.load(c['label_path'])
    assert len(labels) == 18 and sum(len(i['objects']) for i in labels) == 570
    assert len(next(i for i in labels if i['image_id'] == 351017)['objects']) == 49
    assert c['requests'] == request_layout(c), 'frozen row/order/token/budget drift'
    return c


def frontend(c):
    q, inputs, images, requests = m.frontend(c)
    assert c['row_boundary'] == row_boundary(c, q.tokenizer), 'completed-row boundary drift'
    assert c['requests'] == request_layout(c), 'frozen row/order/token/budget drift'
    return q, inputs, images, requests


def prepare(output):
    old, paths, raws = accepted_history()
    c = inherited_fields(old)
    c.update(schema='completed-row-crossover-v1', source=None, native_released=False,
        output_root=str(OUT), execution_checkout=None, order=ORDER, conditions=CONDITIONS, bounds=BOUNDS,
        evidence_bindings=paths, accepted_raw_paths=raws,
        predecessor=dict(acceptance=str(PREDECESSOR/'lead-acceptance-01.json'),
            acceptance_sha256=ACCEPTANCE_SHA, historical_source_commit=old['source']['commit']))
    q = b.rows.frontend(); assert q.model is None
    c['row_boundary'] = row_boundary(c, q.tokenizer)
    c['requests'] = request_layout(c)
    output.mkdir(parents=True, exist_ok=False); b.write(output/'contract.json', c)
    validate_contract(output/'contract.json', current_source=False); frontend(c)
    b.write(output/'cpu-layout.json', dict(status='candidate', native_launched=False,
        row_boundary=c['row_boundary'], requests=c['requests'], bounds=BOUNDS,
        processed_input_lengths=sorted({len(r['processed_prompt_token_ids']) for r in c['requests']})))
    return c


def qualify(path, destination):
    c = validate_contract(path, current_source=False); frontend(c)
    assert c['source'] is None and c['native_released'] is False
    c.update(source=m.capture_source_identity(source_paths(), root=ROOT),
             execution_checkout=str(ROOT), output_root=str(OUT))
    b.write(destination, c)


def summarize(artifacts):
    assert [a['request']['condition'] for a in artifacts] == ORDER
    groups = {k: [a for a in artifacts if a['request']['condition'] == k] for k in CONDITIONS}
    stability = {}
    for name, pair in groups.items():
        assert len(pair) == 2
        stability[name] = dict(request_identity=pair[0]['request']['input_prefix_identity']==pair[1]['request']['input_prefix_identity'],
            semantic=pair[0]['measurements']['primary_semantic_vector']==pair[1]['measurements']['primary_semantic_vector'],
            exact_tokens_and_stop=pair[0]['raw']['token_ids']==pair[1]['raw']['token_ids'] and pair[0]['raw']['stop_reason']==pair[1]['raw']['stop_reason'])
        assert stability[name]['request_identity']
    comparisons = {}
    for left, right in [('C', 'B'), ('C', 'H'), ('B', 'H')]:
        stable = stability[left]['semantic'] and stability[right]['semantic']
        contrasts = [{part: b.transitions(a['measurements'][key], z['measurements'][key])
            for part, key in [('whole','whole'), ('later_free','later_free'),
                ('later_new_unique','later_new_unique'), ('assisted_row','current_row')]}
            for a in groups[left] for z in groups[right]]
        point = {part: {mode: contrasts[0][part][mode] if stable and all(x[part][mode]==contrasts[0][part][mode] for x in contrasts) else None
            for mode in ['raw', 'category']} for part in contrasts[0]}
        ranges = {part: {mode: {key: [min(len(x[part][mode][key]) for x in contrasts), max(len(x[part][mode][key]) for x in contrasts)]
            for key in ['gained', 'lost', 'retained']} for mode in ['raw', 'category']} for part in contrasts[0]}
        burden_deltas = {key: [z['measurements']['burdens'][key]-a['measurements']['burdens'][key]
            for a in groups[left] for z in groups[right]] for key in groups[left][0]['measurements']['burdens']}
        comparisons[left+'-'+right] = dict(semantic_stable=stable, point=point if stable else None,
            observed_count_ranges=ranges, burden_delta_ranges={k:[min(v),max(v)] for k,v in burden_deltas.items()},
            burden_delta_point={k:v[0] if stable and min(v)==max(v) else None for k,v in burden_deltas.items()},
            scope='selected completed-row conditional contrast; independently matched subsets nonadditive')
    burden_ranges = {name: {key: [min(a['measurements']['burdens'][key] for a in pair), max(a['measurements']['burdens'][key] for a in pair)]
        for key in pair[0]['measurements']['burdens']} for name,pair in groups.items()}
    return dict(stability=stability, comparisons=comparisons, burden_ranges=burden_ranges)


def artifact(c, request, acquired, q, inputs, images, expected_sha):
    value = m.artifact(c, request, acquired, q, inputs, images, expected_sha)
    producer = dict(value['raw']['producer'], kind='native_completed_row_crossover')
    value['raw'] = b.online.seal(value['raw'], producer)
    first = value['measurements']['first_row']
    assert first['complete'] and first['strict_valid'] and first['start'] == 27 and first['end'] == 36
    assert first['assisted_credit_only'] and request['supplied_row_positions'] == list(range(27, 36))
    value['credit_boundary'] = dict(inherited_exclusive_end=27, supplied_row_positions=list(range(27, 36)),
        later_free_start=36, supplied_row_natural_recovery_credit=False,
        later_new_unique_excludes_inherited_and_assisted_geometric_owners=True)
    # Historical suffix equality is descriptive; it never stops the six-request schedule.
    if request['condition'] in ['C', 'B']:
        old = b.load(c['accepted_raw_paths'][request['condition']][request['repeat']-1])['raw']
        value['historical_suffix_equality'] = dict(tokens=acquired['token_ids']==old['token_ids'][36:],
            stop=acquired['stop_reason']==old['stop_reason'])
    else:
        value['historical_suffix_equality'] = None
    return value


def run(path, output, expected_sha):
    assert b.sha(path) == expected_sha, 'lead-released contract identity drift'
    c = validate_contract(path, require_release=True)
    assert output.resolve().parent == Path(c['output_root']).resolve(), 'execution output owner drift'
    q, inputs, images, requests = frontend(c)
    from src.qwen.vllm_rollout import VllmDoraRollout
    output.mkdir(parents=True, exist_ok=False); directory=output/SITE; directory.mkdir()
    started=time.monotonic(); artifacts=[]; hashes={}
    counters=dict(requests=0, continuation_requests=0, score_requests=0, generated_tokens=0)
    with VllmDoraRollout(base_model=c['base_model'], checkpoint=c['checkpoint'], identity=c['weight_identity'],
            log_path=output/'vllm.log', device=0, trainer_rank=0, max_model_len=BOUNDS['context'],
            max_num_seqs=1, max_logprobs=-1, kv_cache_memory_bytes=BOUNDS['kv_cache_bytes'],
            seed=c['generation']['seed'], timeout=BOUNDS['whole_wall_seconds']) as engine:
        engine.configure_coordinate_output_norm('off', c['coordinate_ids'], identity=c['weight_identity'])
        for request in c['requests']:
            assert time.monotonic()-started < BOUNDS['whole_wall_seconds']
            acquired=engine.generate_exact([requests[351017]], chat_token_ids=[c['input_reports'][0]['unexpanded_chat_token_ids']],
                extensions=[request['extension']], budgets=[request['budget']],
                eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'), pad_token_id=q.tokenizer.pad_token_id,
                identity=c['weight_identity'], vocab_size=c['vocab_size'], full_scores=False)[0]
            value=artifact(c,request,acquired,q,inputs,images,expected_sha)
            name=f"{request['condition']}-{request['repeat']}.json"
            b.write(directory/name,value); hashes[name]=b.sha(directory/name); artifacts.append(value)
            counters['requests']+=1; counters['continuation_requests']+=1; counters['generated_tokens']+=value['free_tokens']
            assert all(counters[k] <= BOUNDS[k] for k in counters)
        startup=engine.startup; operations=list(engine.receipts)
    ledger=dict(site_id=SITE, artifacts=hashes, **summarize(artifacts)); b.write(directory/'complete.json',ledger)
    b.write(output/'complete.json',dict(schema='completed-row-terminal-v1', status='candidate',
        contract_sha256=expected_sha, sites=[ledger], counters=counters, startup=startup, operations=operations,
        artifacts={SITE+'/complete.json':b.sha(directory/'complete.json')}, seconds=time.monotonic()-started,
        HF_forwards=0, optimizer_steps=0, next_unit_scheduled=False))


def readback(path, output, expected_sha):
    assert b.sha(path) == expected_sha, 'lead-released contract identity drift'
    c=validate_contract(path,require_release=True)
    assert output.resolve().parent == Path(c['output_root']).resolve(), 'execution output owner drift'
    complete=b.load(output/'complete.json')
    assert complete['schema']=='completed-row-terminal-v1' and complete['status']=='candidate'
    assert complete['contract_sha256']==expected_sha
    q,inputs,images,_=frontend(c); directory=output/SITE
    assert set(complete['artifacts'])=={SITE+'/complete.json'} and len(complete['sites'])==1
    assert b.sha(directory/'complete.json')==complete['artifacts'][SITE+'/complete.json']
    ledger=b.load(directory/'complete.json'); assert complete['sites']==[ledger]
    assert set(ledger['artifacts'])=={f"{r['condition']}-{r['repeat']}.json" for r in c['requests']}
    artifacts=[]; counters=dict(requests=0,continuation_requests=0,score_requests=0,generated_tokens=0)
    for request in c['requests']:
        name=f"{request['condition']}-{request['repeat']}.json"
        assert b.sha(directory/name)==ledger['artifacts'][name]
        value=b.load(directory/name)
        assert value==artifact(c,request,value['acquisition'],q,inputs,images,expected_sha), 'false request/raw/semantic credit'
        artifacts.append(value); counters['requests']+=1; counters['continuation_requests']+=1; counters['generated_tokens']+=value['free_tokens']
    assert ledger==dict(site_id=SITE,artifacts=ledger['artifacts'],**summarize(artifacts)), 'false stability/contrast'
    assert counters==complete['counters'] and counters['requests']==6 and all(counters[k]<=BOUNDS[k] for k in counters)
    assert complete['HF_forwards']==complete['optimizer_steps']==0 and complete['next_unit_scheduled'] is False
    from src.qwen.vllm_rollout import validate_device_receipt
    startup=complete['startup']; assert startup['identity']==c['weight_identity']
    validate_device_receipt(startup['device'],startup['device']['requested'])
    assert startup['device']['requested']['rank']==startup['device']['requested']['device']==0
    assert [op['operation'] for op in complete['operations']]==['coordinate_output_norm']+['generate_exact']*6
    for op in complete['operations']:
        assert op['identity']==c['weight_identity'] and op['coordinate_output_norm']['mode']=='off'
    for op in complete['operations'][1:]: m.validate_policy(op['coordinate_output_norm'],'off',c)
    return dict(schema='completed-row-readback-v1', status='candidate', counters=counters, sites=[ledger],
        terminal_sha256=b.sha(output/'complete.json'), contract_sha256=expected_sha,
        predecessor=c['predecessor'], row_boundary=c['row_boundary'], annotation_denominator=570,
        selected_image_annotation_denominator=49,
        scope='Selected supplied completed-row contrast; no natural, population or physical-negative claim')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','qualify','validate','run','readback'])
    parser.add_argument('--contract',type=Path); parser.add_argument('--contract-sha256')
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    if args.command=='prepare': prepare(args.output)
    elif args.command=='qualify': qualify(args.contract,args.output)
    elif args.command=='validate': validate_contract(args.contract)
    elif args.command=='run': run(args.contract,args.output,args.contract_sha256)
    else: b.write(args.output/'readback.json',readback(args.contract,args.output,args.contract_sha256))


if __name__=='__main__': main()
