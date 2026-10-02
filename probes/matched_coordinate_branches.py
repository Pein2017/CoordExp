"""Frozen supplied-coordinate continuations with two planned native replicates."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import time

from probes import greedy_prefix_branching as b
from probes.full_label_fit.coord_norm import validate_policy
from src.artifacts.git_identity import capture_source_identity, verify_source_identity
from src.qwen.generation import trim_suffix

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = Path('/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01')
OUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-03/matched-coordinate-branches-02'
FIXED = [('7511-626', [999,982,968]), ('7511-203', [630,637,632]),
         ('351017-1507', [999,966,968]), ('351017-31', [0,1,72])]
ACCEPTANCE_SHA = 'cea0da67d3bf7f1469035af39d6522e734f7d4ac027c76dc2569311e0fa0d6ab'
RELEASE_SHA = '1dbfb7d38fced4e986d8cf909412ba617405903379427f5deece3a87c6768bdc'
ORDER = ['C','A','B','B','A','C']
CONDITIONS = dict(C='forced_saved', A='alternative1', B='alternative2')
FIELDS = ['images','annotation_denominator','generation','norm','checkpoint','checkpoint_files',
          'checkpoint_acceptance','weight_identity','base_model','base_config_sha256',
          'tokenizer_sha256','coordinate_ids','vocab_size','runtime','input_path','label_path',
          'bindings','raw_paths','input_reports','sites']
BOUNDS = dict(gpus=1, ranks=1, concurrent_sequences=1, score_requests=0,
              continuation_requests=24, requests=24, generated_tokens=59790,
              context=4446, kv_cache_bytes=2*1024**3, whole_wall_seconds=1200,
              cleanup_seconds=30, HF_forwards=0, optimizer_steps=0)


def source_paths():
    return b.source_paths()+['probes/matched_coordinate_branches.py',
                             'tests/probes/test_matched_coordinate_branches.py']


def request_layout(c):
    """The actual caller and consumer share one frozen request layout."""
    result = []
    assert [s['site_id'] for s in c['sites']] == [name for name,_ in FIXED]
    for site, (name, bins) in zip(c['sites'], FIXED, strict=True):
        assert site['eligible'] and site['slot'] == 0
        record = b.load(c['raw_paths'][str(site['image_id'])])
        seen = dict.fromkeys(CONDITIONS, 0)
        for condition in ORDER:
            seen[condition] += 1
            token = c['coordinate_ids'][bins[list(CONDITIONS).index(condition)]]
            prefix,budget = b.branch_prefix(record,site,token)
            result.append(dict(request_index=len(result), site_id=name, condition=condition, repeat=seen[condition],
                condition_name=CONDITIONS[condition], supplied_token_id=token,
                extension=prefix, budget=budget, prefix_tokens=len(prefix),
                processed_prompt_token_ids=record['prompt_token_ids']+prefix,
                input_prefix_identity=b.online.identity(record['prompt_token_ids']+prefix)))
    assert len(result) == 24 and sum(r['budget'] for r in result) == 59790
    return result


def historical_bindings(c):
    """Accepted historical data never passes a current-checkout source gate."""
    assert c['predecessor']['contract_sha256'] == RELEASE_SHA
    old = b.load(c['predecessor']['contract'])
    assert b.sha(Path(c['predecessor']['contract'])) == c['predecessor']['contract_sha256']
    assert old['schema'] == 'greedy-prefix-branching-v1' and old['native_released']
    for key in FIELDS:
        assert c[key] == old[key], key
    paths = dict(old['bindings'])
    paths[c['predecessor']['contract']] = c['predecessor']['contract_sha256']
    root = Path(c['predecessor']['contract']).parent
    release = b.load(root/'lead-release-01.json')
    assert release['contract'] == c['predecessor']['contract'] and release['sha256'] == paths[release['contract']]
    terminal = b.load(root/'native-terminal-candidate-01.json')
    assert terminal['native_exit'] == terminal['readback_exit'] == 0
    assert terminal['cleanup']['engine_close_preceded_terminal_publish']
    assert all(terminal['cleanup']['owned_processes_absent'].values())
    paths.update(terminal['artifacts'])
    for name in ['lead-release-01.json','native-terminal-candidate-01.json']:
        paths[str(root/name)] = c['evidence_bindings'][str(root/name)]
    complete = b.load(root/'native-01/complete.json')
    readback = b.load(root/'native-01/readback.json')
    assert complete['sites'] == readback['sites'] and complete['contract_sha256'] == paths[release['contract']]
    assert [l['site_id'] for l in complete['sites']] == [name for name,_ in FIXED]
    for site, ledger, (_,bins) in zip(c['sites'],complete['sites'],FIXED,strict=True):
        score_path = root/'native-01'/site['site_id']/'scores.json'
        paths[str(score_path)] = ledger['artifacts']['scores.json']
        scoring = b.load(score_path)
        assert scoring['site'] == site
        assert site['emitted_bin'] == bins[0]
        assert [(x['bin'],x['token_id']) for x in scoring['candidates']] == [(v,c['coordinate_ids'][v]) for v in bins[1:]]
    acceptance = c['predecessor']['acceptance']
    if acceptance is not None:
        assert acceptance['sha256'] == ACCEPTANCE_SHA, 'accepted historical receipt identity drift'
        paths[acceptance['path']] = acceptance['sha256']
        accepted = b.load(acceptance['path'])
        assert accepted['status'] == 'lead-accepted', 'finalized predecessor acceptance required'
        assert accepted['worker_candidate']['sha256'] == paths[str(root/'native-terminal-candidate-01.json')]
        assert accepted['worker_candidate']['path'] == str(root/'native-terminal-candidate-01.json')
        assert accepted['execution_source_commit'] == old['source']['commit']
        for key in ['terminal','readback','released_contract']:
            assert paths[accepted[key]['path']] == accepted[key]['sha256']
    for path,expected in paths.items():
        assert b.sha(Path(path)) == expected, path
    assert paths == c['evidence_bindings'], 'historical evidence bindings drift'
    return old


def validate_contract(path, require_release=False, current_source=True):
    c = b.load(path)
    assert c['schema'] == 'matched-coordinate-branches-v1'
    assert c['order'] == ORDER and c['conditions'] == CONDITIONS and c['bounds'] == BOUNDS
    assert type(c['native_released']) is bool
    if require_release: assert c['native_released'], 'lead release required'
    if require_release:
        assert c['predecessor']['acceptance'] is not None, 'finalized predecessor acceptance required'
        assert Path(c['execution_checkout']).resolve() == ROOT.resolve()
        assert Path(c['output_root']).resolve() == OUT.resolve()
    if current_source:
        verify_source_identity(c['source'], required_paths=source_paths(), root=ROOT)
    historical_bindings(c)
    checkpoint = Path(c['checkpoint'])
    assert b.load(checkpoint/'identity.json') == c['checkpoint_files']
    for name,expected in c['checkpoint_files'].items():
        assert b.sha(checkpoint/name) == expected, name
    for package,expected in c['runtime'].items():
        assert version(package) == expected, package
    labels = b.load(c['label_path'])
    assert len(labels) == 18 and sum(len(i['objects']) for i in labels) == 570
    assert c['requests'] == request_layout(c), 'frozen request order/token/budget drift'
    return c


def prepare(output, acceptance=None):
    old_path = PREDECESSOR/'released-contract-01.json'
    old = b.load(old_path)
    c = {k:old[k] for k in FIELDS}
    c.update(schema='matched-coordinate-branches-v1', source=None, native_released=False,
        output_root=str(OUT), execution_checkout=None, bounds=BOUNDS, order=ORDER,
        conditions=CONDITIONS, predecessor=dict(contract=str(old_path), contract_sha256=b.sha(old_path),
            acceptance=dict(path=str(acceptance.resolve()),sha256=b.sha(acceptance)) if acceptance else None))
    paths = dict(old['bindings']);paths[str(old_path)] = c['predecessor']['contract_sha256']
    terminal = b.load(PREDECESSOR/'native-terminal-candidate-01.json');paths.update(terminal['artifacts'])
    for name in ['lead-release-01.json','native-terminal-candidate-01.json']:
        paths[str(PREDECESSOR/name)] = b.sha(PREDECESSOR/name)
    complete = b.load(PREDECESSOR/'native-01/complete.json')
    for ledger in complete['sites']:
        paths[str(PREDECESSOR/'native-01'/ledger['site_id']/'scores.json')] = ledger['artifacts']['scores.json']
    if acceptance: paths[str(acceptance.resolve())] = c['predecessor']['acceptance']['sha256']
    c['evidence_bindings'] = paths
    c['requests'] = request_layout(c)
    output.mkdir(parents=True,exist_ok=False)
    b.write(output/'contract.json',c)
    validate_contract(output/'contract.json',current_source=False)
    q,inputs,images,requests = frontend(c)
    b.write(output/'cpu-layout.json',dict(status='candidate' if acceptance else 'HOLD_predecessor_acceptance',
        native_launched=False, requests=c['requests'], bounds=BOUNDS,
        media_assembly='exact', predecessor_source=old['source']['commit']))
    return c


def qualify(path, destination):
    c = validate_contract(path,current_source=False)
    assert c['source'] is None and c['native_released'] is False
    assert c['predecessor']['acceptance'] is not None, 'finalized predecessor acceptance required'
    c.update(source=capture_source_identity(source_paths(),root=ROOT),
             execution_checkout=str(ROOT), output_root=str(OUT))
    b.write(destination,c)


def frontend(c):
    q = b.rows.frontend()
    assert q.model is None and str(q.base_model_path) == c['base_model']
    assert q.base_config_sha256 == c['base_config_sha256'] and q.tokenizer_sha256 == c['tokenizer_sha256']
    assert q.config.text_config.vocab_size == c['vocab_size']
    assert [q.tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)] == c['coordinate_ids']
    inputs = {r['image_id']:r for r in b.load(c['input_path'])}
    images = {r['image_id']:r for r in b.load(c['label_path'])}
    reports = {r['image_id']:r for r in c['input_reports']}
    requests = {}
    for i in c['images']:
        record = b.load(c['raw_paths'][str(i)])
        assert record['raw_identity'] == b.online.seal(record,record['producer'])['raw_identity']
        for key,value in inputs[i].items(): assert record[key] == value, key
        batch = b.online.native_batch(q,inputs[i])
        assert list(batch.prompt_token_ids[0]) == record['prompt_token_ids']
        assert list(batch.image_grids[0]) == record['image_grid_thw'] and batch.media_sha256[0] == record['media_sha256']
        request = b.online.vllm_requests(q,inputs,[i])[0]
        assert q.tokenizer.encode(request.chat_text,add_special_tokens=False) == reports[i]['unexpanded_chat_token_ids']
        requests[i] = request
    return q,inputs,images,requests


def measure(image, raw, site, tokenizer, supplied):
    result = b.evaluate_branch(image,raw,site,tokenizer,supplied)
    # Keep the independently matched later ledger; re-emissions are not new unique recovery.
    result['later_new_unique'] = {mode:sorted(set(result['later_free'][mode])-
        set(result['inherited']['raw'])-set(result['current_row']['raw'])) for mode in ('raw','category')}
    result['primary_semantic_vector'] = dict(whole_category_owner_ids=result['whole']['category'],
        current_complete=result['first_row']['complete'], current_strict_valid=result['first_row']['strict_valid'],
        current_category_owner_ids=result['current_row']['category'],
        later_free_category_owner_ids=result['later_free']['category'], eos=raw['stop_reason']=='im_end',
        cap=raw['stop_reason']=='length')
    return result


def summarize(artifacts):
    groups = {k:[a for a in artifacts if a['request']['condition']==k] for k in CONDITIONS}
    stability = {}
    for name,pair in groups.items():
        assert len(pair) == 2
        stability[name] = dict(request_identity=pair[0]['request']['input_prefix_identity']==pair[1]['request']['input_prefix_identity'],
            semantic=pair[0]['measurements']['primary_semantic_vector']==pair[1]['measurements']['primary_semantic_vector'],
            exact_tokens_and_stop=pair[0]['raw']['token_ids']==pair[1]['raw']['token_ids'] and pair[0]['raw']['stop_reason']==pair[1]['raw']['stop_reason'])
        assert stability[name]['request_identity']
    comparisons = {}
    for name in ['A','B']:
        stable = stability['C']['semantic'] and stability[name]['semantic']
        contrasts = [dict(whole=b.transitions(c['measurements']['whole'],a['measurements']['whole']),
            later_free=b.transitions(c['measurements']['later_free'],a['measurements']['later_free']),
            later_new_unique=b.transitions(c['measurements']['later_new_unique'],a['measurements']['later_new_unique']),
            assisted_row=b.transitions(c['measurements']['current_row'],a['measurements']['current_row']))
            for c in groups['C'] for a in groups[name]]
        ranges = {part:{mode:{key:[min(len(x[part][mode][key]) for x in contrasts),
            max(len(x[part][mode][key]) for x in contrasts)] for key in ['gained','lost','retained']}
            for mode in ['raw','category']} for part in contrasts[0]}
        comparisons['C-'+name] = dict(semantic_stable=stable,
            point={part:{mode:contrasts[0][part][mode] if all(x[part][mode]==contrasts[0][part][mode] for x in contrasts) else None
                        for mode in ['raw','category']} for part in contrasts[0]} if stable else None, observed_count_ranges=ranges,
            scope='selected supplied-prefix conditional contrast; subset matches nonadditive')
    burden_ranges = {name:{key:[min(a['measurements']['burdens'][key] for a in pair),
                               max(a['measurements']['burdens'][key] for a in pair)]
                          for key in pair[0]['measurements']['burdens']} for name,pair in groups.items()}
    return dict(stability=stability, comparisons=comparisons, burden_ranges=burden_ranges)


def artifact(c, request, acquired, q, inputs, images, expected_sha):
    site = next(s for s in c['sites'] if s['site_id']==request['site_id'])
    assert acquired['request_id'] == inputs[site['image_id']]['request_id']
    assert acquired['processed_prompt_token_ids'] == request['processed_prompt_token_ids']
    assert acquired['full_scores'] is None, 'no new scores'
    assert all(type(t) is int and 0 <= t < c['vocab_size'] for t in acquired['token_ids'])
    checked,reason = trim_suffix(acquired['token_ids'],budget=request['budget'],
        eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),pad_token_id=q.tokenizer.pad_token_id)
    assert list(checked) == acquired['token_ids'] and reason == acquired['stop_reason']
    full = request['extension']+acquired['token_ids']
    assert len(full) <= b.CAP
    producer = dict(kind='matched_supplied_coordinate',source=c['source']['commit'],
        contract_sha256=expected_sha, site_id=site['site_id'], condition=request['condition'],
        repeat=request['repeat'], norm='off',weight_identity=c['weight_identity'])
    raw = b.online.seal(dict(inputs[site['image_id']],token_ids=full,
        text=q.tokenizer.decode(full,skip_special_tokens=False),generated_tokens=len(full),stop_reason=reason),producer)
    measurements = measure(images[site['image_id']],raw,site,q.tokenizer,request['supplied_token_id'])
    return dict(request=request, acquisition=acquired, raw=raw, measurements=measurements,
        free_tokens=len(acquired['token_ids']), historical_raw_identity=site['raw_identity'])


def run(path, output, expected_sha):
    assert b.sha(path) == expected_sha, 'lead-released contract identity drift'
    c = validate_contract(path,require_release=True)
    assert output.resolve().parent == Path(c['output_root']).resolve(), 'execution output owner drift'
    from src.qwen.vllm_rollout import VllmDoraRollout
    q,inputs,images,requests = frontend(c)
    chats = {r['image_id']:r['unexpanded_chat_token_ids'] for r in c['input_reports']}
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic();ledgers = [];counters = dict(requests=0,continuation_requests=0,score_requests=0,generated_tokens=0)
    with VllmDoraRollout(base_model=c['base_model'],checkpoint=c['checkpoint'],identity=c['weight_identity'],
            log_path=output/'vllm.log',device=0,trainer_rank=0,max_model_len=BOUNDS['context'],
            max_num_seqs=1,max_logprobs=-1,kv_cache_memory_bytes=BOUNDS['kv_cache_bytes'],
            seed=92711,timeout=BOUNDS['whole_wall_seconds']) as engine:
        engine.configure_coordinate_output_norm('off',c['coordinate_ids'],identity=c['weight_identity'])
        for site in c['sites']:
            directory = output/site['site_id'];directory.mkdir();artifacts = [];hashes = {}
            for request in [r for r in c['requests'] if r['site_id']==site['site_id']]:
                assert time.monotonic()-started < BOUNDS['whole_wall_seconds']
                acquired = engine.generate_exact([requests[site['image_id']]],
                    chat_token_ids=[chats[site['image_id']]],extensions=[request['extension']],budgets=[request['budget']],
                    eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),pad_token_id=q.tokenizer.pad_token_id,
                    identity=c['weight_identity'],vocab_size=c['vocab_size'],full_scores=False)[0]
                value = artifact(c,request,acquired,q,inputs,images,expected_sha)
                filename = f"{request['condition']}-{request['repeat']}.json"
                b.write(directory/filename,value);hashes[filename] = b.sha(directory/filename);artifacts.append(value)
                counters['requests'] += 1;counters['continuation_requests'] += 1
                counters['generated_tokens'] += value['free_tokens']
                assert all(counters[k] <= BOUNDS[k] for k in counters)
            ledger = dict(site_id=site['site_id'], artifacts=hashes, **summarize(artifacts))
            b.write(directory/'complete.json',ledger);ledgers.append(ledger)
        startup = engine.startup;operations = list(engine.receipts)
    b.write(output/'complete.json',dict(schema='matched-coordinate-terminal-v1',status='candidate',
        contract_sha256=expected_sha,sites=ledgers,counters=counters,startup=startup,operations=operations,
        artifacts={f"{l['site_id']}/complete.json":b.sha(output/l['site_id']/'complete.json') for l in ledgers},
        seconds=time.monotonic()-started,HF_forwards=0,optimizer_steps=0,next_unit_scheduled=False))


def readback(path, output, expected_sha):
    assert b.sha(path) == expected_sha, 'lead-released contract identity drift'
    c = validate_contract(path,require_release=True)
    assert output.resolve().parent == Path(c['output_root']).resolve(), 'execution output owner drift'
    complete = b.load(output/'complete.json')
    assert complete['schema'] == 'matched-coordinate-terminal-v1' and complete['status'] == 'candidate'
    assert complete['contract_sha256'] == expected_sha
    q,inputs,images,requests = frontend(c)
    counters = dict(requests=0,continuation_requests=0,score_requests=0,generated_tokens=0)
    assert [l['site_id'] for l in complete['sites']] == [s['site_id'] for s in c['sites']]
    assert set(complete['artifacts']) == {f"{s['site_id']}/complete.json" for s in c['sites']}
    for site,ledger in zip(c['sites'],complete['sites'],strict=True):
        directory = output/site['site_id']
        assert b.sha(directory/'complete.json') == complete['artifacts'][site['site_id']+'/complete.json']
        assert b.load(directory/'complete.json') == ledger
        expected = [r for r in c['requests'] if r['site_id']==site['site_id']]
        assert set(ledger['artifacts']) == {f"{r['condition']}-{r['repeat']}.json" for r in expected}
        artifacts = []
        for request in expected:
            name = f"{request['condition']}-{request['repeat']}.json"
            assert b.sha(directory/name) == ledger['artifacts'][name]
            value = b.load(directory/name)
            assert value == artifact(c,request,value['acquisition'],q,inputs,images,expected_sha), 'false request/raw/semantic credit'
            artifacts.append(value);counters['requests'] += 1;counters['continuation_requests'] += 1
            counters['generated_tokens'] += value['free_tokens']
        assert ledger == dict(site_id=site['site_id'],artifacts=ledger['artifacts'],**summarize(artifacts)), 'false stability/contrast'
    assert counters == complete['counters'] and counters['requests'] == 24
    assert all(counters[k] <= BOUNDS[k] for k in counters)
    assert complete['HF_forwards'] == complete['optimizer_steps'] == 0 and complete['next_unit_scheduled'] is False
    from src.qwen.vllm_rollout import validate_device_receipt
    startup = complete['startup'];assert startup['identity'] == c['weight_identity']
    validate_device_receipt(startup['device'],startup['device']['requested'])
    assert startup['device']['requested']['rank'] == startup['device']['requested']['device'] == 0
    assert [op['operation'] for op in complete['operations']] == ['coordinate_output_norm']+['generate_exact']*24
    for op in complete['operations']:
        assert op['identity'] == c['weight_identity'] and op['coordinate_output_norm']['mode'] == 'off'
    for op in complete['operations'][1:]: validate_policy(op['coordinate_output_norm'],'off',c)
    return dict(schema='matched-coordinate-readback-v1',status='candidate',counters=counters,sites=complete['sites'],
        terminal_sha256=b.sha(output/'complete.json'),contract_sha256=expected_sha,
        predecessor=c['predecessor'],annotation_denominator=570,
        scope='Selected supplied-coordinate contrast; no empty-history, population or physical-negative claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','qualify','validate','run','readback'])
    parser.add_argument('--contract',type=Path);parser.add_argument('--contract-sha256')
    parser.add_argument('--acceptance',type=Path);parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'prepare':prepare(args.output,args.acceptance)
    elif args.command == 'qualify':qualify(args.contract,args.output)
    elif args.command == 'validate':validate_contract(args.contract)
    elif args.command == 'run':run(args.contract,args.output,args.contract_sha256)
    else:b.write(args.output/'readback.json',readback(args.contract,args.output,args.contract_sha256))


if __name__ == '__main__':main()
