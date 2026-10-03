"""Exact CPU output-table crossover; four native phases require separate lead release."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import time

from probes import mass_versus_ranking as m
from src.artifacts.git_identity import capture_source_identity, verify_source_identity

r, b = m.r, m.b
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-03/output-delta-crossover-08'
CANONICAL = Path('/data/CoordExp/.worktrees/research-probes') / OUT.relative_to(ROOT)
INHERITED = r.HISTORY / 'mass-versus-ranking-07/released-contract-01.json'
INHERITED_SHA = '67e98ce0c666c15a08087be7282b95397b2d751f77eefc7b783f6d9632b55243'
ACCEPTANCE = INHERITED.parent / 'lead-acceptance-01.json'
ACCEPTANCE_SHA = 'ff045f47565bdce6c92e1c8e08eca207876d0b591a20c2db660491acb1825d3f'
PARENT_WEIGHTS = dict(R=m.GMAX_WEIGHT, M='9fc7b57baee2168412001b97c4ac547f03d276dde1216cc3c406ac6aa7548b08')
ARMS = ['R_R', 'R_M', 'M_R', 'M_M']
PHASES = ['native-'+arm for arm in ARMS]
EDGES = [('d_R','R_R','R_M'), ('d_M','M_R','M_M'),
    ('body_at_R','R_R','M_R'), ('body_at_M','R_M','M_M'), ('original','R_R','M_M')]
BOUNDS = dict(m.BOUNDS, optimizer_steps=0, training_replays=0, HF_diagnostics=0,
    score_requests=40, continuation_requests=72, requests=112, generated_tokens=222088,
    retained_bytes=1024**3)
DELTA = 'special_token_embeddings/special_token_embeddings.safetensors'
META = 'special_token_embeddings/special_token_embeddings.json'
PAYLOADS = ['adapter/adapter_config.json', 'adapter/adapter_model.safetensors',
    'adapter/model_card.json', META, DELTA]
frontend, score_measure, natural_measure = m.frontend, m.score_measure, m.natural_measure
resources, live_phase_group, cleanup_phase_group = m.resources, m.live_phase_group, m.cleanup_phase_group


def source_paths():
    return m.source_paths() + ['probes/output_delta_crossover.py', 'tests/probes/test_output_delta_crossover.py']


def inherited():
    assert b.sha(INHERITED)==INHERITED_SHA and b.sha(ACCEPTANCE)==ACCEPTANCE_SHA, 'round07 identity drift'
    accepted=b.load(ACCEPTANCE)
    assert accepted['status']=='lead-accepted' and accepted['release']==dict(path=str(INHERITED),sha256=INHERITED_SHA)
    previous=b.load(INHERITED)
    verified=m.inherited()
    for key in m.INHERITED_FIELDS: assert previous[key]==verified[key], 'input drift: '+key
    return previous, {arm:accepted['checkpoints'][name] for arm,name in [('R','Gmax1'),('M','Gmass1')]}


def inspect_checkpoint(item, c):
    """Five-payload identity plus the exact runtime's CPU-only delta inspector."""
    import torch
    from safetensors.torch import load_file
    from src.qwen.untied_embeddings import inspect_special_token_embedding_delta_payload
    path=Path(item['checkpoint'])
    assert set(item['checkpoint_files'])==set(PAYLOADS), 'five payloads required'
    assert b.load(path/'identity.json')==item['checkpoint_files'], 'copied checkpoint identity'
    for name,sha in item['checkpoint_files'].items(): assert b.sha(path/name)==sha, 'checkpoint payload drift'
    assert item['weight_identity']==b.online.identity(item['checkpoint_files']), 'weight identity drift'
    inspected=inspect_special_token_embedding_delta_payload(path/'special_token_embeddings',
        expected_base_model_path=c['base_model'], expected_base_config_sha256=c['base_config_sha256'],
        expected_tokenizer_sha256=c['tokenizer_sha256'])
    metadata=b.load(path/META)
    assert metadata['semantics']=='additive_delta' and metadata['tie_word_embeddings'] is False
    assert metadata['tensor_dtype']=='float32' and metadata['tensor_shape']==[1004,2048]
    assert metadata['tensor_key']=='input_embed_delta'
    tensors=load_file(str(path/DELTA),device='cpu')
    assert set(tensors)=={'input_embed_delta','output_embed_delta'}, 'delta key drift'
    for t in tensors.values():
        assert t.dtype==torch.float32 and list(t.shape)==[1004,2048] and torch.isfinite(t).all(), 'delta layout/nonfinite'
    return metadata,tensors,inspected


def verify_parents(c, parents):
    values={}
    for arm,item in parents.items():
        assert item['weight_identity']==PARENT_WEIGHTS[arm], 'wrong accepted parent'
        values[arm]=inspect_checkpoint(item,c)
    assert values['R'][0]==values['M'][0], 'parent mapping/semantics drift'
    return values


def provenance(arm, parents):
    body,donor=arm.split('_')
    return dict(schema='output-delta-crossover-v1',arm=arm,body=parents[body],output_donor=parents[donor],
        operation='replace_entire_output_embed_delta',input_embed_delta_source=body,
        shape=[1004,2048],dtype='float32',training=False,scaling=False,blending=False,row_selection=False)


def assemble(arm, parents, c, destination):
    """Serialize tables directly; never construct a model to export CPU weights."""
    from safetensors.torch import save_file
    destination=destination.resolve()
    assert arm in ['R_M','M_R'], 'only hybrids are republished'
    values=verify_parents(c,parents)
    body,donor=arm.split('_')
    destination.mkdir(parents=True,exist_ok=False)
    for name in PAYLOADS:
        if name==DELTA: continue
        target=destination/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(parents[body]['checkpoint'])/name,target)
    save_file({'input_embed_delta':values[body][1]['input_embed_delta'],
        'output_embed_delta':values[donor][1]['output_embed_delta']},str(destination/DELTA))
    files={name:b.sha(destination/name) for name in PAYLOADS}
    b.write(destination/'identity.json',files)
    b.write(destination/'provenance.json',provenance(arm,parents))
    item=dict(checkpoint=str(destination),checkpoint_files=files,weight_identity=b.online.identity(files),
        arm=arm,provenance_sha256=b.sha(destination/'provenance.json'))
    verify_arm(c,parents,arm,item)
    for path in destination.rglob('*'):
        if path.is_file(): path.chmod(0o444)
    for path in sorted(destination.rglob('*'),reverse=True):
        if path.is_dir(): path.chmod(0o555)
    destination.chmod(0o555)
    return item


def verify_arm(c, parents, arm, item):
    import torch
    assert arm in ARMS, 'unknown arm'
    values=verify_parents(c,parents)
    body,donor=arm.split('_')
    if body==donor:
        assert item==parents[body], 'original parent relabelled'
        return
    assert item['arm']==arm, 'wrong hybrid arm'
    metadata,tensors,_=inspect_checkpoint(item,c)
    path=Path(item['checkpoint'])
    assert b.sha(path/'provenance.json')==item['provenance_sha256'], 'hybrid provenance drift'
    assert b.load(path/'provenance.json')==provenance(arm,parents), 'wrong body/donor provenance'
    assert metadata==values[body][0], 'mapping/metadata drift'
    for name in PAYLOADS:
        if name!=DELTA: assert item['checkpoint_files'][name]==parents[body]['checkpoint_files'][name], 'body bytes changed'
    assert torch.equal(tensors['input_embed_delta'],values[body][1]['input_embed_delta']), 'input delta changed'
    assert torch.equal(tensors['output_embed_delta'],values[donor][1]['output_embed_delta']), 'wrong output donor'
    assert item['weight_identity'] not in PARENT_WEIGHTS.values(), 'copied parent weight identity'


def retained_bytes(c, output=None):
    roots={Path(i['checkpoint']) for a,i in c['weights'].items() if a in ['R_M','M_R']}
    if output is not None: roots.add(output)
    total=sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())
    assert total<=BOUNDS['retained_bytes'], 'retained evidence budget'
    return total


def phase_commands(contract, output):
    return [['python','-m','probes.output_delta_crossover','native','--contract',str(contract),
        '--contract-sha256','EXACT_RELEASE_SHA256','--output',str(output),'--arm',arm] for arm in ARMS]


def prepare(output):
    output=output.resolve()
    assert output.resolve().parent==CANONICAL.resolve(), 'canonical preparation owner required'
    previous,parents=inherited()
    c={k:previous[k] for k in m.INHERITED_FIELDS}
    c.update(schema='output-delta-crossover-v1',source=None,native_released=False,execution_checkout=None,
        output_root=str(OUT),bounds=BOUNDS,arms=ARMS,phases=PHASES,edges=EDGES,
        inherited_release=dict(path=str(INHERITED),sha256=INHERITED_SHA),
        inherited_acceptance=dict(path=str(ACCEPTANCE),sha256=ACCEPTANCE_SHA),parents=parents,
        proposed_commands=phase_commands('EXACT_RELEASE_CONTRACT','SELECTED_EXECUTION_OUTPUT'))
    q,inputs,_,reports=frontend(c)
    assert q.model is None and r.contexts(c,q.tokenizer)==c['contexts']
    labels=b.load(c['label_path'])
    assert len(labels)==18 and sum(len(i['objects']) for i in labels)==570
    assert set(inputs)=={i['image_id'] for i in labels}
    values=verify_parents(c,parents)
    metadata=values['R'][0]
    assert metadata['token_ids']==[q.tokenizer.convert_tokens_to_ids(t) for t in metadata['token_strings']], 'ordered token mapping drift'
    output.mkdir(parents=True,exist_ok=False)
    weights={'R_R':parents['R'],'M_M':parents['M']}
    for arm in ['R_M','M_R']: weights[arm]=assemble(arm,parents,c,output/'hybrids'/arm)
    c['weights']={arm:weights[arm] for arm in ARMS}
    size=retained_bytes(c)
    b.write(output/'manifest.json',c)
    b.write(output/'cpu-layout.json',dict(model_loaded=False,native_launched=False,optimizer_created=False,
        input_reports=reports,arms=ARMS,bounds=BOUNDS,hybrid_bytes=size,
        runtime_inspections={arm:inspect_checkpoint(item,c)[2] for arm,item in c['weights'].items()},
        first_hybrid_runtime_boundary='unresolved_until_released_R_M_phase'))
    return c


def validate(path, require_release=False, current_source=True):
    c=b.load(path)
    previous,parents=inherited()
    assert c['schema']=='output-delta-crossover-v1'
    assert c['bounds']==BOUNDS and c['arms']==ARMS and c['phases']==PHASES
    assert c['edges']==[list(edge) for edge in EDGES], 'factorial direction drift'
    for key in m.INHERITED_FIELDS: assert c[key]==previous[key], 'input drift: '+key
    assert c['parents']==parents and set(c['weights'])==set(ARMS)
    assert c['inherited_release']==dict(path=str(INHERITED),sha256=INHERITED_SHA)
    assert c['inherited_acceptance']==dict(path=str(ACCEPTANCE),sha256=ACCEPTANCE_SHA)
    assert c['proposed_commands']==phase_commands('EXACT_RELEASE_CONTRACT','SELECTED_EXECUTION_OUTPUT')
    for arm,item in c['weights'].items():
        if arm in ['R_M','M_R']:
            assert Path(item['checkpoint']).resolve()==CANONICAL/'preparation-01/hybrids'/arm, 'hybrid owner drift'
            if current_source or require_release:
                assert Path(item['checkpoint']).is_absolute(), 'execution inputs require absolute canonical paths'
        verify_arm(c,parents,arm,item)
    retained_bytes(c)
    for package,expected in c['runtime'].items(): assert version(package)==expected, package
    if current_source: verify_source_identity(c['source'],required_paths=source_paths(),root=ROOT)
    if require_release:
        assert c['native_released'] is True, 'exact lead release required'
        assert Path(c['execution_checkout']).resolve()==ROOT.resolve()
        assert Path(c['output_root']).resolve()==OUT.resolve()
    return c


def qualify(path, destination):
    assert not destination.exists(), 'candidate already published'
    c=validate(path,current_source=False)
    assert c['source'] is None and c['native_released'] is False
    q,_,_,_=frontend(c); assert q.model is None
    for item in c['weights'].values(): item['checkpoint']=str(Path(item['checkpoint']).resolve())
    c.update(source=capture_source_identity(source_paths(),root=ROOT),execution_checkout=str(ROOT),output_root=str(OUT))
    b.write(destination,c)


def entry(path, output, expected_sha, phase):
    assert phase in PHASES and b.sha(path)==expected_sha, 'released manifest/phase drift'
    c=validate(path,require_release=True)
    assert output.resolve().parent==Path(c['output_root']).resolve(), 'output owner drift'
    for earlier in PHASES[:PHASES.index(phase)]:
        receipt=b.load(output/earlier/'complete.json')
        assert receipt['phase']==earlier and receipt['contract_sha256']==expected_sha and receipt['source']==c['source']
    return c


def weight_binding(c, output, arm):
    assert arm in ARMS, 'unknown arm'
    item=c['weights'][arm]
    assert Path(item['checkpoint']).is_absolute(), 'execution inputs require absolute canonical paths'
    verify_arm(c,c['parents'],arm,item)
    return Path(item['checkpoint']),item['weight_identity']

def native(path, output, expected_sha, arm, *, engine_factory=None):
    phase = f'native-{arm}'
    c = entry(path, output, expected_sha, phase)
    q, inputs, requests, reports = frontend(c)
    assert q.tokenizer.convert_tokens_to_ids('<|im_end|>') == c['eos_token_id']
    labels = {i['image_id']:i for i in b.load(c['label_path'])}  # evaluator only
    checkpoint, identity = weight_binding(c, output, arm)
    from src.qwen.vllm_rollout import VllmDoraRollout
    engine_factory = VllmDoraRollout if engine_factory is None else engine_factory
    directory = output/phase
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    artifacts = {}
    counters = dict(score_requests=0, continuation_requests=0, requests=0, generated_tokens=0)
    with engine_factory(base_model=c['base_model'], checkpoint=checkpoint, identity=identity,
            log_path=directory/'vllm.log', device=0, trainer_rank=0, max_model_len=4456,
            max_num_seqs=1, max_logprobs=-1, kv_cache_memory_bytes=2*1024**3, seed=92711, timeout=1800) as engine:
        engine.configure_coordinate_output_norm('off', c['coordinate_ids'], identity=identity)
        for context in c['contexts']:
            assert time.monotonic()-started < 1800, 'phase bound'
            image = context['image_id']
            acquired = engine.generate_exact([requests[image]], chat_token_ids=[next(r['unexpanded_chat_token_ids'] for r in reports if r['image_id']==image)],
                extensions=[context['extension']], budgets=[1], eos_token_id=c['eos_token_id'],
                pad_token_id=q.tokenizer.pad_token_id, identity=identity, vocab_size=c['vocab_size'], full_scores=True)[0]
            assert acquired['request_id'] == inputs[image]['request_id']
            value = dict(context=context, acquisition=acquired, measurement=score_measure(c,context,acquired,identity))
            assert len(acquired['token_ids'])==1
            filename = f'score-{len(artifacts):02d}.json'
            b.write(directory/filename, value); artifacts[filename] = b.sha(directory/filename)
            counters['score_requests'] += 1; counters['requests'] += 1; counters['generated_tokens'] += 1
        for image in c['images']:
            assert time.monotonic()-started < 1800, 'phase bound'
            acquired = engine.generate([requests[image]], budgets=[3084], eos_token_id=c['eos_token_id'],
                pad_token_id=q.tokenizer.pad_token_id, identity=identity)[0]
            evidence = dict(request_id=acquired.request_id, token_ids=list(acquired.token_ids),
                stop_reason=acquired.stop_reason, weight_identity=identity)
            value = dict(acquisition=evidence, **natural_measure(c,inputs[image],evidence,q.tokenizer,labels[image],identity))
            filename = f'natural-{image}.json'
            b.write(directory/filename,value); artifacts[filename] = b.sha(directory/filename)
            counters['continuation_requests'] += 1; counters['requests'] += 1
            counters['generated_tokens'] += len(acquired.token_ids)
            retained_bytes(c,output)
        startup, operations = engine.startup, list(engine.receipts)
        active_seconds = time.monotonic()-started
    assert not engine._process.is_alive(), 'native child survived cleanup'
    assert active_seconds<=1800
    assert counters['score_requests']==10 and counters['continuation_requests']==18 and counters['generated_tokens']<=55522
    b.write(directory/'complete.json',dict(phase=phase, arm=arm, source=c['source'], contract_sha256=expected_sha,
        checkpoint=str(checkpoint), weight_identity=identity, artifacts=artifacts, counters=counters,
        startup=startup, operations=operations, engine_closed_before_publish=True,
        owned_child_absent=True, active_seconds=active_seconds, resources=resources(started),
        artifact_bytes=sum((directory/name).stat().st_size for name in artifacts),
        native_cuda_memory='unmeasured_child_engine', native_internal_startup_capture_forwards=None))
    retained_bytes(c,output)

def consume_native(c, output, arm, expected_sha, q, inputs):
    directory = output/f'native-{arm}'
    receipt = b.load(directory/'complete.json')
    checkpoint, identity = weight_binding(c,output,arm)
    assert receipt['phase']==f'native-{arm}' and receipt['arm']==arm and receipt['contract_sha256']==expected_sha
    assert receipt['checkpoint']==str(checkpoint) and receipt['weight_identity']==identity
    assert receipt['source']==c['source'], 'native source drift'
    assert receipt['engine_closed_before_publish'] is True
    assert receipt['owned_child_absent'] is True
    assert receipt['native_internal_startup_capture_forwards'] is None
    expected = [f'score-{i:02d}.json' for i in range(10)] + [f'natural-{i}.json' for i in c['images']]
    assert set(receipt['artifacts']) == set(expected)
    for name, sha in receipt['artifacts'].items(): assert b.sha(directory/name)==sha
    scores, natural = [], []
    labels = {i['image_id']:i for i in b.load(c['label_path'])}
    for i, context in enumerate(c['contexts']):
        value = b.load(directory/f'score-{i:02d}.json')
        assert value['context']==context and value['acquisition']['request_id']==inputs[context['image_id']]['request_id']
        measured = score_measure(c,context,value['acquisition'],identity)
        assert value['measurement']==measured, 'false conditional credit/score'
        scores.append(measured)
    for image in c['images']:
        value = b.load(directory/f'natural-{image}.json')
        measured = natural_measure(c,inputs[image],value['acquisition'],q.tokenizer,labels[image],identity)
        assert value==dict(acquisition=value['acquisition'],**measured), 'false natural credit'
        natural.append(measured['measurement'])
    counters = dict(score_requests=10,continuation_requests=18,requests=28,
        generated_tokens=10+sum(x['burdens']['generated_tokens'] for x in natural))
    assert receipt['counters']==counters and counters['generated_tokens']<=55522
    assert receipt['active_seconds']<=1800 and receipt['resources']['seconds']<=1830
    from src.qwen.vllm_rollout import validate_device_receipt
    validate_device_receipt(receipt['startup']['device'],receipt['startup']['device']['requested'])
    operations=receipt['operations']
    assert [x['operation'] for x in operations]==['coordinate_output_norm']+['generate_exact']*10+['generate']*18
    assert receipt['startup']['identity']==identity
    for op in operations:
        assert op['identity']==identity
        norm=op['coordinate_output_norm']
        assert norm['mode']=='off' and norm['identity']==identity and norm['coordinate_ids']==c['coordinate_ids']
        if op['operation'] != 'coordinate_output_norm':
            assert norm['calls']>0 and norm['first_call']['scaling_active'] is False and norm['first_call']['non_coordinate_unchanged'] is True
    return scores,natural,counters

def factorial(c, scores, natural):
    """Five directed edges and d_M-d_R; owner IDs remain per-image sets."""
    transitions,conditional={},{}
    for name,before_arm,after_arm in EDGES:
        rows=[]
        for before,after in zip(natural[before_arm],natural[after_arm],strict=True):
            assert before['image_id']==after['image_id']
            owners=b.transitions({mode:before['ids'][mode]['retained'] for mode in ['raw','category']},
                {mode:after['ids'][mode]['retained'] for mode in ['raw','category']})
            rows.append(dict(image_id=before['image_id'],owners=owners,
                burden_deltas={key:after['burdens'][key]-before['burdens'][key] for key in before['burdens']}))
        transitions[name]=dict(before=before_arm,after=after_arm,images=rows,
            owner_counts={mode:{key:sum(len(row['owners'][mode][key]) for row in rows)
                for key in ['gained','lost','retained']} for mode in ['raw','category']},
            scalar_deltas={**{key:sum(row['burden_deltas'][key] for row in rows) for key in rows[0]['burden_deltas']},
                **{mode+'_known_owners':sum(len(z['ids'][mode]['retained']) for z in natural[after_arm])-
                    sum(len(z['ids'][mode]['retained']) for z in natural[before_arm]) for mode in ['raw','category']}})
        groups={}
        for site,*_ in r.SITES:
            for split in ['original','training_neighbor','held_out']:
                indices=[i for i,z in enumerate(c['contexts']) if z['site_id']==site and z['split']==split]
                illegal=[i for i in indices if not scores[before_arm][i]['literal_legal']]
                legal=[i for i in indices if scores[before_arm][i]['literal_legal']]
                groups[site+'/'+split]=dict(baseline_illegal_denominator=len(illegal),
                    repaired=[c['contexts'][i]['context_id'] for i in illegal if scores[after_arm][i]['literal_legal']],
                    baseline_legal_denominator=len(legal),
                    retained=[c['contexts'][i]['context_id'] for i in legal if scores[after_arm][i]['literal_legal']],
                    lost=[c['contexts'][i]['context_id'] for i in legal if not scores[after_arm][i]['literal_legal']],
                    margin_changes=[dict(context_id=c['contexts'][i]['context_id'],
                        delta=scores[after_arm][i]['margin']-scores[before_arm][i]['margin']) for i in indices])
        conditional[name]=dict(before=before_arm,after=after_arm,groups=groups)
    interaction={key:transitions['d_M']['scalar_deltas'][key]-transitions['d_R']['scalar_deltas'][key]
        for key in transitions['d_R']['scalar_deltas']}
    return dict(natural_transitions=transitions,interaction=interaction,conditional=conditional)


def readback(path, output, expected_sha):
    c=entry(path,output,expected_sha,PHASES[-1])
    q,inputs,_,_=frontend(c)
    scores,natural={},{}
    counters={key:0 for key in ['optimizer_steps','training_replays','HF_diagnostics','score_requests',
        'continuation_requests','requests','generated_tokens']}
    for arm in ARMS:
        scores[arm],natural[arm],counts=consume_native(c,output,arm,expected_sha,q,inputs)
        for key,value in counts.items(): counters[key]+=value
    assert all(counters[key]==BOUNDS[key] for key in counters if key!='generated_tokens')
    assert counters['generated_tokens']<=BOUNDS['generated_tokens']
    active_seconds=sum(b.load(output/phase/'complete.json')['active_seconds'] for phase in PHASES)
    assert active_seconds<=7200
    summaries={arm:dict(
        known_owner_ids={mode:[[row['image_id'],owner] for row in rows for owner in sorted(set(row['ids'][mode]['retained']))]
            for mode in ['raw','category']},
        burdens={key:sum(row['burdens'][key] for row in rows) for key in rows[0]['burdens']},
        stop_counts={reason:sum(row['stop_reason']==reason for row in rows) for reason in sorted({row['stop_reason'] for row in rows})})
        for arm,rows in natural.items()}
    result=dict(status='candidate',contract_sha256=expected_sha,source=c['source'],phases=PHASES,
        counters=counters,active_seconds=active_seconds,scores=scores,natural=natural,natural_summary=summaries,
        **factorial(c,scores,natural),weights=c['weights'],
        annotation_unmatched_is_physical_negative=False,conditional_natural_recovery_credit=False,
        native_internal_startup_capture_forwards=None,native_cuda_memory='unmeasured_child_engine',
        scientific_acceptance=False,next_unit_scheduled=False)
    retained_bytes(c,output)
    terminal=output/'complete.json'
    if terminal.exists(): assert b.load(terminal)==result, 're-signed false terminal'
    else: b.write(terminal,result)
    readback_value=dict(status='candidate',terminal_sha256=b.sha(terminal),counters=counters)
    if (output/'readback.json').exists(): assert b.load(output/'readback.json')==readback_value
    else: b.write(output/'readback.json',readback_value)
    return result

def package(path, output, expected_sha):
    c=entry(path,output,expected_sha,PHASES[0])
    output.mkdir(parents=True,exist_ok=False)
    commands=phase_commands(path,output)
    commands=[[(expected_sha if x=='EXACT_RELEASE_SHA256' else x) for x in command] for command in commands]
    outcomes=[]
    for phase,command in zip(PHASES,commands,strict=True):
        with (output/f'{phase}.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=process.wait(timeout=c['bounds']['phase_seconds'])
            except subprocess.TimeoutExpired:
                code=124
            cleanup_deadline=time.monotonic()+c['bounds']['cleanup_seconds']
            drained=True
            if live_phase_group(process.pid):
                code=code or 125  # A successful leader with live descendants is partial.
                drained=cleanup_phase_group(process,cleanup_deadline)
            if not drained: code=code or 125
        outcomes.append(dict(phase=phase,command=command,exit=code,owned_group_drained=drained,
            active_timeout_seconds=c['bounds']['phase_seconds'],cleanup_timeout_seconds=c['bounds']['cleanup_seconds']))
        b.write(output/f'package-status-{len(outcomes)}.json',dict(status='partial' if code else 'unreviewed',phases=outcomes))
        if code: raise RuntimeError(f'{phase} failed ({code}); preserve partial evidence, no replacement')
    readback(path,output,expected_sha)
    retained_bytes(c,output)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','qualify','validate','native','package','readback'])
    parser.add_argument('--contract',type=Path)
    parser.add_argument('--contract-sha256')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.output)
    elif args.command=='qualify': qualify(args.contract,args.output)
    elif args.command=='validate': validate(args.contract)
    elif args.command=='native': native(args.contract,args.output,args.contract_sha256,args.arm)
    else: globals()[args.command](args.contract,args.output,args.contract_sha256)


if __name__=='__main__': main()
