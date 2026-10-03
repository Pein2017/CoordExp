"""Saved DoRA/input factorial with exact anchor output; CPU preparation before release."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import time

from probes import endpoint_block_ablation as previous
from src.artifacts.git_identity import capture_source_identity, verify_source_identity

m, r, b = previous.m, previous.r, previous.b
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-03/dora-input-ablation-10'
CANONICAL = Path('/data/CoordExp/.worktrees/research-probes') / OUT.relative_to(ROOT)
INHERITED = r.HISTORY / 'endpoint-block-ablation-09/released-contract-01.json'
INHERITED_SHA = 'eb75dcb988e3de0833802c0852d179032cd92ff7cd61511155736672a3136948'
ACCEPTANCE = INHERITED.parent / 'lead-acceptance-01.json'
ACCEPTANCE_SHA = '8e97a2dd07d6a632673e0e165c9da6bd52a3cc09e648e8eef5b02c3ee6d61f5b'
PARENT_WEIGHTS = dict(A='8dcd444f01ef806b743fd5d0f25518ac305e251f2f1dfdb2609fcc1993b80eb3',
    B='a5456cec3f59102d83f60b53302266f5a0e05d926e121118f838ef6d54a666d1')
ORIGINAL_WEIGHT = '07ea98e90220a9126042a27b3a76f60e9955320fed4bf77c2a8803ac9d52001a'
ARMS = ['A_A', 'A_I', 'D_A', 'D_I']
PHASES = ['native-'+arm for arm in ARMS]
EDGES = [('d_A','A_A','A_I'), ('d_D','D_A','D_I'),
    ('dora_at_A','A_A','D_A'), ('dora_at_I','A_I','D_I'), ('joint_update','A_A','D_I')]
BOUNDS, DELTA, META, PAYLOADS = previous.BOUNDS, previous.DELTA, previous.META, previous.PAYLOADS
frontend, score_measure, natural_measure = previous.frontend, previous.score_measure, previous.natural_measure
resources, live_phase_group, cleanup_phase_group = previous.resources, previous.live_phase_group, previous.cleanup_phase_group
inspect_checkpoint = previous.inspect_checkpoint


def source_paths():
    return previous.source_paths()+['probes/dora_input_ablation.py','tests/probes/test_dora_input_ablation.py']


def inherited():
    assert b.sha(INHERITED)==INHERITED_SHA and b.sha(ACCEPTANCE)==ACCEPTANCE_SHA, 'round09 identity drift'
    accepted=b.load(ACCEPTANCE)
    assert accepted['status']=='lead-accepted' and accepted['release']==dict(path=str(INHERITED),sha256=INHERITED_SHA)
    c=b.load(INHERITED)
    verified,_,evidence=previous.inherited()
    for key in m.INHERITED_FIELDS: assert c[key]==verified[key], 'input drift: '+key
    assert c['anchor_export_evidence']==evidence, 'accepted export evidence drift'
    parents={key:accepted['weights'][arm] for key,arm in [('A','A_A'),('B','M_A')]}
    assert parents['A']==c['weights']['A_A'] and parents['B']==c['weights']['M_A']
    # B keeps the historical M_A label and provenance; current D_I is a separate role.
    assert parents['B']['arm']=='M_A', 'historical label confusion'
    return c,parents,evidence


def verify_parents(c, parents):
    import torch
    from safetensors import safe_open
    assert set(parents)=={'A','B'}, 'wrong parent set'
    assert b.sha(ACCEPTANCE)==ACCEPTANCE_SHA, 'round09 acceptance drift'
    accepted=b.load(ACCEPTANCE)['weights']
    values={}
    for key,arm in [('A','A_A'),('B','M_A')]:
        item=parents[key]
        assert item==accepted[arm] and item['weight_identity']==PARENT_WEIGHTS[key], 'wrong accepted parent'
        assert Path(item['checkpoint']).is_absolute(), 'absolute canonical paths required'
        values[key]=inspect_checkpoint(item,c)
    path=Path(parents['B']['checkpoint'])
    assert b.sha(path/'provenance.json')==parents['B']['provenance_sha256'], 'historical provenance drift'
    assert b.load(path/'provenance.json')['arm']==parents['B']['arm']=='M_A', 'historical label confusion'
    assert values['A'][0]==values['B'][0], 'parent mapping/semantics drift'
    assert torch.equal(values['A'][1]['output_embed_delta'],values['B'][1]['output_embed_delta']), 'anchor output changed'
    with safe_open(str(Path(parents['A']['checkpoint'])/DELTA),framework='pt',device='cpu') as a, \
            safe_open(str(path/DELTA),framework='pt',device='cpu') as other:
        assert a.metadata()==other.metadata(), 'parent tensor metadata drift'
    return values


def donors(arm):
    assert arm in ARMS, 'unknown arm'
    return ('A' if arm[0]=='A' else 'B'), ('A' if arm[-1]=='A' else 'B')


def provenance(arm, parents):
    dora,input_donor=donors(arm)
    return dict(schema='dora-input-ablation-v1',arm=arm,dora_donor=parents[dora],
        input_donor=parents[input_donor],output_donor=parents['A'],
        operation='copy_delta_file_after_exact_common_anchor_output_proof',
        dora_source=dora,input_embed_delta_source=input_donor,output_embed_delta_source='A',
        shape=[1004,2048],dtype='float32',training=False,scaling=False,blending=False,row_selection=False)


def assemble(arm, parents, c, destination):
    assert arm in ['A_I','D_A'], 'only two new hybrids are republished'
    verify_parents(c,parents); dora,input_donor=donors(arm)
    destination=destination.resolve(); destination.mkdir(parents=True,exist_ok=False)
    for name in PAYLOADS:
        donor=input_donor if name==DELTA else dora
        target=destination/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(parents[donor]['checkpoint'])/name,target)
    files={name:b.sha(destination/name) for name in PAYLOADS}
    b.write(destination/'identity.json',files); b.write(destination/'provenance.json',provenance(arm,parents))
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
    assert Path(item['checkpoint']).is_absolute(), 'absolute canonical paths required'
    values=verify_parents(c,parents)
    if arm in ['A_A','D_I']:
        assert item==parents['A' if arm=='A_A' else 'B'], 'original parent relabelled'
        return
    dora,input_donor=donors(arm)
    assert item['arm']==arm, 'wrong hybrid arm'
    metadata,tensors,_=inspect_checkpoint(item,c); path=Path(item['checkpoint'])
    assert b.sha(path/'provenance.json')==item['provenance_sha256']
    assert b.load(path/'provenance.json')==provenance(arm,parents), 'wrong component provenance'
    assert metadata==values['A'][0], 'mapping/metadata drift'
    for name in PAYLOADS:
        donor=input_donor if name==DELTA else dora
        assert item['checkpoint_files'][name]==parents[donor]['checkpoint_files'][name], 'wrong DoRA/input donor bytes'
    assert torch.equal(tensors['input_embed_delta'],values[input_donor][1]['input_embed_delta']), 'wrong input donor'
    assert torch.equal(tensors['output_embed_delta'],values['A'][1]['output_embed_delta']), 'anchor output changed'
    assert item['weight_identity'] not in PARENT_WEIGHTS.values(), 'copied parent identity'


def retained_bytes(c, output=None):
    roots={Path(c['weights'][a]['checkpoint']) for a in ['A_I','D_A']}
    if output is not None: roots.add(output)
    # Count each retained file once even when the evidence root contains the hybrids.
    total=sum(p.stat().st_size for p in {p for root in roots for p in root.rglob('*') if p.is_file()})
    assert total<=BOUNDS['retained_bytes'], 'retained evidence budget'
    return total


def phase_commands(contract, output):
    return [['python','-m','probes.dora_input_ablation','native','--contract',str(contract),
        '--contract-sha256','EXACT_RELEASE_SHA256','--output',str(output),'--arm',arm] for arm in ARMS]


def prepare(output):
    output=output.resolve()
    assert output==CANONICAL/'preparation-01', 'canonical preparation owner required'
    previous,parents,evidence=inherited()
    c={k:previous[k] for k in m.INHERITED_FIELDS}
    c.update(schema='dora-input-ablation-v1',source=None,native_released=False,execution_checkout=None,
        output_root=str(OUT),bounds=BOUNDS,arms=ARMS,phases=PHASES,edges=EDGES,
        inherited_release=dict(path=str(INHERITED),sha256=INHERITED_SHA),
        inherited_acceptance=dict(path=str(ACCEPTANCE),sha256=ACCEPTANCE_SHA),parents=parents,
        anchor_export_evidence=evidence,proposed_commands=phase_commands('EXACT_RELEASE_CONTRACT','SELECTED_EXECUTION_OUTPUT'))
    q,inputs,_,reports=frontend(c)
    assert q.model is None and r.contexts(c,q.tokenizer)==c['contexts']
    labels=b.load(c['label_path'])
    assert len(labels)==18 and sum(len(i['objects']) for i in labels)==570 and set(inputs)=={i['image_id'] for i in labels}
    metadata=verify_parents(c,parents)['A'][0]
    assert metadata['token_ids']==[q.tokenizer.convert_tokens_to_ids(t) for t in metadata['token_strings']], 'ordered token mapping drift'
    output.mkdir(parents=True,exist_ok=False)
    weights={'A_A':parents['A'],'D_I':parents['B']}
    for arm in ['A_I','D_A']: weights[arm]=assemble(arm,parents,c,output/'hybrids'/arm)
    c['weights']={arm:weights[arm] for arm in ARMS}
    b.write(output/'manifest.json',c)
    b.write(output/'cpu-layout.json',dict(model_loaded=False,native_launched=False,optimizer_created=False,
        input_reports=reports,arms=ARMS,bounds=BOUNDS,hybrid_bytes=retained_bytes(c),
        runtime_inspections={arm:inspect_checkpoint(item,c)[2] for arm,item in c['weights'].items()},
        anchor_export_evidence=evidence,pending_real_runtime=['first scheduled A_I new hybrid load']))
    return c


def validate(path, require_release=False, current_source=True):
    c=b.load(path); previous,parents,evidence=inherited()
    assert c['schema']=='dora-input-ablation-v1' and c['bounds']==BOUNDS and c['arms']==ARMS and c['phases']==PHASES
    assert c['edges']==[list(edge) for edge in EDGES], 'factorial direction drift'
    for key in m.INHERITED_FIELDS: assert c[key]==previous[key], 'input drift: '+key
    assert c['parents']==parents and c['anchor_export_evidence']==evidence and set(c['weights'])==set(ARMS)
    assert c['inherited_release']==dict(path=str(INHERITED),sha256=INHERITED_SHA)
    assert c['inherited_acceptance']==dict(path=str(ACCEPTANCE),sha256=ACCEPTANCE_SHA)
    assert c['proposed_commands']==phase_commands('EXACT_RELEASE_CONTRACT','SELECTED_EXECUTION_OUTPUT')
    for arm,item in c['weights'].items():
        if arm in ['A_I','D_A']: assert Path(item['checkpoint'])==CANONICAL/'preparation-01/hybrids'/arm, 'hybrid owner drift'
        verify_arm(c,parents,arm,item)
    retained_bytes(c)
    for package,expected in c['runtime'].items(): assert version(package)==expected, package
    if current_source: verify_source_identity(c['source'],required_paths=source_paths(),root=ROOT)
    if require_release:
        assert c['native_released'] is True, 'exact lead release required'
        assert Path(c['execution_checkout']).resolve()==ROOT.resolve() and Path(c['output_root']).resolve()==OUT.resolve()
    return c


def qualify(path, destination):
    assert not destination.exists(), 'candidate already published'
    c=validate(path,current_source=False)
    assert c['source'] is None and c['native_released'] is False
    q,_,_,_=frontend(c); assert q.model is None
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
    item=c['weights'][arm]; verify_arm(c,c['parents'],arm,item)
    return Path(item['checkpoint']),item['weight_identity']


def native(path, output, expected_sha, arm, *, engine_factory=None):
    c=entry(path,output,expected_sha,'native-'+arm)
    q,inputs,requests,reports=frontend(c)
    assert q.model is None and q.tokenizer.convert_tokens_to_ids('<|im_end|>')==c['eos_token_id']
    checkpoint,identity=weight_binding(c,output,arm)
    labels={i['image_id']:i for i in b.load(c['label_path'])}
    if engine_factory is None:
        from src.qwen.vllm_rollout import VllmDoraRollout
        engine_factory=VllmDoraRollout
    directory=output/('native-'+arm); directory.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); artifacts={}
    counters=dict(score_requests=0,continuation_requests=0,requests=0,generated_tokens=0)
    with engine_factory(base_model=c['base_model'],checkpoint=checkpoint,identity=identity,
            log_path=directory/'vllm.log',device=0,trainer_rank=0,max_model_len=4456,max_num_seqs=1,
            max_logprobs=-1,kv_cache_memory_bytes=2*1024**3,seed=92711,timeout=1800) as engine:
        engine.configure_coordinate_output_norm('off',c['coordinate_ids'],identity=identity)
        for i,context in enumerate(c['contexts']):
            assert time.monotonic()-started<1800, 'phase bound'
            image=context['image_id']
            acquired=engine.generate_exact([requests[image]],chat_token_ids=[next(z['unexpanded_chat_token_ids'] for z in reports if z['image_id']==image)],
                extensions=[context['extension']],budgets=[1],eos_token_id=c['eos_token_id'],
                pad_token_id=q.tokenizer.pad_token_id,identity=identity,vocab_size=c['vocab_size'],full_scores=True)[0]
            assert acquired['request_id']==inputs[image]['request_id']
            name=f'score-{i:02d}.json'
            b.write(directory/name,dict(context=context,acquisition=acquired,measurement=score_measure(c,context,acquired,identity)))
            artifacts[name]=b.sha(directory/name)
            counters['score_requests']+=1; counters['requests']+=1; counters['generated_tokens']+=1
            retained_bytes(c,output)
        for image in c['images']:
            assert time.monotonic()-started<1800, 'phase bound'
            acquired=engine.generate([requests[image]],budgets=[3084],eos_token_id=c['eos_token_id'],
                pad_token_id=q.tokenizer.pad_token_id,identity=identity)[0]
            evidence=dict(request_id=acquired.request_id,token_ids=list(acquired.token_ids),stop_reason=acquired.stop_reason,weight_identity=identity)
            name=f'natural-{image}.json'
            b.write(directory/name,dict(acquisition=evidence,**natural_measure(c,inputs[image],evidence,q.tokenizer,labels[image],identity)))
            artifacts[name]=b.sha(directory/name)
            counters['continuation_requests']+=1; counters['requests']+=1; counters['generated_tokens']+=len(acquired.token_ids)
            retained_bytes(c,output)
        startup,operations=engine.startup,list(engine.receipts); active_seconds=time.monotonic()-started
    assert not engine._process.is_alive(), 'native child survived cleanup'
    assert active_seconds<=1800 and counters['score_requests']==10 and counters['continuation_requests']==18 and counters['generated_tokens']<=55522
    b.write(directory/'complete.json',dict(phase='native-'+arm,arm=arm,source=c['source'],contract_sha256=expected_sha,
        checkpoint=str(checkpoint),weight_identity=identity,artifacts=artifacts,counters=counters,startup=startup,operations=operations,
        engine_closed_before_publish=True,owned_child_absent=True,active_seconds=active_seconds,resources=resources(started),
        artifact_bytes=sum((directory/name).stat().st_size for name in artifacts),native_cuda_memory='unmeasured_child_engine',native_internal_startup_capture_forwards=None))
    retained_bytes(c,output)


def consume_native(c, output, arm, expected_sha, q, inputs):
    directory=output/('native-'+arm); receipt=b.load(directory/'complete.json')
    checkpoint,identity=weight_binding(c,output,arm)
    assert receipt['phase']=='native-'+arm and receipt['arm']==arm and receipt['contract_sha256']==expected_sha
    assert receipt['checkpoint']==str(checkpoint) and receipt['weight_identity']==identity and receipt['source']==c['source'], 'native source/weight drift'
    assert receipt['engine_closed_before_publish'] is receipt['owned_child_absent'] is True
    assert receipt['native_internal_startup_capture_forwards'] is None and receipt['native_cuda_memory']=='unmeasured_child_engine'
    assert set(receipt['artifacts'])=={f'score-{i:02d}.json' for i in range(10)}|{f'natural-{i}.json' for i in c['images']}
    for name,sha in receipt['artifacts'].items(): assert b.sha(directory/name)==sha
    scores,natural=[],[]; labels={i['image_id']:i for i in b.load(c['label_path'])}
    for i,context in enumerate(c['contexts']):
        value=b.load(directory/f'score-{i:02d}.json')
        assert value['context']==context and value['acquisition']['request_id']==inputs[context['image_id']]['request_id']
        measured=score_measure(c,context,value['acquisition'],identity)
        assert value['measurement']==measured, 'false conditional credit/score'
        scores.append(measured)
    for image in c['images']:
        value=b.load(directory/f'natural-{image}.json')
        measured=natural_measure(c,inputs[image],value['acquisition'],q.tokenizer,labels[image],identity)
        assert value==dict(acquisition=value['acquisition'],**measured), 'false natural credit'
        natural.append(measured['measurement'])
    counters=dict(score_requests=10,continuation_requests=18,requests=28,generated_tokens=10+sum(z['burdens']['generated_tokens'] for z in natural))
    assert receipt['counters']==counters and counters['generated_tokens']<=55522
    assert 0<=receipt['active_seconds']<=1800 and 0<=receipt['resources']['seconds']<=1830
    from src.qwen.vllm_rollout import validate_device_receipt
    validate_device_receipt(receipt['startup']['device'],receipt['startup']['device']['requested'])
    assert receipt['startup']['identity']==identity
    assert [z['operation'] for z in receipt['operations']]==['coordinate_output_norm']+['generate_exact']*10+['generate']*18
    for op in receipt['operations']:
        assert op['identity']==identity
        norm=op['coordinate_output_norm']
        assert norm['mode']=='off' and norm['identity']==identity and norm['coordinate_ids']==c['coordinate_ids']
        if op['operation']!='coordinate_output_norm':
            assert norm['calls']>0 and norm['first_call']['scaling_active'] is False and norm['first_call']['non_coordinate_unchanged'] is True
    return scores,natural,counters


def outcome(aa, ai, da, di):
    """Literal outcomes only: sufficiency/necessity are conditional on this endpoint/context."""
    assert all(type(v) is bool for v in [aa,ai,da,di])
    if aa: return 'baseline_legal'
    if not di: return 'antagonism' if ai or da else 'no_repair'
    if ai and da: return 'either_sufficient'
    if ai: return 'input_sufficient_required_with_D_DoRA'
    if da: return 'DoRA_sufficient_required_with_I_input'
    return 'complementary_interaction'


def factorial(c, scores, natural):
    tables=[]
    for i,context in enumerate(c['contexts']):
        observations={arm:scores[arm][i] for arm in ARMS}
        assert all(row['context_id']==context['context_id'] for row in observations.values()), 'context alignment drift'
        legal={arm:row['literal_legal'] for arm,row in observations.items()}
        tables.append(dict(context_id=context['context_id'],site_id=context['site_id'],split=context['split'],
            baseline_illegal=not legal['A_A'],outcome=outcome(*(legal[a] for a in ARMS)),arms=observations,
            repairing_ablations=[a for a in ['A_I','D_A'] if not legal['A_A'] and legal[a]],
            antagonistic_ablations=[a for a in ['A_I','D_A'] if not legal['A_A'] and not legal['D_I'] and legal[a]]))
    illegal=[z for z in tables if z['baseline_illegal']]; legal=[z for z in tables if not z['baseline_illegal']]
    conditional=dict(truth_tables=tables,baseline_illegal_denominator=len(illegal),repair_contrast_present=bool(illegal),
        outcome_counts={kind:sum(z['outcome']==kind for z in illegal) for kind in sorted({z['outcome'] for z in illegal})},
        baseline_legal_denominator=len(legal),baseline_legal_retention={arm:dict(
            retained=[z['context_id'] for z in legal if z['arms'][arm]['literal_legal']],
            lost=[z['context_id'] for z in legal if not z['arms'][arm]['literal_legal']]) for arm in ARMS},
        groups={site+'/'+split:dict(baseline_illegal_denominator=sum(z['baseline_illegal'] for z in tables if z['site_id']==site and z['split']==split),
            contexts=[z['context_id'] for z in tables if z['site_id']==site and z['split']==split]) for site,*_ in r.SITES for split in ['original','training_neighbor','held_out']})
    transitions={}
    for name,before_arm,after_arm in EDGES:
        rows=[]
        for before,after in zip(natural[before_arm],natural[after_arm],strict=True):
            assert before['image_id']==after['image_id']
            owners=b.transitions({mode:before['ids'][mode]['retained'] for mode in ['raw','category']},
                {mode:after['ids'][mode]['retained'] for mode in ['raw','category']})
            rows.append(dict(image_id=before['image_id'],owners=owners,
                burden_deltas={key:after['burdens'][key]-before['burdens'][key] for key in before['burdens']}))
        transitions[name]=dict(before=before_arm,after=after_arm,images=rows,
            owner_counts={mode:{key:sum(len(z['owners'][mode][key]) for z in rows) for key in ['gained','lost','retained']} for mode in ['raw','category']},
            scalar_deltas={**{key:sum(z['burden_deltas'][key] for z in rows) for key in rows[0]['burden_deltas']},
                **{mode+'_known_owners':sum(len(z['ids'][mode]['retained']) for z in natural[after_arm])-sum(len(z['ids'][mode]['retained']) for z in natural[before_arm]) for mode in ['raw','category']}})
    return dict(conditional=conditional,natural_transitions=transitions,
        interaction={key:transitions['d_D']['scalar_deltas'][key]-transitions['d_A']['scalar_deltas'][key] for key in transitions['d_A']['scalar_deltas']})


def readback(path, output, expected_sha):
    c=entry(path,output,expected_sha,PHASES[-1]); q,inputs,_,_=frontend(c)
    scores,natural={},{}; counters={key:0 for key in ['optimizer_steps','training_replays','HF_diagnostics','score_requests','continuation_requests','requests','generated_tokens']}
    for arm in ARMS:
        scores[arm],natural[arm],counts=consume_native(c,output,arm,expected_sha,q,inputs)
        for key,value in counts.items(): counters[key]+=value
    assert all(counters[key]==BOUNDS[key] for key in counters if key!='generated_tokens') and counters['generated_tokens']<=BOUNDS['generated_tokens']
    active_seconds=sum(b.load(output/phase/'complete.json')['active_seconds'] for phase in PHASES)
    assert active_seconds<=7200
    summaries={arm:dict(known_owner_ids={mode:[[z['image_id'],owner] for z in rows for owner in sorted(set(z['ids'][mode]['retained']))] for mode in ['raw','category']},
        burdens={key:sum(z['burdens'][key] for z in rows) for key in rows[0]['burdens']},
        stop_counts={reason:sum(z['stop_reason']==reason for z in rows) for reason in sorted({z['stop_reason'] for z in rows})}) for arm,rows in natural.items()}
    result=dict(status='candidate',contract_sha256=expected_sha,source=c['source'],phases=PHASES,counters=counters,
        active_seconds=active_seconds,scores=scores,natural=natural,natural_summary=summaries,**factorial(c,scores,natural),weights=c['weights'],
        annotation_unmatched_is_physical_negative=False,conditional_natural_recovery_credit=False,
        native_internal_startup_capture_forwards=None,native_cuda_memory='unmeasured_child_engine',scientific_acceptance=False,next_unit_scheduled=False)
    retained_bytes(c,output); terminal=output/'complete.json'
    if terminal.exists(): assert b.load(terminal)==result, 're-signed false terminal'
    else: b.write(terminal,result)
    value=dict(status='candidate',terminal_sha256=b.sha(terminal),counters=counters)
    if (output/'readback.json').exists(): assert b.load(output/'readback.json')==value
    else: b.write(output/'readback.json',value)
    retained_bytes(c,output)
    return result


def package(path, output, expected_sha):
    c=entry(path,output,expected_sha,PHASES[0]); output.mkdir(parents=True,exist_ok=False)
    commands=[[expected_sha if z=='EXACT_RELEASE_SHA256' else z for z in command] for command in phase_commands(path,output)]
    outcomes=[]
    for phase,command in zip(PHASES,commands,strict=True):
        with (output/(phase+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=process.wait(timeout=c['bounds']['phase_seconds'])
            except subprocess.TimeoutExpired: code=124
            drained=True
            if live_phase_group(process.pid):
                code=code or 125
                drained=cleanup_phase_group(process,time.monotonic()+c['bounds']['cleanup_seconds'])
            if not drained: code=code or 125
        outcomes.append(dict(phase=phase,command=command,exit=code,owned_group_drained=drained,
            active_timeout_seconds=c['bounds']['phase_seconds'],cleanup_timeout_seconds=c['bounds']['cleanup_seconds']))
        b.write(output/f'package-status-{len(outcomes)}.json',dict(status='partial' if code else 'unreviewed',phases=outcomes))
        if code: raise RuntimeError(f'{phase} failed ({code}); preserve partial evidence, no replacement')
    readback(path,output,expected_sha)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','qualify','validate','native','package','readback'])
    parser.add_argument('--contract',type=Path); parser.add_argument('--contract-sha256')
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.output)
    elif args.command=='qualify': qualify(args.contract,args.output)
    elif args.command=='validate': validate(args.contract)
    elif args.command=='native': native(args.contract,args.output,args.contract_sha256,args.arm)
    else: globals()[args.command](args.contract,args.output,args.contract_sha256)


if __name__=='__main__': main()
