"""One frozen four-update ranking trajectory; model execution needs exact lead release."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import math
from pathlib import Path
import subprocess
import time

from probes import prefix_exposure_ranking as r
from probes import iterative_positive as p
from src.artifacts.git_identity import capture_source_identity, verify_source_identity

b = r.b
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-03/short-dose-ranking-06'
INHERITED = r.HISTORY / 'prefix-exposure-ranking-04/released-contract-01.json'
INHERITED_SHA = '0fe3deac8f3eff7ec542686bdda8c7673aeee91b993bd3ca1628ac2233bf441b'
ACCEPTANCE = INHERITED.parent / 'lead-acceptance-01.json'
ACCEPTANCE_SHA = 'c93f0863496fde0f14404ca4af8daa2733f7c88f37bc405553131b4b21cba931'
ENDPOINTS = ['step1', 'step4']
PHASES = ['train', 'native-anchor', 'native-step1', 'native-step4']
BOUNDS = dict(r.BOUNDS, optimizer_steps=4, training_replays=24, active_seconds=7200)
INHERITED_FIELDS = r.FIELDS + ['contexts', 'images', 'input_reports', 'eos_token_id',
    'offsets', 'annotation_denominator', 'objective', 'gt_role', 'acquisition_role']
frontend = r.frontend
replay = r.replay
hf_measure = r.hf_measure
live_training_binding = r.live_training_binding
score_measure = r.score_measure
natural_measure = r.natural_measure
resources = r.resources
live_phase_group = r.live_phase_group
cleanup_phase_group = r.cleanup_phase_group


def source_paths():
    return r.source_paths() + ['probes/short_dose_ranking.py', 'tests/probes/test_short_dose_ranking.py']


def inherited():
    assert b.sha(INHERITED) == INHERITED_SHA, 'round04 release drift'
    assert b.sha(ACCEPTANCE) == ACCEPTANCE_SHA, 'round04 acceptance drift'
    accepted = b.load(ACCEPTANCE)
    assert accepted['status'] == 'lead-accepted' and accepted['release'] == dict(path=str(INHERITED), sha256=INHERITED_SHA)
    c = b.load(INHERITED)
    assert c['weight_identity'] == r.WEIGHT and c['norm'] == 'off'
    # Reuse accepted historical payload/cleanup evidence; verify active inputs, not old16 outputs.
    for path in [c['input_path'], c['label_path'], *c['raw_paths'].values()]:
        assert b.sha(Path(path)) == c['evidence_bindings'][path], 'inherited input drift'
    return c


def training_binding(c):
    result = r.training_binding(c)
    result['optimizer'] = {k:v for k,v in result['optimizer'].items() if k != 'fresh_per_arm'}
    result['optimizer']['fresh_once'] = True
    result.pop('updates_per_arm')
    result.update(updates=4, checkpoints=[1,4], same_live_optimizer=True)
    return result


def phase_commands(contract, output):
    return [['python', '-m', 'probes.short_dose_ranking', 'train' if phase == 'train' else 'native',
        '--contract', str(contract), '--contract-sha256', 'EXACT_RELEASE_SHA256', '--output', str(output)]
        + ([] if phase == 'train' else ['--endpoint', phase.removeprefix('native-')]) for phase in PHASES]


def prepare(output):
    previous = inherited()
    c = {k:previous[k] for k in INHERITED_FIELDS}
    c.update(schema='short-dose-ranking-v1', source=None, native_released=False,
        execution_checkout=None, output_root=str(OUT), bounds=BOUNDS, phases=PHASES,
        inherited_release=dict(path=str(INHERITED), sha256=INHERITED_SHA),
        inherited_acceptance=dict(path=str(ACCEPTANCE), sha256=ACCEPTANCE_SHA),
        training=training_binding(previous), proposed_commands=phase_commands('EXACT_RELEASE_CONTRACT','SELECTED_EXECUTION_OUTPUT'))
    q, inputs, _, reports = frontend(c)
    assert q.model is None and r.contexts(c,q.tokenizer) == c['contexts']
    labels = b.load(c['label_path'])
    assert len(labels)==18 and sum(len(i['objects']) for i in labels)==570
    assert set(inputs)=={i['image_id'] for i in labels}
    output.mkdir(parents=True, exist_ok=False)
    b.write(output/'manifest.json', c)
    b.write(output/'cpu-layout.json', dict(model_loaded=False, native_launched=False,
        input_reports=reports, context_identities=[dict(context_id=x['context_id'],
        input_prefix_identity=x['input_prefix_identity'], support_identity=b.online.identity(x['support'])) for x in c['contexts']],
        schedule=[x['context_id'] for x in r.views(c,'R-single')], bounds=BOUNDS))
    return c


def validate(path, require_release=False, current_source=True):
    c = b.load(path)
    assert c['schema']=='short-dose-ranking-v1' and c['bounds']==BOUNDS and c['phases']==PHASES
    previous = inherited()
    for key in INHERITED_FIELDS: assert c[key]==previous[key], key
    assert c['inherited_release']==dict(path=str(INHERITED),sha256=INHERITED_SHA)
    assert c['inherited_acceptance']==dict(path=str(ACCEPTANCE),sha256=ACCEPTANCE_SHA)
    assert c['training']==training_binding(previous), 'training binding drift'
    assert c['proposed_commands']==phase_commands('EXACT_RELEASE_CONTRACT','SELECTED_EXECUTION_OUTPUT')
    for package, expected in c['runtime'].items(): assert version(package)==expected, package
    if current_source: verify_source_identity(c['source'],required_paths=source_paths(),root=ROOT)
    if require_release:
        assert c['native_released'] is True, 'exact lead release required'
        assert Path(c['execution_checkout']).resolve()==ROOT.resolve()
        assert Path(c['output_root']).resolve()==OUT.resolve()
    return c


def qualify(path, destination):
    assert not destination.exists(), 'candidate publication already exists'
    c = validate(path,current_source=False)
    frontend(c)
    assert c['source'] is None and c['native_released'] is False
    c.update(source=capture_source_identity(source_paths(),root=ROOT), execution_checkout=str(ROOT), output_root=str(OUT))
    b.write(destination,c)


def entry(path, output, expected_sha, phase):
    assert b.sha(path)==expected_sha, 'released manifest drift'
    c = validate(path,require_release=True)
    assert output.resolve().parent==Path(c['output_root']).resolve(), 'output owner drift'
    for earlier in PHASES[:PHASES.index(phase)]:
        receipt=b.load(output/earlier/'complete.json')
        assert receipt['phase']==earlier and receipt['contract_sha256']==expected_sha and receipt['source']==c['source']
    return c


def checkpoint_binding(c, output, step):
    receipt=b.load(output/'train/complete.json')
    item=receipt['checkpoints'][str(step)]
    checkpoint=output/'train'/f'checkpoint-{step}'
    assert item['step']==step and item['checkpoint']==str(checkpoint), 'wrong endpoint binding'
    assert item['checkpoint_files']==b.load(checkpoint/'identity.json')
    for name, expected in item['checkpoint_files'].items():
        assert b.sha(checkpoint/name)==expected, 'endpoint checkpoint drift'
    assert item['weight_identity']==b.online.identity(item['checkpoint_files'])
    trace=b.load(output/'train'/f'update-{step:02d}.json')
    assert item['parameter_identity']==trace['after']['parameter_identity'], 'swapped checkpoint publication'
    publication=b.load(output/'train'/f'publication-{step}.json')
    assert publication['checkpoint']==item, 'swapped checkpoint publication'
    return checkpoint,item['weight_identity']


def weight_binding(c, output, arm):
    if arm=='anchor': return Path(c['checkpoint']), c['weight_identity']
    assert arm in ENDPOINTS
    return checkpoint_binding(c,output,int(arm[-1]))


def snapshot(q, optimizer):
    """Persist parameter objects/values and Adam steps/moment objects/values in one live process."""
    named={n:t for n,t in q.model.named_parameters() if t.requires_grad}
    state={}
    for name,tensor in named.items():
        values=optimizer.state.get(tensor,{})
        state[name]=dict(step=int(values['step'].item()) if values else 0,
            objects={k:id(v) for k,v in values.items()},
            values={k:p.tensor_hash(v.detach().reshape(-1)) for k,v in values.items()})
    return dict(optimizer_object=id(optimizer),parameter_objects={n:id(t) for n,t in named.items()},
        parameter_identity=b.online.identity({n:p.tensor_hash(t.detach()) for n,t in named.items()}),
        optimizer_state=state)


def diagnostics(q, batches, c, directory, endpoint):
    import torch
    q.model.eval()
    with torch.no_grad():
        values=[hf_measure(replay(q,batches[x['image_id']],x),x) for x in c['contexts']]
    b.write(directory/f'hf-{endpoint}.json',values)
    q.model.train()


def train_updates(q, delta, batches, c, directory, started):
    import torch
    params,dora,deltas=live_training_binding(q,delta,c)
    optimizer=torch.optim.AdamW([dict(params=dora,lr=1e-5),dict(params=deltas,lr=5e-6)],
        betas=(.9,.999),eps=1e-8,weight_decay=0)
    assert not optimizer.state
    q.model.train()
    initial=snapshot(q,optimizer)
    diagnostics(q,batches,c,directory,'anchor')
    assert snapshot(q,optimizer)==initial and q.model.training, 'anchor diagnostics mutated training state'
    previous=initial
    checkpoints={}
    for step in range(1,5):
        assert time.monotonic()-started<1800, 'phase bound'
        assert q.model.training, 'training mode not restored'
        before=snapshot(q,optimizer)
        assert before==previous, 'optimizer/parameter continuity broken'
        optimizer.zero_grad(set_to_none=True)
        terms=[]
        for context in r.views(c,'R-single'):
            z=replay(q,batches[context['image_id']],context)
            assert z.numel()==c['vocab_size'], 'HF full vocabulary required'
            loss=b.online.max_geometry_margin(z,context['support'])
            assert torch.isfinite(loss)
            (loss/6).backward()
            terms.append(dict(weight=1/6,loss=float(loss.detach()),**hf_measure(z,context)))
        norms={n:float(t.grad.float().norm()) if t.grad is not None else None for n,t in q.model.named_parameters() if t.requires_grad}
        assert all(v is not None and math.isfinite(v) for v in norms.values()), 'missing/nonfinite gradient'
        gradient_norm=float(torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True))
        assert [g['lr'] for g in optimizer.param_groups]==[1e-5,5e-6]
        optimizer.step()
        assert all(torch.isfinite(t).all() for t in params), 'nonfinite parameter'
        after=snapshot(q,optimizer)
        assert all(v['step']==step for v in after['optimizer_state'].values()), 'Adam step reset'
        value=dict(step=step,arm='R-single',before=before,after=after,training_mode=True,terms=terms,
            mean_loss=sum(t['loss'] for t in terms)/6,global_clip_count=1,optimizer_step_count=1,
            gradient_norm=gradient_norm,gradient_norms=norms,lrs=[g['lr'] for g in optimizer.param_groups])
        b.write(directory/f'update-{step:02d}.json',value)
        if step in [1,4]:
            checkpoint=directory/f'checkpoint-{step}'
            p.save_checkpoint(q,delta,checkpoint)
            files=b.load(checkpoint/'identity.json')
            checkpoints[str(step)]=dict(step=step,checkpoint=str(checkpoint),checkpoint_files=files,
                weight_identity=b.online.identity(files),parameter_identity=after['parameter_identity'])
            diagnostics(q,batches,c,directory,f'step{step}')
            restored=snapshot(q,optimizer)
            assert restored==after and q.model.training, 'checkpoint/diagnostic optimizer continuity broken'
            b.write(directory/f'publication-{step}.json',dict(checkpoint=checkpoints[str(step)],before=after,
                after=restored,training_mode_restored=True,no_grad=True,diagnostic_forwards=10))
        previous=after
    return checkpoints


def train(path, output, expected_sha):
    import torch
    c=entry(path,output,expected_sha,'train')
    directory=output/'train'; directory.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    torch.manual_seed(92711); torch.cuda.manual_seed_all(92711)
    q,delta,composition=p.compose(Path(c['checkpoint']),evaluation=False)
    b.write(directory/'composition.json',composition)
    b.online.set_checkpointing(q.model,False)
    live_training_binding(q,delta,c)
    p.save_checkpoint(q,delta,directory/'checkpoint-0')
    b.online.verify_start_export(Path(c['checkpoint']),directory/'checkpoint-0')
    inputs={i['image_id']:i for i in b.load(c['input_path'])}
    batches={i:b.online.native_batch(q,inputs[i]) for i in [7511,351017]}
    checkpoints=train_updates(q,delta,batches,c,directory,started)
    names=[f'update-{i:02d}.json' for i in range(1,5)]+['composition.json']
    names += [f'hf-{i}.json' for i in ['anchor','step1','step4']]+['publication-1.json','publication-4.json']
    artifacts={name:b.sha(directory/name) for name in names}
    stats=resources(started,gpu=True)
    del q,delta,batches
    torch.cuda.empty_cache()
    assert stats['seconds']<1800
    b.write(directory/'complete.json',dict(phase='train',source=c['source'],contract_sha256=expected_sha,
        anchor_checkpoint=c['checkpoint'],anchor_weight_identity=r.WEIGHT,composition_count=1,
        fresh_optimizer=True,optimizer_count=1,checkpoints=checkpoints,artifacts=artifacts,
        counters=dict(optimizer_steps=4,training_replays=24,HF_diagnostics=30),resources=stats,
        active_seconds=stats['seconds'],HF_closed_before_publish=True,
        artifact_bytes=sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())))


def native(path, output, expected_sha, arm):
    phase = f'native-{arm}'
    c = entry(path, output, expected_sha, phase)
    q, inputs, requests, reports = frontend(c)
    assert q.tokenizer.convert_tokens_to_ids('<|im_end|>') == c['eos_token_id']
    labels = {i['image_id']:i for i in b.load(c['label_path'])}  # evaluator only
    checkpoint, identity = weight_binding(c, output, arm)
    from src.qwen.vllm_rollout import VllmDoraRollout
    directory = output/phase
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    artifacts = {}
    counters = dict(score_requests=0, continuation_requests=0, requests=0, generated_tokens=0)
    with VllmDoraRollout(base_model=c['base_model'], checkpoint=checkpoint, identity=identity,
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
        startup, operations = engine.startup, list(engine.receipts)
        active_seconds = time.monotonic()-started
    assert not engine._process.is_alive(), 'native child survived cleanup'
    assert counters['score_requests']==10 and counters['continuation_requests']==18 and counters['generated_tokens']<=55522
    b.write(directory/'complete.json',dict(phase=phase, arm=arm, source=c['source'], contract_sha256=expected_sha,
        checkpoint=str(checkpoint), weight_identity=identity, artifacts=artifacts, counters=counters,
        startup=startup, operations=operations, engine_closed_before_publish=True,
        owned_child_absent=True, active_seconds=active_seconds, resources=resources(started),
        artifact_bytes=sum((directory/name).stat().st_size for name in artifacts),
        native_cuda_memory='unmeasured_child_engine', native_internal_startup_capture_forwards=None))


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


def consume_train(c, output, expected_sha):
    directory=output/'train'
    receipt=b.load(directory/'complete.json')
    assert receipt['phase']=='train' and receipt['contract_sha256']==expected_sha and receipt['source']==c['source']
    assert receipt['anchor_checkpoint']==c['checkpoint'] and receipt['anchor_weight_identity']==r.WEIGHT
    assert receipt['composition_count']==receipt['optimizer_count']==1 and receipt['fresh_optimizer'] is True
    assert receipt['HF_closed_before_publish'] is True
    b.online.verify_start_export(Path(c['checkpoint']),directory/'checkpoint-0')
    assert set(receipt['checkpoints'])=={'1','4'}
    for step in [1,4]: checkpoint_binding(c,output,step)
    names=[f'update-{i:02d}.json' for i in range(1,5)]+['composition.json']
    names += [f'hf-{i}.json' for i in ['anchor','step1','step4']]+['publication-1.json','publication-4.json']
    assert set(receipt['artifacts'])==set(names)
    for name,sha in receipt['artifacts'].items(): assert b.sha(directory/name)==sha
    previous=None
    for step in range(1,5):
        value=b.load(directory/f'update-{step:02d}.json')
        assert value['step']==step and value['arm']=='R-single' and value['training_mode'] is True
        assert value['global_clip_count']==value['optimizer_step_count']==1
        assert value['lrs']==[1e-5,5e-6] and len(value['terms'])==6
        before,after=value['before'],value['after']
        if previous is not None: assert before==previous, 'optimizer/parameter continuity broken'
        assert before['optimizer_object']==after['optimizer_object'] and before['parameter_objects']==after['parameter_objects']
        assert set(before['optimizer_state'])==set(after['optimizer_state'])==set(before['parameter_objects'])
        for name,state in after['optimizer_state'].items():
            assert before['optimizer_state'][name]['step']==step-1 and state['step']==step, 'Adam step reset'
            assert set(state['values'])==set(state['objects'])=={'step','exp_avg','exp_avg_sq'}
            if step>1: assert before['optimizer_state'][name]['objects']==state['objects'], 'Adam moment replaced'
            else: assert before['optimizer_state'][name]['values']==before['optimizer_state'][name]['objects']=={}
        for term,context in zip(value['terms'],r.views(c,'R-single'),strict=True):
            assert term['context_id']==context['context_id'] and term['input_prefix_identity']==context['input_prefix_identity']
            assert term['weight']==1/6 and term['causal_logits_position']==context['causal_logits_position']
            assert math.isfinite(term['loss']) and math.isfinite(term['margin'])
            expected=math.log1p(math.exp(1-term['margin'])) if 1-term['margin']<700 else 1-term['margin']
            assert math.isclose(term['loss'],expected,rel_tol=2e-6,abs_tol=2e-6), 'false Gmax term'
        assert math.isclose(value['mean_loss'],sum(t['loss'] for t in value['terms'])/6,rel_tol=1e-8)
        assert math.isfinite(value['gradient_norm']) and all(v is not None and math.isfinite(v) for v in value['gradient_norms'].values())
        if step in [1,4]:
            publication=b.load(directory/f'publication-{step}.json')
            assert publication['before']==publication['after']==after, 'publication continuity broken'
            assert publication['training_mode_restored'] is publication['no_grad'] is True
            assert publication['diagnostic_forwards']==10
        previous=after
    diagnostics={}
    for endpoint in ['anchor','step1','step4']:
        data=b.load(directory/f'hf-{endpoint}.json')
        assert len(data)==10
        for value,context in zip(data,c['contexts'],strict=True):
            assert value['context_id']==context['context_id'] and value['input_prefix_identity']==context['input_prefix_identity']
            assert value['causal_logits_position']==context['causal_logits_position']
            assert math.isfinite(value['margin']) and 0<=value['legal_mass']<=1.00001
        diagnostics[endpoint]=data
    counters=dict(optimizer_steps=4,training_replays=24,HF_diagnostics=30)
    assert receipt['counters']==counters and receipt['resources']['seconds']<=1800
    return diagnostics,counters


def contrast(c, scores):
    result = {}
    for arm in ENDPOINTS:
        groups = {}
        for site,*_ in r.SITES:
            for split in ['original','training_neighbor','held_out']:
                indices = [i for i,x in enumerate(c['contexts']) if x['site_id']==site and x['split']==split]
                illegal = [i for i in indices if not scores['anchor'][i]['literal_legal']]
                legal = [i for i in indices if scores['anchor'][i]['literal_legal']]
                groups[site+'/'+split] = dict(baseline_illegal_denominator=len(illegal),
                    repaired=[c['contexts'][i]['context_id'] for i in illegal if scores[arm][i]['literal_legal']],
                    baseline_legal_denominator=len(legal),
                    retained=[c['contexts'][i]['context_id'] for i in legal if scores[arm][i]['literal_legal']],
                    lost=[c['contexts'][i]['context_id'] for i in legal if not scores[arm][i]['literal_legal']],
                    margin_changes=[dict(context_id=c['contexts'][i]['context_id'],delta=scores[arm][i]['margin']-scores['anchor'][i]['margin']) for i in indices])
        result[arm] = groups
    result['original_error_contrast_present'] = any(not scores['anchor'][i]['literal_legal'] for i,x in enumerate(c['contexts']) if x['offset']==0)
    return result


def readback(path, output, expected_sha):
    c = entry(path,output,expected_sha,PHASES[-1])
    q, inputs, _, _ = frontend(c)
    scores, natural, counters, diagnostics = {}, {}, {k:0 for k in ['optimizer_steps','training_replays','HF_diagnostics','score_requests','continuation_requests','requests','generated_tokens']}, {}
    for arm in ['anchor']+ENDPOINTS:
        scores[arm],natural[arm],counts=consume_native(c,output,arm,expected_sha,q,inputs)
        for k,v in counts.items(): counters[k]+=v
    diagnostics,counts=consume_train(c,output,expected_sha)
    for k,v in counts.items(): counters[k]+=v
    assert all(counters[k]==BOUNDS[k] for k in counters if k!='generated_tokens') and counters['generated_tokens']<=BOUNDS['generated_tokens']
    transitions = {}
    for arm,before_endpoint,after_endpoint in [('step1','anchor','step1'),('step4','anchor','step4'),('step1-to-step4','step1','step4')]:
        rows = []
        for before,after in zip(natural[before_endpoint],natural[after_endpoint],strict=True):
            assert before['image_id']==after['image_id']
            owners = b.transitions({m:before['ids'][m]['retained'] for m in ['raw','category']},
                                   {m:after['ids'][m]['retained'] for m in ['raw','category']})
            rows.append(dict(image_id=before['image_id'],owners=owners,
                burden_deltas={k:after['burdens'][k]-before['burdens'][k] for k in before['burdens']}))
        transitions[arm]=dict(images=rows,owner_counts={m:{k:sum(len(r['owners'][m][k]) for r in rows) for k in ['gained','lost','retained']} for m in ['raw','category']},
            burden_deltas={k:sum(r['burden_deltas'][k] for r in rows) for k in rows[0]['burden_deltas']})
    descriptive={}
    for endpoint in ENDPOINTS:
        rows=[]
        for image in c['images']:
            anchor=b.load(output/'native-anchor'/f'natural-{image}.json')['acquisition']['token_ids']
            tokens=b.load(output/f'native-{endpoint}'/f'natural-{image}.json')['acquisition']['token_ids']
            divergence=next((i for i,(a,z) in enumerate(zip(anchor,tokens)) if a!=z), min(len(anchor),len(tokens)) if len(anchor)!=len(tokens) else None)
            visited={x['context_id']:tokens[:len(x['extension'])]==x['extension'] for x in c['contexts'] if x['image_id']==image}
            rows.append(dict(image_id=image,first_divergence=divergence,exact_context_visitation=visited))
        descriptive[endpoint]=rows
    active_seconds=sum(b.load(output/phase/'complete.json')['active_seconds'] for phase in PHASES)
    assert active_seconds<=7200
    result=dict(status='candidate',contract_sha256=expected_sha,source=c['source'],phases=PHASES,
        counters=counters,active_seconds=active_seconds,scores=scores,conditional=contrast(c,scores),
        HF_diagnostics=diagnostics,saved_token_descriptive=descriptive,natural=natural,natural_transitions=transitions,
        annotation_unmatched_is_physical_negative=False,conditional_natural_recovery_credit=False,
        scientific_acceptance=False,next_unit_scheduled=False)
    # Readback recomputes all decision fields; re-signing claimed results cannot admit false credit.
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','qualify','validate','train','native','package','readback'])
    parser.add_argument('--contract',type=Path)
    parser.add_argument('--contract-sha256')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--endpoint',choices=['anchor']+ENDPOINTS)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.output)
    elif args.command=='qualify': qualify(args.contract,args.output)
    elif args.command=='validate': validate(args.contract)
    elif args.command=='native': native(args.contract,args.output,args.contract_sha256,args.endpoint)
    else: globals()[args.command](args.contract,args.output,args.contract_sha256)


if __name__=='__main__': main()
