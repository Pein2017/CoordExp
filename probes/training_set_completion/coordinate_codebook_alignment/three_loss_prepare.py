"""Freeze the amended three-loss packet using accepted data and saved cells."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.coordinate_codebook_alignment.scale_prepare import V3, REPO, PARENT
from src.config.loader import load_train_config
from src.training.schedule import resolve_planned_step_schedule

ROOT = PARENT / '2026-09-22-coordinate-codebook-three-loss'
OLD = PARENT / '2026-09-22-coordinate-codebook-scale'
RUN = ROOT / 'three-loss1024-seed1729-16epoch'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        out.write(json.dumps(value, sort_keys=True, indent=2) + '\n')

def prepare():
    config = json.loads((OLD / 'training-config-repair-v2.json').read_text())
    original = copy.deepcopy(config)
    config['losses']['protected']['token_type_gate']['weight'] = .2
    config['losses']['protected']['raw_axis_validity_hinge']['weight'] = .01
    config['training']['epochs'] = 16
    config['checkpoint']['steps'] = [62,123,246,492,984]
    config['run'].update(artifact_root=str(ROOT), name=RUN.name, output_dir=RUN.name)
    write(ROOT/'training-config-v1.json', config)
    changes = []
    def diff(a,b,path=''):
        if isinstance(a,dict) and isinstance(b,dict):
            for k in sorted(set(a)|set(b)): diff(a.get(k),b.get(k),f'{path}.{k}'.strip('.'))
        elif a != b: changes.append({'field':path,'before':a,'after':b})
    diff(original,config)
    write(ROOT/'config-delta-v1.json',{'changes':changes,'source':binding(OLD/'training-config-repair-v2.json')})
    schedule = resolve_planned_step_schedule(load_train_config(ROOT/'training-config-v1.json').config,packs_per_epoch=492,world_size=4)
    assert schedule.resolved_max_steps == 984 and schedule.tail_fill_pack_count == 0
    exposure=json.loads((V3/'packing-exposure.json').read_text())
    exposure['schedule']=schedule.to_artifact_dict()
    exposure['global_pack_indices']=exposure['global_pack_indices'][:7872]
    exposure['rank_checks']['4']['microsteps_per_rank']=1968
    exposure['prefix_source']=binding(V3/'packing-exposure.json')
    write(ROOT/'packing-exposure-v1.json',exposure)
    old=json.loads((V3/'analytical-cells.json').read_text())
    full=[s for s in old['specs'] if s['condition']=='epoch32']
    sentinel=[s for s in old['specs'] if s['condition']=='epoch16']
    sources=[s for s in old['specs'] if s['condition']=='source']
    retained=[s['row_id'] for s in sources if 'reuse' in s]
    assert len(retained)==32
    specs=[]
    for s in sources:
        s=copy.deepcopy(s)
        p=Path(s['reuse']['path']) if 'reuse' in s else OLD/'production/source/cells'/f"{s['cell_key']}.json"
        s['reuse']=binding(p);specs.append(s)
    sentinel_by_id={s['row_id']:s for s in sentinel}
    evaluation={}
    for condition,rows,checkpoint in [('ce_only_epoch16',full,OLD/'scale1024-seed1729-32epoch-repair-v2/checkpoints/step-984'),('three_loss_epoch8',sentinel,RUN/'checkpoints/step-492'),('three_loss_epoch16',full,RUN/'checkpoints/step-984')]:
        new=[]
        for i,row in enumerate(rows):
            s=copy.deepcopy(row);s.update(condition=condition,checkpoint_ref=str(checkpoint),cell_key=f"{condition}-{row['image_id']}",queue_index=i)
            s.pop('reuse',None)
            if condition=='ce_only_epoch16' and row['row_id'] in sentinel_by_id:
                prior=sentinel_by_id[row['row_id']]
                s['reuse']=binding(OLD/'production/epoch16/cells'/f"{prior['cell_key']}.json")
            else:new.append(s)
            specs.append(s)
        queue=json.loads((V3/'queue-epoch32.json').read_text());queue.update(specs=new,status='frozen_authorized_ruling01')
        qp=ROOT/'queues'/condition/'queue.json';write(qp,queue)
        evaluation[condition]={'queue':str(qp),'checkpoint':str(checkpoint),'output':str(ROOT/'production'/condition),'cells':len(new)}
    assert len(specs)==3936 and sum('reuse' in s for s in specs)==1376
    write(ROOT/'analytical-cells.json',{'specs':specs,'retained32_row_ids':retained,'count':3936,'new':2560,'reused':1376,'trajectory_cells':2656})
    write(ROOT/'evaluation-plan-v1.json',evaluation)
    write(ROOT/'evaluation-admission.json',json.loads((V3/'evaluation-admission.json').read_text()))
    write(ROOT/'packet-v1.json',{'status':'CPU_packet','root':str(ROOT),'bindings':[binding(ROOT/p) for p in ['training-config-v1.json','packing-exposure-v1.json','analytical-cells.json','evaluation-plan-v1.json','evaluation-admission.json','config-delta-v1.json']],'retained32_row_ids':retained,'model_calls':0})

if __name__=='__main__': prepare()

def freeze_launch():
    from src.artifacts.source_provenance import preserve_source
    from src.config.paths import resolve_run_directory
    from src.qwen import load_qwen_components
    from src.training.pack_cache import build_packing_cache_fingerprint, load_cache_manifest
    import importlib
    import importlib.metadata
    cfg=load_train_config(ROOT/'training-config-v1.json').config
    assert resolve_run_directory(cfg,cwd=REPO).run_dir == RUN and not RUN.exists()
    assert cfg.run.collision_policy == 'fail'
    components=load_qwen_components(cfg,load_model=False)
    fp=build_packing_cache_fingerprint(cfg,components,dataset=cfg.data.train,split='train')
    cache=V3/'packing-cache'/fp
    cache_manifest=load_cache_manifest(cache,expected_fingerprint=fp)
    assert cache_manifest['micro_step_count']==492
    assert fp=='78406920e488c24dfbe4fd37711dcbc45ff54322e8a6bc558de470a1d19c0e18'
    old_launch=json.loads((OLD/'launch-v1.json').read_text())
    # Source/base/data bindings are verified against the accepted launch. Current
    # implementation is captured below, rather than pretending historical code is current.
    inputs=[x for x in old_launch['bindings'] if not x['path'].startswith(str(REPO)) and not '/queues/' in x['path']]
    for b in inputs:
        assert binding(b['path'])['sha256']==b['sha256'], b['path']
    evaluation=json.loads((ROOT/'evaluation-plan-v1.json').read_text())
    assert len({Path(x['queue']).parent for x in evaluation.values()})==3
    for e in evaluation.values():
        assert not Path(e['queue']).with_name('queue-state.json').exists()
    specs=json.loads((ROOT/'analytical-cells.json').read_text())['specs']
    accepted=json.loads((OLD/'reductions/lead-final-replay-v1.json').read_text())
    accepted_hashes={b['path']:b['sha256'] for b in accepted['input_bindings']}
    for s in specs:
        if 'reuse' in s:
            b=s['reuse'];assert binding(b['path'])['sha256']==b['sha256']==accepted_hashes[b['path']]
    unit=REPO/'research/experiments/2026-09-22-coordinate-codebook-three-loss/unit.md'
    ruling=unit.with_name('lead-ruling-01.md')
    assert binding(unit)['sha256']=='779d7833fb586311c2e15809f3faa6cd97151524aaa08f4e914d2373d8513f47'
    assert binding(ruling)['sha256']=='01314f58ea2069fffcf18c67b88dd523ab0c56eccbc81f1d144306c0542afdc5'
    sources=list((REPO/'src').rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))+[unit,ruling,REPO/'probes/training_set_completion/artifacts.py']
    captures=[]
    for source in sorted(set(sources)):
        capture=preserve_source(source,run_root=ROOT/'launch-v1.json',relative_name=source.relative_to(REPO))
        captures.append({'current':binding(source),'capture':binding(capture)})
    for name in ('transformers.models.qwen3_vl.modeling_qwen3_vl','transformers.models.qwen3_vl.processing_qwen3_vl','peft.tuners.lora.layer','peft.tuners.lora.dora'):
        source=Path(importlib.import_module(name).__file__)
        capture=preserve_source(source,run_root=ROOT/'launch-v1.json',relative_name=Path('runtime')/(name+'.py'))
        captures.append({'current':binding(source),'capture':binding(capture)})
    records=inputs+[binding(ROOT/f) for f in ['training-config-v1.json','packing-exposure-v1.json','analytical-cells.json','evaluation-admission.json','packet-v1.json']]+[binding(e['queue']) for e in evaluation.values()]+[x['current'] for x in captures]
    argv=['python','-B','-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','probes.training_set_completion.coordinate_codebook_alignment.three_loss_train','--config',str(ROOT/'training-config-v1.json'),'--output',str(ROOT/'first-production.json'),'--packing-plan',str(ROOT/'packing-exposure-v1.json')]
    write(ROOT/'launch-v1.json',{'status':'mechanical_launch_checks_passed','root':str(ROOT),'run_dir':str(RUN),'cache_root':str(V3/'packing-cache'),'cache_fingerprint':fp,'cache_rebuilt':False,'training_argv':argv,'training_environment':old_launch['training_environment'],'evaluation':evaluation,'evaluation_admission':str(ROOT/'evaluation-admission.json'),'condition_order_during':['ce_only_epoch16','three_loss_epoch8','three_loss_epoch16'],'condition_order_after':['three_loss_epoch16','ce_only_epoch16','three_loss_epoch8'],'bindings':records,'source_captures':captures,'limits':old_launch['limits'],'runtime':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate','flash-attn')},'model_calls':0,'reuse_qualification':old_launch['reuse_qualification'],'implementation_allowlist':['three_loss_prepare.py','three_loss_train.py','test_three_loss_train.py','three_loss_checks.py','test_three_loss_checks.py','scale_execute.py','test_scale_execute.py','scale_reduce.py','test_scale_reduce.py','three_loss_reduce.py','test_three_loss_reduce.py']})
