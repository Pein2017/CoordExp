"""Frozen soft-preservation-10 cold natural read; shared original metric consumers."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from .candidate_opportunity import digest,file_hash,indexed,require,rows
from .entrance_ce_eval import (
    OUTPUT as POINT_EVAL,validate_receipt as validate_source_receipt,consume,validate_natural,
    reduce_records as source_reduction,aggregate_scores,owner_change,
)
from .branch_bridge import summarize_logits
from .route_access import CONFIG,checkpoint_config,publish
from .round1_realization import ROOT
from .reward_rows import _pred_objects
from probes.source_rweak_row_cross.owner_row_robustness import native_record
from probes.source_rweak_row_cross.run import build_requests

ARM='soft-preservation-10'
ARM_ROOT=ROOT/'2026-09-10-selective-owner-learning-autonomous'/ARM
OUTPUT=ARM_ROOT/'evaluation'
ORIGINAL_SHA='54ff4bc5addd0eb900b35726b9eb848a3cc61b70e23ca031a41b4c4df5e95981'


def validate_manifest(packet):
    require(packet['schema']=='selective_preservation.eval.v1' and packet['arm']==ARM,'wrong evaluation arm/schema')
    original=json.loads((POINT_EVAL/'manifest.json').read_text())
    require(file_hash(POINT_EVAL/'manifest.json')==ORIGINAL_SHA,'original18 manifest changed')
    for k in ('records','configs','source_identity','source_model','selection'):
        require(packet[k]==original[k],f'frozen original {k} changed')
    cases=indexed(packet['records'],'example_id');point=indexed(packet['point_ce23'],'example_id')
    require(len(cases)==18 and set(cases)==set(point),'pointCE/natural18 IDs')
    for eid,row in cases.items():require(point[eid]['split']==row['split'] and point[eid]['parsed']['gt']==row['baseline']['gt'],'pointCE split/GT identity')


def prepare(output):
    require(not output.exists(),'occupied evaluation output')
    original=POINT_EVAL/'manifest.json';point=POINT_EVAL/'execution/consumer.json'
    require(file_hash(original)==ORIGINAL_SHA,'original natural18 manifest changed')
    packet=json.loads(original.read_text());packet.update(schema='selective_preservation.eval.v1',arm=ARM,
        point_ce23=json.loads(point.read_text()),original_manifest_sha256=ORIGINAL_SHA,
        evaluation_scope='Reused exposed train2/guard16, not new or untouched evaluation data.')
    packet['source_files'].update({str(original):file_hash(original),str(point):file_hash(point)})
    validate_manifest(packet);output.mkdir(parents=True)
    publish(output/'manifest.json',packet)
    print(json.dumps(dict(status='frozen',manifest_sha256=digest(packet),train_ids=[r['image_id'] for r in packet['records'] if r['split']=='train'],
        guard_ids=[r['image_id'] for r in packet['records'] if r['split']=='guard'])))


def validate_receipt(receipt,packet):
    require(receipt['schema_version']=='selective_preservation.training.v1' and receipt['status']=='completed','wrong/incomplete training schema')
    require(receipt['stop_reason']=='fixed_steps' and receipt['updates']==23,'candidate is not frozen fixed23 arm')
    require(Path(receipt['adapter']['root']).resolve()==(ARM_ROOT/'training/adapter').resolve(),'wrong saved candidate path')
    require(receipt['lambda_kl']==10.,'wrong preservation objective weight')
    inputs=ARM_ROOT/'training/inputs.json';require(file_hash(inputs)==receipt['inputs_sha256'],'training input packet changed')
    recipe=json.loads(inputs.read_text());require(recipe['schema_version']=='selective_preservation.inputs.v1' and recipe['lambda_kl']==10. and recipe['updates']==23,'wrong frozen training recipe')
    validate_manifest(packet)
    return validate_source_receipt(receipt,packet)


def reduce_records(records,packet):
    result=source_reduction(records,packet);result.update(schema='selective_preservation.eval_reduction.v1',arm=ARM)
    point=indexed(packet['point_ce23'],'example_id');frozen=indexed(packet['records'],'example_id')
    for split in ('train','guard'):
        selected=[r for r in records if frozen[r['example_id']]['split']==split]
        result['splits'][split]['point_ce23']=aggregate_scores([point[r['example_id']]['score'] for r in selected])
        result['splits'][split]['owner_changes_vs_point_ce23']={t:{k:sum(len(owner_change(r['score'][t]['owners'],point[r['example_id']]['score'][t]['owners'])[k]) for r in selected)
            for k in ('gained','lost','retained')} for t in ('50','60','80')}
    result['per_case_comparisons']={}
    for r in records:
        eid=r['example_id'];prior=point[eid]
        result['per_case_comparisons'][eid]=dict(vs_source=r['owner_changes'],vs_point_ce23={t:owner_change(r['score'][t]['owners'],prior['score'][t]['owners']) for t in ('50','60','80')})
        if frozen[eid]['split']=='train':
            owner=frozen[eid]['bridge_case']['owner'];result['per_case_comparisons'][eid]['target_boxes']={}
            for label,predicted,scored in [('source',frozen[eid]['baseline'],frozen[eid]['baseline_score']),('point_ce23',prior['parsed'],prior['score']),('candidate',r['parsed'],r['score'])]:
                result['per_case_comparisons'][eid]['target_boxes'][label]=target_boxes(predicted,scored,owner)
    return result


def target_boxes(parsed,scored,owner):
    projection=[i for i,p in enumerate(parsed['pred']) if _pred_objects(dict(parsed,pred=[p]))[0]]
    indices={m['pred_index'] for t in ('50','60','80') for m in scored[t]['matches'] if m['owner']==owner}
    return [dict(projected_pred_index=i,native_pred_index=projection[i],description=parsed['pred'][projection[i]]['description'],
        pixel_bbox=parsed['pred'][projection[i]]['bbox'],coord_bins=parsed['pred'][projection[i]]['coord_bins'],
        matches={t:[m for m in scored[t]['matches'] if m['owner']==owner and m['pred_index']==i] for t in ('50','60','80')}) for i in sorted(indices)]


def render_all(run,cold):
    from src.vis import render_gt_vs_prediction
    projection=run/'visualization-input';projection.mkdir(exist_ok=False)
    # Native parsed rows are projected unchanged; no likelihood scores are invented.
    for name in ('gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl'):
        with (projection/name).open('x') as stream:
            for row in cold:stream.write(json.dumps(row['parsed'])+'\n')
    publish(projection/'projection.json',dict(scope='Geometry-only unchanged native rows for shared renderer; no likelihood scores or metric filtering.',
        source=str(run/'consumer.json'),sha256=file_hash(run/'consumer.json')))
    out=run/'visualizations';require(not out.exists(),'occupied visualization output')
    rendered=render_gt_vs_prediction(projection,out,row_ids=[r['example_id'] for r in cold],duplicate_iou_threshold=.95)
    require(len(rendered.image_paths)==18 and all(p.is_file() for p in rendered.image_paths),'rendered natural18 coverage')


def verify(output):
    import numpy as np
    from tokenizers import Tokenizer
    packet=json.loads((output/'manifest.json').read_text());validate_manifest(packet);run=output/'execution'
    terminal=json.loads((run/'terminal.json').read_text());require(terminal['status']=='completed' and terminal['model_loads']==1 and terminal['score_forwards']==2 and terminal['continuations']==18
        and terminal['new_tokens']<=55512 and terminal['elapsed_seconds']<3600,'terminal bounds')
    training=json.loads(Path(terminal['training_receipt_path']).read_text());validate_receipt(training,packet)
    require(file_hash(terminal['training_receipt_path'])==terminal['training_receipt_sha256'],'completed receipt changed')
    for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'frozen source changed')
    tokenizer=Tokenizer.from_file(packet['source_model']['base_model_path']+'/tokenizer.json')
    cold=consume(run/'rows.jsonl',packet,tokenizer,[r['example_id'] for r in packet['records']])
    require(cold==json.loads((run/'consumer.json').read_text()) and reduce_records(cold,packet)==json.loads((run/'reduction.json').read_text()),'native consumer/reduction mismatch')
    require(sum(len(r['action_ids']) for r in cold)==terminal['new_tokens'],'new token counter')
    for r in packet['records']:
        if r['split']!='train':continue
        cid=r['bridge_case']['case_id'];stem=cid.replace(':','_');saved=json.loads((run/f'{stem}-cold-entry.json').read_text());path=run/f'{stem}-cold-entry-logits.npy'
        require(summarize_logits(np.load(path,allow_pickle=False),r['entrance'])==saved['readout'] and file_hash(path)==saved['logits_sha256'],'cold full-vocab score mismatch')
        require(saved['trainer']==training['final_scores'][cid] and all(abs(v)<=1e-5 for v in saved['deltas'].values()),'cold/training mismatch')
    require(len(list((run/'visualizations').glob('*.png')))==18,'visual18 coverage')
    return dict(status='candidate_cpu_verified',manifest_sha256=digest(packet),natural_outputs=18,score_checks=2,rendered_images=18,
        files_sha256={str(p):file_hash(p) for p in [Path(__file__),run/'rows.jsonl',run/'consumer.json',run/'reduction.json',run/'visualizations/manifest.json']})
def execute(output,receipt_path):
    import numpy as np
    import torch
    from src.config.inference import load_research_infer_config
    from src.qwen.native import prepare_native_inputs,prepare_replay
    from src.qwen.generation import generate_continuations
    from .runtime import load_policy
    packet=json.loads((output/'manifest.json').read_text());training=json.loads(receipt_path.read_text())
    adapter=validate_receipt(training,packet)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.device_count()==1,'GPU1 only')
    run=output/'execution';run.mkdir(exist_ok=False)
    started=time.monotonic();terminal=dict(status='running',pid=os.getpid(),model_loads=0,score_forwards=0,model_forwards=0,image_forwards=0,
        continuations=0,new_tokens=0,manifest_sha256=digest(packet),training_receipt_path=str(receipt_path),training_receipt_sha256=file_hash(receipt_path))
    publish(run/'launch.json',terminal)
    def expired(*_):raise TimeoutError('3600 second cumulative evaluator budget')
    signal.signal(signal.SIGALRM,expired);signal.alarm(3600)
    try:
        for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'frozen evaluation source changed')
        config=checkpoint_config(load_research_infer_config(CONFIG).config,adapter['root'])
        require(str(config.model.base_model)==packet['source_model']['base_model_path'] and str(config.embedding_delta.path)==packet['source_model']['source_embedding']['root'],'candidate base/embedding config')
        qwen,identity=load_policy(config,device=torch.device('cuda:0'));terminal['model_loads']=1
        require(identity['model_identity']['adapter']['adapter_path']==adapter['root'] and not identity['model_identity']['adapter']['merged_adapters'],'saved unmerged candidate identity')
        require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names']==['torch.float32'] and identity['effective_settings']['observed_attn_implementation']=='sdpa','observed FP32 SDPA')
        publish(run/'model.json',identity);publish(run/'config.json',config.model_dump(mode='json'))
        publish(run/'code.json',{str(p):file_hash(p) for p in [Path(__file__),Path(__file__).with_name('branch_bridge.py'),Path(__file__).with_name('route_access.py'),CONFIG]})
        qwen.model.eval();torch.cuda.reset_peak_memory_stats()
        def count_model(*_):terminal['model_forwards']+=1
        def count_image(*_):terminal['image_forwards']+=1
        qwen.model.register_forward_pre_hook(count_model);visual=[m for n,m in qwen.model.named_modules() if n.endswith('visual')]
        require(len(visual)==1,'visual counter identity');visual[0].register_forward_pre_hook(count_image)
        requests={}
        for frozen in packet['records']:
            request,_=build_requests(qwen,packet['configs'][frozen['split']],[frozen['case']])
            require(list(request[0].expected_token_ids)==frozen['prompt_token_ids'],'cold reconstructed prompt changed')
            requests[frozen['example_id']]=request[0]
        def prepare_batch(frozen):
            batch=prepare_native_inputs(qwen.processor,[requests[frozen['example_id']]],device=torch.device('cuda:0'),record_media_identity=True)
            require(list(batch.prompt_token_ids[0])==frozen['prompt_token_ids'] and batch.media_sha256[0]==frozen['case']['image_plan']['executed_media_sha256'],'cold prompt/media identity')
            return batch
        # Complete both checkpoint-score checks before ANY natural outcome.
        for frozen in [r for r in packet['records'] if r['split']=='train']:
            batch=prepare_batch(frozen);ent=frozen['entrance'];cid=frozen['bridge_case']['case_id']
            with torch.inference_mode():
                replay=prepare_replay(qwen.model,batch.inputs,prompt_token_ids=frozen['prompt_token_ids'],continuation_token_ids=ent['state_ids']+[ent['A_id']])
                terminal['score_forwards']+=1
                logits=replay.aligned_logits(qwen.model(**replay.inputs).logits)
                require(logits.shape[0]==ent['action_index']+1 and int(replay.target_ids[-1])==ent['A_id'],'cold score causal alignment')
                values=logits[-1].detach().float().cpu().numpy().copy();del logits,replay
            path=run/f"{cid.replace(':','_')}-cold-entry-logits.npy"
            with path.open('xb') as stream:np.save(stream,values,allow_pickle=False)
            observed=summarize_logits(np.load(path,allow_pickle=False),ent);saved=training['final_scores'][cid]
            deltas={k:observed[k]-saved[k] for k in ('logprob','A_vs_best_other_margin')}
            require(all(abs(v)<=1e-5 for v in deltas.values()) and observed['A_id']==saved['target_id'] and
                all(observed[k]==saved[k] for k in ('rank_min','top1_id')),'saved checkpoint final score mismatch')
            publish(run/f"{cid.replace(':','_')}-cold-entry.json",dict(readout=observed,trainer=saved,deltas=deltas,tolerance=1e-5,logits_sha256=file_hash(path)))
        require(terminal['score_forwards']==2,'cold fixed-state score coverage')
        path=run/'rows.jsonl';expected=[]
        with path.open('x') as stream:
            for frozen in packet['records']:
                batch=prepare_batch(frozen);tick=time.monotonic()
                require(terminal['continuations']<18 and terminal['new_tokens']+3084<=55512,'natural execution bound')
                result=generate_continuations(qwen.model,batch,extensions=[[]],budgets=[3084],eos_token_id=151645,pad_token_id=qwen.tokenizer.pad_token_id,trace='none')[0]
                require(result.request_id==frozen['example_id'],'natural request association');terminal['continuations']+=1;terminal['new_tokens']+=len(result.token_ids)
                ids=list(result.token_ids);text=qwen.tokenizer.decode(ids,skip_special_tokens=False)
                row=dict(example_id=frozen['example_id'],split=frozen['split'],manifest_sha256=digest(packet),action_ids=ids,prefix_ids=[],forced_ids=[],remaining_budget=3084,
                    text=text,stop_reason=result.stop_reason,seconds=time.monotonic()-tick,
                    parsed=native_record(text,frozen['case'],frozen['baseline'],result.stop_reason))
                validate_natural(row,frozen,qwen.tokenizer);stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno());expected.append(row['example_id'])
        cold=consume(path,packet,qwen.tokenizer,expected);publish(run/'consumer.json',cold);publish(run/'reduction.json',reduce_records(cold,packet))
        require(terminal['continuations']==18,'natural output count');render_all(run,cold);terminal['status']='completed'
    except BaseException as exc:
        terminal.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);terminal.update(elapsed_seconds=time.monotonic()-started,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        publish(run/'terminal.json',terminal)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute','verify']);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--receipt',type=Path,default=ARM_ROOT/'training/receipt.json');args=p.parse_args()
    if args.command=='prepare':prepare(args.output)
    elif args.command=='execute':execute(args.output,args.receipt)
    else:print(json.dumps(verify(args.output),indent=2))


if __name__=='__main__':main()
