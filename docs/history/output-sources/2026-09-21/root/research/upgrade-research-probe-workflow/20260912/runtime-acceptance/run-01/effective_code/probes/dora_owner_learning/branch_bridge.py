"""Two verified Source branch states: actual update entrance and continuation read."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from .candidate_opportunity import digest, file_hash, indexed, require, rows
from .route_access import OUTPUT as ROUTE_ROOT, POST, CONFIG, checkpoint_config, first_fork, publish, score_logits, validate_score
from .round1_realization import ROOT, SOURCE_ROOT
from probes.source_rweak_row_cross.owner_row_robustness import (
    OUTPUT as ROW_ROOT, branch, consume, incidence, native_record, score, validate_admission,
)
from probes.source_rweak_row_cross.run import build_requests

OUTPUT=ROOT/'2026-09-10-verified-branch-update-bridge'
FROZEN_DIGEST='cf58b0fff2fb3832b864e2dd1fabdfca07210fea38e8dadf309d76a4526e4785'
FROZEN_BYTES='ce42fd5bacf7cb053fd079f114e3589f5b34a3ef72cebd90e7035f0e7b2c1f74'
ENTRANCES={'coco2017_train_000000000368:2022537':97,'coco2017_train_000000007116:181378':22}


def entrance(case):
    """The score state ends immediately BEFORE forced x1_A, not after it."""
    job=branch(case,'partial_A');prefix=job['prefix'];a=case['A'];b=case['B']
    position=ENTRANCES[case['case_id']]
    require(len(job['forced'])==a['positions'][0]+1,'partial A must stop at x1')
    require(a['ids'][:a['positions'][0]]==b['ids'][:b['positions'][0]],'description/format changed')
    require(len(prefix)+len(job['forced'])-1==position,'entrance action-index off-by-one')
    state=prefix+job['forced'][:-1]
    require(state==case['source_ids'][:position],'entrance is not exact Source native history')
    aid,bid=a['ids'][a['positions'][0]],b['ids'][b['positions'][0]]
    require(aid!=bid and case['source_ids'][position]==bid,'A/B entrance token identity')
    return dict(action_index=position,state_ids=state,A_id=aid,B_id=bid,
                source_completed_row_prefix_tokens=len(prefix),shared_row_tokens=len(job['forced'])-1)


def summarize_logits(values, entry):
    """Full vocabulary rank, exact ties and float64 log-normalizer on saved FP32 logits."""
    import numpy as np
    require(values.ndim==1 and values.dtype==np.float32 and np.isfinite(values).all(),'entry logits schema')
    a,b=entry['A_id'],entry['B_id'];require(0<=a<len(values) and 0<=b<len(values) and a!=b,'target/B vocabulary identity')
    logits=values.astype(np.float64);target=float(logits[a]);top=int(logits.argmax())
    other=logits.copy();other[a]=-np.inf;best=int(other.argmax())
    z=float(logits.max()+np.log(np.exp(logits-logits.max()).sum()))
    logp=target-z
    return dict(A_id=a,B_id=b,vocabulary_size=len(values),target_logit=target,B_logit=float(logits[b]),
        logprob=logp,probability=math.exp(logp),rank_min=1+int((logits>target).sum()),
        rank_max=int((logits>=target).sum()),target_tie_count=int((logits==target).sum()),
        top1_id=top,top1_logit=float(logits[top]),top1_tie_count=int((logits==logits[top]).sum()),
        best_other_id=best,best_other_logit=float(logits[best]),
        A_vs_best_other_margin=target-float(logits[best]),A_vs_B_margin=target-float(logits[b]),
        log_normalizer=z,tie_policy='exact FP32 values; lowest ID wins exact argmax ties')


def prepare(output):
    from tokenizers import Tokenizer
    require(not output.exists(),'occupied output root')
    manifest_path=ROW_ROOT/'manifest-v2.json';manifest=json.loads(manifest_path.read_text())
    require(file_hash(manifest_path)==FROZEN_BYTES and digest(manifest)==FROZEN_DIGEST,'frozen row manifest identity')
    admission=json.loads((ROW_ROOT/'admission-v1.json').read_text());cases=validate_admission(manifest,admission)
    require({c['case_id'] for c in cases}==set(ENTRANCES),'exact two admitted cases required')
    prior=json.loads((ROUTE_ROOT/'inputs.json').read_text())
    groups=indexed(prior['groups'],'example_id');routes=prior['routes']
    old_consumer=json.loads((ROW_ROOT/'execution/consumer.json').read_text())
    post_raw=indexed(rows(POST/'gt_vs_pred.jsonl'),'row_id')
    tokenizer=Tokenizer.from_file(prior['plan']['model']['base_model_path']+'/tokenizer.json')
    bindings=[]
    for case in cases:
        eid=case['example_id'];ent=entrance(case);group=groups[eid]
        refs={name:next(r['ids'] for r in routes if r['example_id']==eid and f'{name}_greedy' in r['roles']) for name in ('source','post')}
        require(refs['source']==case['source_ids'],'Source reference identity')
        live=[a for a in group['actions'] if a['action_token_ids'][:len(ent['state_ids'])]==ent['state_ids'] and len(a['action_token_ids'])>len(ent['state_ids'])]
        exposures=dict(K=4,exact_state_visits=len(live),A_action_visits=sum(a['action_token_ids'][ent['action_index']]==ent['A_id'] for a in live),
            actions=[dict(seed=a['seed'],next_token_id=a['action_token_ids'][ent['action_index']],advantage=a['advantage']) for a in live])
        reach={name:dict(reached=ids[:ent['action_index']]==ent['state_ids'],
            first_divergence_from_source=first_fork(ids,refs['source'])) for name,ids in refs.items()}
        prefix_len=len(case['prefix_ids']);prefix_owners={}
        for name,ids in refs.items():
            require(ids[prefix_len:ent['action_index']]==ent['state_ids'][prefix_len:],'native row-slot header differs')
            parsed=native_record(tokenizer.decode(ids[:prefix_len],skip_special_tokens=False),{'row_id':eid},case['golden'],'conditional')
            require(parsed['dropped_prediction_count']==0 and len(parsed['pred'])==case['boundary'],'native completed-prefix row slot differs')
            prefix_owners[name]=sorted(score(parsed,seed=-1,length=prefix_len,stop='conditional')['50']['owners'])
        require(prefix_owners['source']==prefix_owners['post'],'native prefix covered-owner set differs')
        reach['native_context_validation']=dict(same_completed_prefix_owner_set=True,covered_owners=prefix_owners,
            same_row_slot_header=True,limitation='Exact-token divergence is not semantic unreachability or evidence that this prefix is necessary.')
        baseline=[r for r in old_consumer if r['case_id']==case['case_id'] and r['arm'] in ('partial_A','A')]
        require(len(baseline)==2,'Source conditional artifact coverage')
        for r in baseline:
            job=branch(case,r['arm']);require(r['prefix_ids']==job['prefix'] and r['forced_ids']==job['forced'] and r['remaining_budget']==job['remaining'],'Source saved intervention identity')
        bindings.append(dict(case=case,entrance=ent,group=group,references=refs,exposure=exposures,reachability=reach,
                             source_conditional=baseline,post_natural=post_raw[eid]))
    output.mkdir(parents=True)
    sources=[manifest_path,ROW_ROOT/'admission-v1.json',ROW_ROOT/'execution/consumer.json',ROUTE_ROOT/'inputs.json',POST/'gt_vs_pred.jsonl',CONFIG]
    packet=dict(schema='verified_branch_bridge.v1',bindings=bindings,model=prior['plan']['model'],
        train_source=prior['plan']['sources']['train_jsonl'],update_receipt=prior['update_receipt'],
        manifest=manifest,source_files={str(p):file_hash(p) for p in sources})
    publish(output/'inputs.json',packet)
    publish(output/'cpu-state-exposure.json',[dict(case_id=b['case']['case_id'],entrance=b['entrance'],exposure=b['exposure'],reachability=b['reachability']) for b in bindings])
    print(json.dumps(dict(cases=len(bindings),states=[dict(case_id=b['case']['case_id'],visits=b['exposure']['exact_state_visits'],reach=b['reachability']) for b in bindings])))


def conditional_accounting(record, case):
    """Native parser span distinguishes the inserted row from later free rows."""
    parsed=record['parsed'];row_index=case['boundary']
    # Both authorized arms use the same exact completed-row Source prefix.
    require(record['prefix_ids']==case['prefix_ids'],'conditional prefix changed')
    # Character spans prevent a parser-dropped current row from being mistaken for
    # the first valid later row. Native prefix text is retained by the caller.
    start=record.get('prefix_text_length')
    if start is None:
        # Accepted Source fixtures have no drops, so their parsed prefix ends here.
        require(parsed['dropped_prediction_count']==0,'missing native prefix boundary for malformed output')
        start=parsed['pred'][row_index-1]['char_end'] if row_index else 0
    close=record['text'].find('<|box_end|>',start)
    stop=close+len('<|box_end|>') if close>=0 else len(record['text'])
    current_indices=[i for i,p in enumerate(parsed['pred']) if p['char_start']>=start and p['char_end']<=stop]
    require(len(current_indices)<=1,'ambiguous current row spans')
    current=dict(parsed,pred=[parsed['pred'][i] for i in current_indices])
    free_indices=[i for i,p in enumerate(parsed['pred']) if p['char_start']>=stop]
    free_incidence={o:[i for i in inds if i in free_indices] for o,inds in incidence(parsed).items()}
    direct=incidence(current)
    record['bridge_current_row']=dict(parser_prediction_index=row_index,
        A_direct=bool(direct[case['owner']]),B_direct=bool(direct[case['B_owner']]),
        complete_current_row_exists=bool(current_indices),actual_current_prediction_indices=current_indices,
        free_prediction_indices=free_indices,B_free_direct=bool(free_incidence[case['B_owner']]),
        A_later_suffix_direct=bool(free_incidence[case['owner']]))
    return record


def validate_intervention(record, case):
    require(record['arm'] in ('partial_A','A'),'unauthorized intervention arm')
    job=branch(case,record['arm'])
    require(record['prefix_ids']==job['prefix'] and record['forced_ids']==job['forced'] and
            record['remaining_budget']==job['remaining'],'altered intervention prefix/forced/budget')


def reduction(packet, cold, entries):
    result=[]
    for binding in packet['bindings']:
        case=binding['case'];cid=case['case_id'];old={r['arm']:r for r in binding['source_conditional']}
        new={r['arm']:r for r in cold if r['case_id']==cid};require(set(new)=={'partial_A','A'},'post output coverage')
        entry={name:entries[(cid,name)] for name in ('source','post')}
        contrasts={}
        for arm,row in new.items():
            natural_source=score(case['golden'],seed=-1,length=len(case['source_ids']),stop=case['golden']['decode_stop_reason'])
            natural_post=score(binding['post_natural'],seed=-1,length=len(binding['references']['post']),stop=binding['post_natural']['decode_stop_reason'])
            comps={}
            for label,baseline in [('source_conditional',old[arm]['score']),('source_natural',natural_source),('post_natural',natural_post)]:
                comps[label]={}
                for t in ('50','60','80'):
                    before,after=set(baseline[t]['owners']),set(row['score'][t]['owners'])
                    comps[label][t]=dict(gained=sorted(after-before),lost=sorted(before-after),
                        baseline_metrics={k:baseline[t][k] for k in ('tp','fp','fn','f1')},
                        post_metrics={k:row['score'][t][k] for k in ('tp','fp','fn','f1')})
            source_owners=set(natural_source['50']['owners'])
            conditional_accounting(old[arm],case)
            contrasts[arm]=dict(comparisons=comps,source_current_row=old[arm]['bridge_current_row'],post_current_row=row['bridge_current_row'],
                source_B_free=old[arm]['bridge_current_row']['B_free_direct'],post_B_free=row['bridge_current_row']['B_free_direct'],
                old_Source_owners_lost=sorted(source_owners-set(row['score']['50']['owners'])),
                old_Source_owners_retained=sorted(source_owners&set(row['score']['50']['owners'])),
                source_free_tokens=len(old[arm]['suffix_ids']),post_free_tokens=len(row['suffix_ids']),
                free_suffix_tokens_identical=old[arm]['suffix_ids']==row['suffix_ids'],
                matching_reassignment_gains={t:sorted(o for o in set(row['score'][t]['owners'])-set(old[arm]['score'][t]['owners'])
                    if old[arm]['conditional']['direct_incidence_by_threshold'][t][o]) for t in ('50','60','80')},
                matching_reassignment_losses={t:sorted(o for o in set(old[arm]['score'][t]['owners'])-set(row['score'][t]['owners'])
                    if row['conditional']['direct_incidence_by_threshold'][t][o]) for t in ('50','60','80')})
        delta={k:entry['post'][k]-entry['source'][k] for k in ('logprob','probability','rank_min','A_vs_best_other_margin','A_vs_B_margin')}
        native={name:entries[(cid,name,'native')] for name in ('source','post')}
        native_delta={k:native['post'][k]-native['source'][k] for k in ('logprob','probability','rank_min','A_vs_best_other_margin','A_vs_B_margin')}
        result.append(dict(case_id=cid,entry=entry,entry_post_minus_source=delta,
            supplementary_native_context_entry=native,supplementary_native_context_post_minus_source=native_delta,
            exposure=binding['exposure'],reachability=binding['reachability'],conditional=contrasts))
    return dict(schema='verified_branch_bridge_reduction.v1',cases=result)


def execute(output):
    import numpy as np
    import torch
    from src.config.inference import load_research_infer_config
    from src.qwen.native import prepare_native_inputs,prepare_replay
    from src.qwen.generation import generate_continuations
    from .runtime import load_policy
    packet=json.loads((output/'inputs.json').read_text())
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and torch.cuda.device_count()==1,'GPU0 only')
    run=output/'execution';run.mkdir(exist_ok=False)
    started=time.monotonic();receipt=dict(status='running',pid=os.getpid(),model_loads=0,score_forwards=0,
        model_forwards=0,image_forwards=0,continuations=0,new_tokens=0,inputs_sha256=file_hash(output/'inputs.json'))
    publish(run/'launch.json',receipt)
    def expired(*_):raise TimeoutError('900 second cumulative model budget')
    signal.signal(signal.SIGALRM,expired);signal.alarm(900)
    try:
        for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'frozen source bytes changed')
        config=load_research_infer_config(CONFIG).config
        require(str(config.model.base_model)==packet['model']['base_model_path'] and str(config.adapter.path)==packet['model']['current_adapter']['root'] and
                str(config.embedding_delta.path)==packet['model']['source_embedding']['root'],'Source model config identity')
        require(config.model.dtype=='fp32' and config.backend.hf.attn_implementation=='sdpa' and config.backend.hf.patch_embed_linearization=='enabled','frozen execution numerics')
        require(file_hash(config.data.input_jsonl)==packet['train_source']['sha256'],'training input bytes')
        train=indexed(rows(config.data.input_jsonl),'image_id');images=indexed(rows(SOURCE_ROOT/'image_plan.jsonl'),'row_id')
        entries={};expected=[];output_rows=run/'rows.jsonl'
        publish(run/'code-identity.json',{str(p):file_hash(p) for p in [Path(__file__),Path(__file__).with_name('route_access.py'),Path(__file__).parents[1]/'source_rweak_row_cross/owner_row_robustness.py',CONFIG]})
        for name in ('source','post'):
            cfg=checkpoint_config(config,packet['update_receipt']['saved_adapter']['root']) if name=='post' else config
            adapter=packet['model']['current_adapter'] if name=='source' else packet['update_receipt']['saved_adapter']
            for payload,path in ((adapter,Path(cfg.adapter.path)),(packet['model']['source_embedding'],Path(cfg.embedding_delta.path))):
                for f in payload['files']:require(file_hash(path/f['relative_path'])==f['sha256'],'immutable checkpoint bytes')
            qwen,identity=load_policy(cfg,device=torch.device('cuda:0'));receipt['model_loads']+=1
            require(identity['model_identity']['adapter']['adapter_path']==str(cfg.adapter.path) and not identity['model_identity']['adapter']['merged_adapters'],'correct unmerged checkpoint')
            require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names']==['torch.float32'] and identity['effective_settings']['observed_attn_implementation']=='sdpa','observed numerics')
            publish(run/f'{name}-model.json',identity);publish(run/f'{name}-config.json',cfg.model_dump(mode='json'))
            qwen.model.eval();torch.cuda.reset_peak_memory_stats()
            def count_model(*_):receipt['model_forwards']+=1
            def count_image(*_):receipt['image_forwards']+=1
            qwen.model.register_forward_pre_hook(count_model)
            visuals=[m for n,m in qwen.model.named_modules() if n.endswith('visual')];require(len(visuals)==1,'visual counter identity')
            visuals[0].register_forward_pre_hook(count_image)
            request_cases=[]
            for b in packet['bindings']:
                c=b['case'];g=c['golden'];request_cases.append(dict(row_id=c['example_id'],row_index=g['row_index'],image_path=g['image_path'],
                    image_width=g['image_width'],image_height=g['image_height'],input_record=train[str(c['image_id'])],image_plan=images[c['example_id']]))
            requests,prompt_metadata=build_requests(qwen,cfg.model_dump(mode='json'),request_cases)
            publish(run/f'{name}-prompt-metadata.json',prompt_metadata)
            for binding,request in zip(packet['bindings'],requests,strict=True):
                case=binding['case'];ent=entrance(case);cid=case['case_id'];stem=f'{name}-{cid.replace(":","_")}'
                batch=prepare_native_inputs(qwen.processor,[request],device=torch.device('cuda:0'),record_media_identity=True)
                prompt=list(batch.prompt_token_ids[0]);require(prompt==binding['group']['prompt_token_ids'] and batch.media_sha256[0]==binding['group']['executed_media_sha256'],'prompt/media identity')
                # Primary entry forward. Full saved vocabulary enables independent arithmetic.
                require(receipt['score_forwards']<8,'score forward cap')
                with torch.inference_mode():
                    replay=prepare_replay(qwen.model,batch.inputs,prompt_token_ids=prompt,continuation_token_ids=ent['state_ids']+[ent['A_id']])
                    receipt['score_forwards']+=1
                    logits=replay.aligned_logits(qwen.model(**replay.inputs).logits)
                    require(int(replay.target_ids[-1])==ent['A_id'] and logits.shape[0]==ent['action_index']+1,'causal entry alignment')
                    values=logits[-1].detach().float().cpu().numpy().copy();del logits,replay
                path=run/f'{stem}-entry-logits.npy'
                with path.open('xb') as stream:np.save(stream,values,allow_pickle=False)
                observed=summarize_logits(values,ent)
                require(np.array_equal(np.load(path,allow_pickle=False),values) and summarize_logits(np.load(path,allow_pickle=False),ent)==observed,'entry publication/reload')
                publish(run/f'{stem}-entry.json',dict(model=name,case_id=cid,entrance=ent,readout=observed,
                    logits_sha256=file_hash(path),causal_logit_position=len(prompt)-1+ent['action_index']))
                entries[(cid,name)]=observed
                # Each model's own natural reference, not the counterfactual entrance.
                reference=binding['references'][name];require(receipt['score_forwards']<8,'parity forward cap')
                with torch.inference_mode():
                    replay=prepare_replay(qwen.model,batch.inputs,prompt_token_ids=prompt,continuation_token_ids=reference)
                    receipt['score_forwards']+=1
                    logits=replay.aligned_logits(qwen.model(**replay.inputs).logits)
                    parity=score_logits(logits,replay.target_ids,prompt_length=len(prompt))
                    native_values=logits[ent['action_index']].detach().float().cpu().numpy().copy();del logits,replay
                native_path=run/f'{stem}-native-entry-logits.npy'
                with native_path.open('xb') as stream:np.save(stream,native_values,allow_pickle=False)
                native_readout=summarize_logits(np.load(native_path,allow_pickle=False),ent)
                entries[(cid,name,'native')]=native_readout
                publish(run/f'{stem}-native-entry.json',dict(model=name,case_id=cid,readout=native_readout,
                    context_ids=reference[:ent['action_index']],logits_sha256=file_hash(native_path),
                    label='supplementary same validated row slot under each models own retained native history'))
                publish(run/f'{stem}-native-parity.json',parity)
                require(parity['first_non_argmax'] is None,'own retained native greedy parity failed')
                if name=='post':
                    for arm in ('partial_A','A'):
                        job=branch(case,arm);require(receipt['continuations']<4 and receipt['new_tokens']+job['remaining']<=12336,'continuation/token cap')
                        tick=time.monotonic()
                        result=generate_continuations(qwen.model,batch,extensions=[job['prefix']+job['forced']],budgets=[job['remaining']],
                            eos_token_id=151645,pad_token_id=qwen.tokenizer.pad_token_id,trace='none')[0]
                        require(result.request_id==case['example_id'],'continuation association');receipt['continuations']+=1;receipt['new_tokens']+=len(result.token_ids)
                        ids=job['prefix']+job['forced']+list(result.token_ids);text=qwen.tokenizer.decode(ids,skip_special_tokens=False)
                        row=dict(case_id=cid,arm=arm,model='post',manifest_sha256=FROZEN_DIGEST,action_ids=ids,prefix_ids=job['prefix'],forced_ids=job['forced'],
                            suffix_ids=list(result.token_ids),remaining_budget=job['remaining'],text=text,
                            prefix_text_length=len(qwen.tokenizer.decode(job['prefix'],skip_special_tokens=False)),
                            stop_reason=result.stop_reason,seconds=time.monotonic()-tick,
                            parsed=native_record(text,{'row_id':case['example_id']},case['golden'],result.stop_reason))
                        validate_intervention(row,case)
                        with output_rows.open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                        expected.append((cid,arm));consume(output_rows,packet['manifest'],qwen.tokenizer,expected)
            torch.cuda.synchronize()
            publish(run/f'{name}-resources.json',dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                cumulative_seconds=time.monotonic()-started,score_forwards=receipt['score_forwards']))
            if name=='post':
                cold=consume(output_rows,packet['manifest'],qwen.tokenizer,expected)
                for row in cold:
                    case=next(b['case'] for b in packet['bindings'] if b['case']['case_id']==row['case_id'])
                    validate_intervention(row,case);conditional_accounting(row,case)
                publish(run/'consumer.json',cold);publish(run/'reduction.json',reduction(packet,cold,entries))
            del qwen,batch,visuals
            gc.collect();torch.cuda.empty_cache()
        require(receipt['score_forwards']==8 and receipt['continuations']==4 and receipt['model_loads']==2,'frozen execution coverage')
        receipt['status']='completed'
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);receipt.update(elapsed_seconds=time.monotonic()-started,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        publish(run/'terminal.json',receipt)


def verify(output):
    """Repeatable CPU-only independent disk replay; never loads model weights."""
    import numpy as np
    from tokenizers import Tokenizer
    class TokenizerAdapter:
        def __init__(self,tokenizer):self.tokenizer=tokenizer
        def decode(self,*args,**kwargs):return self.tokenizer.decode(*args,**kwargs)
        def convert_tokens_to_ids(self,token):return self.tokenizer.token_to_id(token)
    packet=json.loads((output/'inputs.json').read_text());run=output/'execution'
    terminal=json.loads((run/'terminal.json').read_text())
    require(terminal['status']=='completed' and terminal['model_loads']==2 and terminal['score_forwards']==8 and
            terminal['continuations']==4 and terminal['new_tokens']<=12336 and terminal['elapsed_seconds']<900,'terminal execution bounds')
    for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'input source identity changed')
    tokenizer=TokenizerAdapter(Tokenizer.from_file(packet['model']['base_model_path']+'/tokenizer.json'))
    expected=[(b['case']['case_id'],arm) for b in packet['bindings'] for arm in ('partial_A','A')]
    cold=consume(run/'rows.jsonl',packet['manifest'],tokenizer,expected)
    entries={};files={}
    for binding in packet['bindings']:
        c=binding['case'];cid=c['case_id'];ent=entrance(c)
        for name in ('source','post'):
            stem=f'{name}-{cid.replace(":","_")}'
            for context in ('entry','native-entry'):
                path=run/f'{stem}-{context}-logits.npy';saved=json.loads((run/f'{stem}-{context}.json').read_text())
                values=np.load(path,allow_pickle=False);observed=summarize_logits(values,ent)
                require(file_hash(path)==saved['logits_sha256'] and observed==saved['readout'],'full vocabulary disk arithmetic differs')
                if context=='entry':
                    require(saved['entrance']==ent and saved['causal_logit_position']==len(binding['group']['prompt_token_ids'])-1+ent['action_index'],'saved entry alignment')
                    entries[(cid,name)]=observed
                else:
                    require(saved['context_ids']==binding['references'][name][:ent['action_index']],'native context identity')
                    entries[(cid,name,'native')]=observed
                files[str(path)]=file_hash(path)
            parity=json.loads((run/f'{stem}-native-parity.json').read_text())
            validate_score(parity,{'ids':binding['references'][name]},len(binding['group']['prompt_token_ids']))
            require(parity['first_non_argmax'] is None and all(p['target_is_argmax'] for p in parity['positions']),'native parity evidence differs')
        for row in cold:
            if row['case_id']==cid:validate_intervention(row,c);conditional_accounting(row,c)
    require(cold==json.loads((run/'consumer.json').read_text()),'fresh native consumer differs')
    result=reduction(packet,cold,entries)
    require(result==json.loads((run/'reduction.json').read_text()),'fresh reduction differs')
    require(sum(len(r['suffix_ids']) for r in cold)==terminal['new_tokens'],'generated token counter differs')
    files.update({str(p):file_hash(p) for p in (Path(__file__),run/'rows.jsonl',run/'consumer.json',run/'reduction.json')})
    return dict(status='candidate_cpu_verified',cases=2,score_forwards=8,continuations=len(cold),full_vocabulary_files=8,
        native_parity_references=4,raw_consumer_equal=True,reduction_equal=True,files_sha256=files)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute','verify']);p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args()
    if args.command=='verify':print(json.dumps(verify(args.output),indent=2))
    else:(prepare if args.command=='prepare' else execute)(args.output)


if __name__=='__main__':main()
