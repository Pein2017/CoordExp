"""Task-local width-four row search using stateless native full-prefix forwards."""
from __future__ import annotations
import argparse,json,math,os,time
from pathlib import Path
import torch
from probes.dora_owner_learning.runtime import load_policy
from probes.training_set_completion.artifacts import literal_binding as _binding, write_pretty_json as _write
from src.qwen.input_identity import input_identity as _input_identity, tensor_hash as _tensor_hash
from src.config.inference import InferConfig
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs,exact_history_inputs
from src.qwen.special_token_embeddings import SelectedDeltaOutputHead


def stopped(tokens,specials):
    if not tokens:return None
    t=tokens[-1]
    if t==151649:return 'complete' if len(tokens)==4 and all(151670<=x<=152669 for x in tokens[:3]) else 'malformed_box_end'
    if t==151645:return 'eos'
    if t in specials and not 151670<=t<=152669:return 'structural_termination'
    if len(tokens)==8:return 'suffix_cap'
    return None


def row_beam(query,specials):
    beam=[dict(tokens=(),suffix_logprob=0.,stop=None)];ledger=[];discarded=0.
    for depth in range(1,9):
        pool=[];expanded=[];outside=0.
        for parent in beam:
            if parent['stop']:
                pool.append(parent);continue
            z=query(parent['tokens']);lp=z.double().log_softmax(-1)
            choices=torch.argsort(z,descending=True,stable=True)[:4].tolist()
            omitted=math.exp(parent['suffix_logprob'])*(1-float(lp[choices].exp().sum()));outside+=omitted
            children=[]
            for token in choices:
                suffix=parent['tokens']+(token,)
                child=dict(tokens=suffix,suffix_logprob=parent['suffix_logprob']+float(lp[token]),stop=stopped(suffix,specials));pool.append(child);children.append(child)
            expanded.append(dict(parent=list(parent['tokens']),parent_logprob=parent['suffix_logprob'],children=children,unexpanded_mass=omitted))
        ordered=sorted(pool,key=lambda x:(-x['suffix_logprob'],x['tokens']));kept=ordered[:4];pruned=ordered[4:];lost=sum(math.exp(x['suffix_logprob']) for x in pruned)
        discarded+=outside+lost;beam=kept
        ledger.append(dict(depth=depth,expanded=expanded,kept=kept,pruned=pruned,unexpanded_mass=outside,enumerated_pruned_mass=lost,cumulative_discarded_mass=discarded))
        assert abs(sum(math.exp(x['suffix_logprob']) for x in beam)+discarded-1)<1e-10
        if all(x['stop'] for x in beam):break
    assert len(beam)<=4 and all(x['stop'] for x in beam)
    return beam,ledger


def self_check():
    # Immutable prefix construction catches the list-aliasing pattern a cache fork must avoid.
    parent=(1,);left=parent+(2,);right=parent+(3,);assert parent==(1,) and left==(1,2) and right==(1,3)
    probabilities=torch.tensor([.4,.3,.2,.1,.0],dtype=torch.float64)
    def q(tokens):return probabilities.log()
    beam,ledger=row_beam(q,{0,1,2,3,4});assert [x['tokens'] for x in beam]==[(0,),(1,),(2,),(3,)];assert abs(sum(math.exp(x['suffix_logprob']) for x in beam)-1)<1e-12
    tied,_=row_beam(lambda _:torch.zeros(5),{0,1,2,3,4});assert [x['tokens'] for x in tied]==[(0,),(1,),(2,),(3,)]
    # A forced low-probability root cannot win cross-branch just because its suffix is easier.
    conditional=[math.log(.9),math.log(.5)];forced=[math.log(.01),math.log(.8)];assert conditional[0]>conditional[1] and forced[0]+conditional[0]<forced[1]+conditional[1]
    assert stopped((151926,151736,151985,151649),set())=='complete'
    assert stopped((151649,),set())=='malformed_box_end'
    return dict(status='passed',immutable_parent=True,tie_lowest_token=True,terminal_mass_retained=True,forced_score_omission_falsified=True)


def execute(root:Path,coordinate:int):
    start=time.monotonic();p=json.loads((root/'panel.json').read_text());out=root/'runtime'/str(coordinate);out.mkdir(parents=True,exist_ok=False)
    rec=dict(status='running',pid=os.getpid(),coordinate=coordinate,panel=_binding(root/'panel.json'),producer=_binding(Path(__file__)),model_forwards=0,vision_forwards=0,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    _write(out/'receipt.json',rec)
    try:
        assert coordinate in p['branches'];assert self_check()['status']=='passed'
        for b in p['sources']:assert _binding(Path(b['path']))==b,b['path']
        q,identity=load_policy(InferConfig.model_validate(p['config']),device=torch.device('cuda:0'));m=q.model.eval();tok=q.tokenizer;target=p['target'];g=p['group']
        requests,_=build_bound_native_requests(q,p['config'],g['cases']);batch=prepare_native_inputs(q.processor,requests,device='cuda:0',record_media_identity=True)
        accepted=json.loads(Path(p['accepted_native']['path']).read_text());assert identity==accepted['loaded_identity'] and _input_identity(batch)==accepted['input_identity'];rec.update(loaded_identity=identity,input_identity=_input_identity(batch))
        saved=json.loads(Path(p['native_raw']['path']).read_text())['rows'];assert saved[target]['token_ids'][:49]==p['history'] and saved[target]['token_ids'][49:54]==p['prelude']
        head=m.get_output_embeddings();assert isinstance(head,SelectedDeltaOutputHead) and head.bias is None
        coords=torch.tensor(p['coordinate_ids'],device='cuda:0');lookup={int(t):i for i,t in enumerate(head.selected_token_ids.tolist())};ix=torch.tensor([lookup[int(t)] for t in coords],device='cuda:0');W=head.base.weight[coords].detach()+head.shared_embed_delta[ix].detach();assert torch.equal(W,m.get_input_embeddings()(coords).detach())
        cf=torch.load(p['coefficients']['path'],map_location='cpu',weights_only=True);assert _tensor_hash(W)==cf['effective_rows_sha256'];torch.save(dict(output_rows=W.cpu(),coordinate_ids=coords.cpu(),norms=cf['norms'],bias=None),out/'readout.pt')
        versions={n:v._version for n,v in m.named_parameters()};last={}
        def count(module,args,kwargs):rec['model_forwards']+=1;assert rec['model_forwards']<=44;assert kwargs['use_cache'] is False and 'past_key_values' not in kwargs
        def vision(*args):rec['vision_forwards']+=1
        def pos(module,args,kwargs):last['positions']=kwargs['position_ids'][:,target].detach().cpu().clone()
        def capture(module,args):last['head_input']=args[0][target].detach().cpu().clone()
        hooks=[m.register_forward_pre_hook(count,with_kwargs=True),m.model.visual.register_forward_pre_hook(vision),m.model.language_model.register_forward_pre_hook(pos,with_kwargs=True),head.register_forward_pre_hook(capture)]
        def forward(actions,keep):
            immutable=tuple(actions);histories=[]
            for j,prompt in enumerate(batch.prompt_token_ids):
                suffix=list(immutable) if j==target else saved[j]['token_ids'][:len(immutable)];suffix=suffix+[tok.pad_token_id]*(len(immutable)-len(suffix));histories.append(list(prompt)+suffix)
            inputs=exact_history_inputs(m,batch.inputs,histories,pad_token_id=tok.pad_token_id,logits_to_keep=keep)
            value=m(**inputs);assert value.past_key_values is None and tuple(actions)==immutable
            return dict(logits=value.logits[target].detach().float().cpu(),head_input=last['head_input'].clone(),positions=last['positions'].clone(),action_tokens=list(immutable),input_ids_sha256=_tensor_hash(inputs['input_ids']),attention_mask_sha256=_tensor_hash(inputs['attention_mask']))
        def score(z,token):
            top=z.topk(2);competitor=float(top.values[1] if int(top.indices[0])==token else top.values[0]);return dict(token=token,logprob=float(z.double().log_softmax(-1)[token]),rank=1+int((z>z[token]).sum()),chosen_minus_best_other=float(z[token])-competitor,argmax=int(z.argmax()))
        with torch.inference_mode():
            base=p['history']+p['prelude'];r=forward(base,6);torch.save(r,out/'root.pt');old=torch.load(p['accepted_capture']['path'],map_location='cpu',weights_only=True);gate=[]
            for j in range(6):
                offset=49+j;z=r['logits'][j];v=old[str(offset)]['logits'];eps=float((z-v).abs().max());margin=float(v.topk(2).values.diff().abs()[0]);assert int(z.argmax())==int(v.argmax()) and 2*eps<margin;assert torch.equal(r['positions'][:,-6+j],old[str(offset)]['positions'][:,-1]);gate.append(dict(offset=offset,max_abs=eps,twice_error_over_margin=2*eps/margin))
            common_terms=[score(r['logits'][j],t) for j,t in enumerate(p['prelude'])];common=sum(x['logprob'] for x in common_terms);forced_token=151670+coordinate;forced=score(r['logits'][-1],forced_token);nodes={}
            def query(suffix):
                key=tuple(suffix)
                if key not in nodes:
                    data=forward(base+[forced_token]+list(key),1);data['logits']=data['logits'][0];data['head_input']=data['head_input'][0];nodes[key]=data
                return nodes[key]['logits']
            greedy=dict(tokens=(),suffix_logprob=0.,stop=None)
            specials=set(tok.all_special_ids)
            while not greedy['stop']:
                z=query(greedy['tokens']);t=int(z.argmax());new=greedy['tokens']+(t,);greedy=dict(tokens=new,suffix_logprob=greedy['suffix_logprob']+score(z,t)['logprob'],stop=stopped(new,specials))
            if coordinate==0:
                assert p['prelude']+[forced_token]+list(greedy['tokens'])==p['native_row']
                for suffix,data in nodes.items():
                    off=55+len(suffix);v=old[str(off)]['logits'];eps=float((data['logits']-v).abs().max());margin=float(v.topk(2).values.diff().abs()[0]);assert int(data['logits'].argmax())==int(v.argmax()) and 2*eps<margin;assert torch.equal(data['positions'][:,-1],old[str(off)]['positions'][:,-1]);gate.append(dict(offset=off,max_abs=eps,twice_error_over_margin=2*eps/margin))
            beam,ledger=row_beam(query,specials)
            fresh=forward(base,6);assert torch.equal(fresh['logits'],r['logits']) and torch.equal(fresh['positions'],r['positions']);rec['fresh_parent_requery_exact']=True
            finals={}
            for kind,paths in [('greedy',[greedy]),('beam',beam)]:
                for path in paths:
                    key=tuple(path['tokens'])
                    if key not in finals:finals[key]={**path,'provenance':[]}
                    finals[key]['provenance'].append(kind)
            rescored={};paths=[]
            for i,(suffix,path) in enumerate(sorted(finals.items())):
                row=p['prelude']+[forced_token]+list(suffix);data=forward(p['history']+row,len(row)+1);selected=[score(data['logits'][j],t) for j,t in enumerate(row)];full=sum(x['logprob'] for x in selected);expected=common+forced['logprob']+path['suffix_logprob'];assert abs(full-expected)<.005,(full,expected)
                ident=f'b{coordinate}-p{i}';rescored[ident]=data
                for j,t in enumerate(suffix):
                    seen=nodes[suffix[:j]]['logits'];z=data['logits'][6+j];eps=float((seen-z).abs().max());margin=float(seen.topk(2).values.diff().abs()[0]);assert int(seen.argmax())==int(z.argmax()) and 2*eps<margin,(ident,j,eps,margin)
                paths.append(dict(id=ident,suffix_token_ids=list(suffix),row_token_ids=row,row_text=tok.decode(row,skip_special_tokens=False,clean_up_tokenization_spaces=False),stop=path['stop'],provenance=path['provenance'],suffix_logprob=path['suffix_logprob'],common_prelude_logprob=common,forced_x1_logprob=forced['logprob'],full_row_logprob=expected,rescored_full_row_logprob=full,rescore_error=abs(full-expected),token_scores=selected,first_deviation_from_native=next((j for j,(a,b) in enumerate(zip(row,p['native_row'])) if a!=b),None)))
            torch.save({','.join(map(str,k)):v for k,v in nodes.items()},out/'nodes.pt');torch.save(rescored,out/'rescored.pt')
            _write(out/'tree.json',dict(branch=coordinate,common_prelude_terms=common_terms,common_prelude_logprob=common,forced_x1=forced,greedy=greedy,beam=beam,ledger=ledger,paths=paths,special_ids=sorted(specials),node_count=len(nodes),gate=gate,beam_conditional_retained_mass=sum(math.exp(x['suffix_logprob']) for x in beam),beam_conditional_discarded_mass=ledger[-1]['cumulative_discarded_mass'],unique_final_conditional_mass=sum(math.exp(x['suffix_logprob']) for x in paths),root_noop_exact=True,cache='No KV cache created or reused; fresh full-prefix inputs per node'))
        for h in hooks:h.remove()
        assert versions=={n:v._version for n,v in m.named_parameters()};torch.cuda.synchronize();rec.update(status='candidate_complete',parameters_unchanged=True,elapsed_seconds=time.monotonic()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved(),retained_unique_paths=len(paths),node_count=len(nodes))
    except BaseException as exc:
        rec.update(status='technical_invalid',error=repr(exc),elapsed_seconds=time.monotonic()-start);_write(out/'receipt.json',rec);raise
    _write(out/'receipt.json',rec)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path);ap.add_argument('--coordinate',type=int);ap.add_argument('--self-check',action='store_true');a=ap.parse_args()
    if a.self_check:print(json.dumps(self_check()))
    else:execute(a.root,a.coordinate)
