"""Lead CPU replay of natural readout-norm policy and owner consequences."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

torch.set_num_threads(4)
R=Path(__file__).resolve().parent
U=Path('/data/CoordExp/.worktrees/research-probes/research/experiments')/R.name
read=lambda p:json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def th(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
p,res,term=(read(R/n) for n in ['panel.json','result.json','terminal.json'])
for b in p['sources']+term['bindings']+term['runtime_receipts']+term['raw_outputs']+term['first_fork_tensors']:
    path=Path(b['path'])
    if path.parent==U:path=R/'candidate-records'/path.name
    assert sha(path)==b['sha256'],str(path)
co=torch.load(R/'coefficients.pt',map_location='cpu',weights_only=True)
prev=R.parent/'2026-09-16-endpoint-loop-readout-state/runtime/R16-477415/tensors.pt'
W=torch.load(prev,map_location='cpu',weights_only=True)['output_rows']
norms=W.double().norm(dim=1);factors=norms.median()/norms
assert th(W)==co['effective_rows_sha256']==p['coefficients_summary']['effective_rows_sha256']
assert torch.equal(norms,co['norms']) and torch.equal(factors,co['factors'])
assert norms.median()==co['median'] and p['coordinate_ids']==list(range(151670,152670))
spec=importlib.util.spec_from_file_location('frozen_norm_consumer',R/'reduce.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
tok=AutoTokenizer.from_pretrained(p['config']['model']['base_model'],local_files_only=True)
summary={};recs=[];seqs=0;forks=0;live=[];checks=[]
for group in p['groups']:
    key=group['key'];receipts={};raws={}
    for policy in ['identity','norm']:
        root=R/'runtime'/f'{key}-{policy}';rec=read(root/'receipt.json');raw=read(root/'raw.json')
        assert rec['status']=='candidate_complete' and rec['no_parameter_mutation'] and raw['empty_prefix']
        assert rec['panel']['sha256']==sha(R/'panel.json') and rec['producer']['sha256']==sha(R/'producer.py') and rec['raw']['sha256']==sha(root/'raw.json')
        assert rec['readout']['effective_sha256']==th(W)==rec['readout']['input_sha256']
        assert rec['readout']['tied_base'] and rec['readout']['shared_delta'] and not rec['readout']['bias_exists']
        assert rec['model_forwards']==rec['processor_calls']<=3084 and rec['vision_forwards']==1
        for row in raw['rows']:
            assert row['text']==tok.decode(row['token_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        proc=Path('/proc')/str(rec['pid'])/'cmdline'
        if proc.exists() and str(R/'producer.py').encode() in proc.read_bytes().split(b'\0'):live.append(rec['pid'])
        receipts[policy]=rec;raws[policy]=raw;recs.append(rec);checks+=rec['offline_checks']
    for k in ['loaded_identity','batch_shape','prompt_position_sha256','generate_settings','readout']:
        assert receipts['identity'][k]==receipts['norm'][k],(key,k)
    for b,saved in enumerate(group['rows']):
        base=raws['identity']['rows'][b];treated=raws['norm']['rows'][b]
        assert base['token_ids']==saved['generated_token_ids'] and base['stop']==saved['decode_stop_reason'];seqs+=1
        iid=str(saved['image_id']);fork=receipts['norm']['first_forks'].get(iid)
        if fork:
            j=fork['offset'];before_ids=base['token_ids'];after_ids=treated['token_ids']
            assert before_ids[:j]==after_ids[:j] and before_ids[j]!=after_ids[j]
            assert hashlib.sha256(json.dumps(before_ids[:j]).encode()).hexdigest()==fork['history_sha256']
            z=torch.load(Path(fork['tensor']['path']),map_location='cpu',weights_only=True);native,scaled=z['before'],z['after']
            assert torch.equal(native[:151670],scaled[:151670]) and torch.equal(native[152670:],scaled[152670:])
            assert torch.equal((native[151670:152670].double()*factors).to(native.dtype),scaled[151670:152670])
            assert native[151645]==scaled[151645]
            assert int(native.argmax())==before_ids[j]==fork['original_token']==fork['native_before_token']
            assert int(scaled.argmax())==after_ids[j]==fork['treated_token'];forks+=1
        if saved['image_id'] not in p['focus_images']:continue
        expected=res['images'][iid]
        a=c.score(base,group['cases'][b],p['banks'][iid]);z=c.score(treated,group['cases'][b],p['banks'][iid])
        assert a==expected['baseline'] and z==expected['treated']
        old=set(a['matches']['covered_owner_ids']);new=set(z['matches']['covered_owner_ids'])
        assert sorted(new-old)==expected['gained_owner_ids'] and sorted(old-new)==expected['lost_incumbent_owner_ids'] and sorted(old&new)==expected['retained_incumbent_owner_ids']
        assert expected['first_changed_token']==fork['offset']
        summary[iid]={'baseline_matches':len(old),'treated_matches':len(new),'gained':sorted(new-old),'lost':sorted(old-new),'retained':sorted(old&new),'baseline_FN':a['matches']['fn_count'],'treated_FN':z['matches']['fn_count'],'baseline_tokens':a['token_count'],'treated_tokens':z['token_count'],'baseline_stop':a['stop'],'treated_stop':z['stop'],'baseline_burden':a['burden'],'treated_burden':z['burden'],'first_fork':{k:fork[k] for k in ['offset','field','original_token','treated_token']}}
assert seqs==forks==16 and len(summary)==4 and not live
assert {i:(v['baseline_matches'],v['treated_matches'],len(v['gained']),len(v['lost'])) for i,v in summary.items()}=={'351017':(1,1,0,0),'417044':(2,14,12,0),'477415':(2,18,16,0),'7116':(4,4,0,0)}
assert len(checks)==22 and checks==res['offline_checks'] and all(x['twice_error_over_margin']<1 for x in checks)
assert sum(x['model_forwards'] for x in recs)==res['cost']['model_forwards']==13086
exits={x.name:int(x.read_text()) for x in R.glob('*.exit')};assert exits and set(exits.values())=={0}
out={'status':'lead-accepted','scope':'Four-image natural R16 output-only fixed norm policy; two useful recoveries, one persistent loop, one healthy check','result':bind(R/'result.json'),'panel':bind(R/'panel.json'),'terminal':bind(R/'terminal.json'),'verifier':bind(Path(__file__)),'coefficient_formula_recomputed_from_original_effective_rows':True,'baseline_sequences_exact':seqs,'full_vocab_first_forks_recomputed':forks,'all_noncoordinate_logits_exact':True,'all_focus_saved_output_scores_recomputed_equal':True,'all_owner_G_L_sets_verified':True,'summary':summary,'exit_codes':exits,'live_owned_jobs':live,'next_phase_executed':False,'limits':['Selected four images; no population or general deployment claim.','All-coordinate output policy also changes coordinate versus noncoordinate competition.','Successful first forks are interior coordinates; endpoint-only origin and prevention versus maintenance remain unidentified.','UNKNOWN predictions remain unreviewed; physical precision is not certified.']}
(R/'lead-acceptance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'baseline_sequences':seqs,'first_fork_tensors':forks,'matches':{i:[v['baseline_matches'],v['treated_matches']] for i,v in summary.items()},'gained':sum(len(v['gained']) for v in summary.values()),'lost':sum(len(v['lost']) for v in summary.values()),'live_owned_jobs':live}))
