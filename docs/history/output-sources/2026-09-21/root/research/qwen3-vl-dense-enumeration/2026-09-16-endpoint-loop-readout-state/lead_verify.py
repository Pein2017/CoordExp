"""Independent CPU recomputation of endpoint readout/state evidence."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
import torch

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parent
UNIT = Path('/data/CoordExp/.worktrees/research-probes/research/experiments') / ROOT.name
read = lambda p: json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def bind(p):
    return {'path':str(p),'sha256':sha(p)}
def tensor_sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()

panel,result,terminal = (read(ROOT/n) for n in ['panel.json','result.json','terminal.json'])
for b in panel['sources'] + terminal['bindings'] + terminal['runtime_receipts'] + terminal['raw_tensors']:
    path = Path(b['path'])
    if path.parent == UNIT:
        path = ROOT/'candidate-records'/path.name
    assert sha(path) == b['sha256'], str(path)
assert result['panel'] == bind(ROOT/'panel.json') and result['consumer'] == bind(ROOT/'reduce.py')
assert panel['coordinate_ids'] == list(range(151670,152670))
reference = {(r['checkpoint'],r['image'],r['row'],r['slot']):r for r in result['slots']}
rows=[];recs=[];first=None;saved_rows=0;errors=[];ratios=[];live=[]
for label in panel['checkpoints']:
    for case in panel['cells']:
        cell=f"{label}-{case['image_id']}";folder=ROOT/'runtime'/cell
        rec=read(folder/'receipt.json');recs.append(rec)
        assert rec['status']=='candidate_complete' and rec['generated_tokens']==0
        assert rec['panel']==bind(ROOT/'panel.json') and rec['tensors']==bind(folder/'tensors.pt')
        t=torch.load(folder/'tensors.pt',map_location='cpu',weights_only=True)
        assert torch.equal(t['output_rows'],t['base_rows']+t['delta_rows'])
        assert torch.equal(t['output_rows'],t['input_rows'])
        assert torch.count_nonzero(t['bias'])==0 and not rec['readout']['bias_exists']
        assert rec['readout']['tied_base'] and rec['readout']['shared_delta']
        if first is None:first=t['output_rows'].clone()
        assert torch.equal(first,t['output_rows'])
        assert tensor_sha(first)==rec['readout']['output_effective_sha256']==rec['readout']['input_effective_sha256']
        for key in ['head_source','model_source','language_source']:
            b=rec['readout'][key];assert sha(b['path'])==b['sha256']
        W=t['output_rows'].double();norms=W.norm(dim=1)
        assert norms.argsort(descending=True)[:2].tolist()==[999,0]
        for row,rrec in zip(case['rows'],rec['rows']):
            assert row['key']==rrec['key']
            hf=t[row['key']+'.hidden'];zf=t[row['key']+'.logits']
            assert tensor_sha(hf)==rrec['hidden_sha256'] and tensor_sha(zf)==rrec['logits_sha256']
            H=hf.double();Z=zf.double();affine=H@W.T
            cosine=(H/H.norm(dim=1,keepdim=True))@(W/norms[:,None]).T
            equal=H.norm(dim=1,keepdim=True)*norms.median()*cosine
            raw=Z.argmax(1);cf=cosine.argmax(1)
            assert torch.equal(affine.argmax(1),raw)
            assert [panel['coordinate_ids'][int(i)] for i in raw]==rrec['native_full_vocab_argmax']
            if row['saved_row'] is not None:
                old=ROOT.parent/'2026-09-16-corner-loop-mechanism'/f"runtime/{cell}/row{row['saved_row']}-observed-logits.pt"
                oldt=torch.load(old,map_location='cpu',weights_only=True)
                prior=torch.stack([oldt[s+'.full'][panel['coordinate_ids']] for s in ['x1','y1','x2','y2']])
                assert torch.equal(prior,zf);saved_rows+=1
            for j,slot in enumerate(['x1','y1','x2','y2']):
                ref=reference[(label,case['image_id'],row['key'],slot)]
                assert int(raw[j])==ref['raw_argmax'] and int(cf[j])==ref['equal_norm_argmax']
                ep=ref['endpoint'];inter=ref['interior']
                assert ep==(0 if Z[j,0]>=Z[j,999] else 999)
                assert inter==int(Z[j,1:999].argmax())+1
                values={'endpoint_gap':float(Z[j,ep]-Z[j,inter]),'endpoint_norm':float(norms[ep]),'interior_norm':float(norms[inter]),'h_norm':float(H[j].norm()),'endpoint_cosine':float(cosine[j,ep]),'interior_cosine':float(cosine[j,inter]),'bias_gap':0.0,'equal_endpoint_gap':float(equal[j,[0,999]].max()-equal[j,1:999].max())}
                for key,val in values.items():assert math.isclose(val,ref[key],rel_tol=1e-9,abs_tol=1e-10),(cell,row['key'],slot,key)
                err=float((affine[j]-Z[j]).abs().max());margin=float(Z[j].topk(2).values.diff().abs()[0])
                assert 2*err<margin;errors.append(err);ratios.append(2*err/margin)
                rows.append(ref)
        proc=Path('/proc')/str(rec['pid'])/'cmdline'
        if proc.exists() and str(ROOT/'producer.py').encode() in proc.read_bytes().split(b'\0'):live.append(rec['pid'])
assert len(rows)==144 and saved_rows==27 and not live
aggregate={}
for key,subset in [('all',rows)]+[(label,[r for r in rows if r['checkpoint']==label]) for label in panel['checkpoints']]+[(stage,[r for r in rows if r['stage']==stage]) for stage in sorted({r['stage'] for r in rows})]:
    ep=[r for r in subset if r['raw_argmax'] in (0,999)]
    aggregate[key]={'slots':len(subset),'raw_endpoint_wins':len(ep),'equal_norm_endpoint_wins':sum(r['equal_norm_argmax'] in (0,999) for r in subset),'raw_endpoint_survives':sum(r['equal_norm_argmax'] in (0,999) for r in ep),'argmax_changes':sum(r['raw_argmax']!=r['equal_norm_argmax'] for r in subset)}
assert aggregate==result['aggregates']
exposure=read(ROOT/'teacher-exposure.json');b=exposure['source'];assert sha(b['path'])==b['sha256']
dataset=[json.loads(line) for line in Path(b['path']).read_text().splitlines()]
counts={s:[0]*1000 for s in ['x1','y1','x2','y2']}
for item in dataset:
    for obj in item['objects']:
        for slot,token in zip(counts,obj['bbox_2d']):counts[slot][int(re.fullmatch(r'<\|coord_(\d+)\|>',token)[1])]+=1
assert counts==exposure['counts_by_role'] and len(dataset)==exposure['images']==256
assert sum(r['forwards'] for r in recs)==sum(r['vision_calls'] for r in recs)==36
exits={p.name:int(p.read_text()) for p in ROOT.glob('*.exit')};assert len(exits)==10 and set(exits.values())=={0}
out={'status':'lead-accepted','scope':'Fixed 144 conditional slots; endpoint norm contribution with surviving alignment; no repair or training-origin claim','result':bind(ROOT/'result.json'),'panel':bind(ROOT/'panel.json'),'terminal':bind(ROOT/'terminal.json'),'verifier':bind(Path(__file__)),'all_slot_predictions_and_decompositions_recomputed':True,'all_aggregates_recomputed_equal':True,'native_saved_rows_exact':saved_rows,'common_effective_readout_verified':True,'max_affine_error':max(errors),'max_twice_error_over_margin':max(ratios),'aggregates':aggregate,'canonical_teacher_counts':{s:{'total':sum(v),'bin0':v[0],'bin999':v[999]} for s,v in counts.items()},'exit_codes':exits,'live_owned_jobs':live,'candidate_records_archive':str(ROOT/'candidate-records'),'limits':['Selected conditional/off-policy slots; not natural failure rates.','Coordinate-only equal-norm sensitivity, not an executed decoder intervention.','Readout factors do not identify upstream input embedding or training cause.']}
(ROOT/'lead-acceptance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['status','native_saved_rows_exact','max_affine_error','max_twice_error_over_margin','live_owned_jobs']}|{'all':aggregate['all']}))
