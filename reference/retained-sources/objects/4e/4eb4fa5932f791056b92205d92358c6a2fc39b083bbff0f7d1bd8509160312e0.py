import json,hashlib,collections,csv
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def rank(v,i):return int((v>v[i]).sum())+1
def main():
 p=read(R/'panel.json');rows=[];receipts=[];identities=[];readouts=[]
 for label in p['checkpoints']:
  for case in p['cells']:
   root=R/'runtime'/f"{label}-{case['image_id']}";rec=read(root/'receipt.json');assert rec['status']=='candidate_complete';assert rec['panel']==bind(R/'panel.json');assert rec['tensors']==bind(root/'tensors.pt');receipts.append(rec);identities.append(bind(root/'receipt.json'));t=torch.load(root/'tensors.pt',map_location='cpu',weights_only=True);w=t['output_rows'].double();bias=t['bias'].double();norm=w.norm(dim=1);median=norm.median();assert torch.equal(t['input_rows'],t['output_rows']);assert not rec['readout']['bias_exists'];assert torch.equal(bias,torch.zeros_like(bias))
   readouts.append(dict(cell=rec['cell'],norm_min=float(norm.min()),norm_max=float(norm.max()),norm_median=float(median),endpoint_norms={str(i):float(norm[i]) for i in [0,999]},endpoint_norm_ranks={str(i):rank(norm,i) for i in [0,999]},base_norms={str(i):float(t['base_rows'][i].norm()) for i in [0,999]},delta_norms={str(i):float(t['delta_rows'][i].norm()) for i in [0,999]},identity=rec['readout']))
   for row in case['rows']:
    H=t[row['key']+'.hidden'].double();Z=t[row['key']+'.logits'].double();cos=(H@w.T)/(H.norm(dim=1)[:,None]*norm[None,:]);equal=cos*H.norm(dim=1)[:,None]*median;recon=H@w.T+bias;assert torch.equal(recon.argmax(1),Z.argmax(1))
    for j,slot in enumerate(['x1','y1','x2','y2']):
     z=Z[j];co=cos[j];eq=equal[j];ep=0 if z[0]>=z[999] else 999;inter=int(z[1:999].argmax())+1;ce=0 if co[0]>=co[999] else 999;ci=int(co[1:999].argmax())+1;winner=int(z.argmax());cw=int(co.argmax());ix=row['tokens'].index(151648)+1+j
     rows.append(dict(checkpoint=label,image=case['image_id'],row=row['key'],stage=row['stage'],origin=row['origin'],slot=slot,observed_bin=row['tokens'][ix]-151670,raw_argmax=winner,equal_norm_argmax=cw,raw_margin=float(z.topk(2).values.diff().abs()[0]),cosine_margin=float(co.topk(2).values.diff().abs()[0]),equal_norm_margin=float(eq.topk(2).values.diff().abs()[0]),endpoint=ep,interior=inter,endpoint_gap=float(z[ep]-z[inter]),endpoint_norm=float(norm[ep]),interior_norm=float(norm[inter]),h_norm=float(H[j].norm()),endpoint_cosine=float(co[ep]),interior_cosine=float(co[inter]),bias_gap=float(bias[ep]-bias[inter]),equal_endpoint=ce,equal_interior=ci,equal_endpoint_gap=float(eq[ce]-eq[ci]),cosine_endpoint_gap=float(co[ce]-co[ci]),endpoint_raw_rank=rank(z,ep),endpoint_cosine_rank=rank(co,ep),affine_error=float((recon[j]-z).abs().max())))
 assert len(rows)==144;assert len({x['identity']['output_effective_sha256'] for x in readouts})==1
 aggregates={}
 for key,subset in [('all',rows)]+[(f'{label}',[x for x in rows if x['checkpoint']==label]) for label in p['checkpoints']]+[(stage,[x for x in rows if x['stage']==stage]) for stage in sorted({x['stage'] for x in rows})]:
  ep=[x for x in subset if x['raw_argmax'] in [0,999]];aggregates[key]=dict(slots=len(subset),raw_endpoint_wins=len(ep),equal_norm_endpoint_wins=sum(x['equal_norm_argmax'] in [0,999] for x in subset),raw_endpoint_survives=sum(x['equal_norm_argmax'] in [0,999] for x in ep),argmax_changes=sum(x['raw_argmax']!=x['equal_norm_argmax'] for x in subset))
 cost=dict(forwards=sum(x['forwards'] for x in receipts),vision_calls=sum(x['vision_calls'] for x in receipts),allocated_gpu_seconds=sum(x['elapsed_seconds'] for x in receipts),peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in receipts),raw_tensor_bytes=sum((R/'runtime'/x['cell']/'tensors.pt').stat().st_size for x in receipts),max_affine_error=max(x['affine_error'] for x in rows),max_saved_logits_error=max(y['saved_logits_max_abs'] or 0 for x in receipts for y in x['rows']),max_twice_error_over_margin=max(2*x['affine_error']/x['raw_margin'] for x in rows))
 assert cost['forwards']==36 and cost['allocated_gpu_seconds']<7200 and cost['raw_tensor_bytes']<p['bounds']['raw_tensor_bytes_ceiling']
 result=dict(status='candidate',panel=bind(R/'panel.json'),consumer=bind(Path(__file__)),slots=rows,aggregates=aggregates,readouts=readouts,cost=cost,receipt_bindings=identities)
 (R/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 with (R/'slots.tsv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
 print(json.dumps(dict(aggregates=aggregates,readout=readouts[0],cost=cost),indent=2))
if __name__=='__main__':main()
