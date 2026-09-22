"""Independent retained-draw verification; no model forwards."""
import argparse,hashlib,json
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def bind(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
def check(folder,cuda=False):
 rec=read(folder/'receipt.json');raw=read(folder/'raw.json');assert rec['status']=='candidate_complete';assert rec['raw']==bind(folder/'raw.json')
 panel=read(rec['panel']['path']);assert rec['panel']==bind(rec['panel']['path']);c=panel['cases'][0];t=c['target_position'];old=read(c['saved_raw']['path']);ids=raw['rows'][t]['token_ids'];assert ids[:54]==c['target_prefix_token_ids']
 for j,row in enumerate(raw['rows']):
  if j!=t:assert row['token_ids']==old['rows'][j]['token_ids'] and row['stop']==old['rows'][j]['stop']
 assert raw['input_identity']==read(c['saved_receipt']['path'])['input_identity']
 p=raw['sampling_pulse'];assert p['seed']==c['sampling_seed'] and p['temperature']==c['sampling_temperature'];assert 1<=p['draw_count']<=32
 assert p['logits']==bind(folder/'sampled-logits.pt') and p['rng_states']==bind(folder/'rng-states.pt')
 caps=torch.load(folder/'sampled-logits.pt',map_location='cpu',weights_only=True);rng=torch.load(folder/'rng-states.pt',map_location='cpu',weights_only=True)
 assert caps['offsets']==list(range(54,54+p['draw_count']));assert p['last_active_offset']==caps['offsets'][-1]
 if cuda:
  seeded=torch.Generator(device='cuda:0').manual_seed(p['seed']);assert torch.equal(seeded.get_state().cpu(),rng['before'])
 previous=rng['before'];errors=[];decisions=[]
 for s in p['steps']:
  off=s['offset'];v=caps['captures'][off];z=v['raw_logits'];after=v['after_scores'];chosen=s['sampled_token_id'];assert ids[off]==chosen
  assert torch.equal(v['rng_before'],previous);previous=v['rng_after'];assert torch.isfinite(z).all() and torch.isfinite(v['lm_head_input']).all()
  assert int(z.argmax())==s['raw_winner_token_id']==s['scaled_winner_token_id'];assert int(after.argmax())==chosen
  assert after[chosen]==0 and torch.isneginf(after).sum()==len(after)-1
  lp=float(torch.log_softmax(z.float()/p['temperature'],0)[chosen]);errors.append(abs(lp-s['sampled_logprob_at_temperature']));assert errors[-1]<2e-5
  assert s['winner_changed']==(chosen!=int(z.argmax()))
  top=torch.topk(z,2);last_open=max((j for j,vv in enumerate(ids[:off]) if vv==151648),default=-1);slot=off-last_open-1
  field=['x1','y1','x2','y2'][slot] if last_open>=0 and 0<=slot<4 else ('category' if ids[off-1]==151646 else 'wrapper_or_stop')
  probs=torch.softmax(z,0);lpraw=torch.log_softmax(z,0)
  decisions.append(dict(offset=off,field=field,raw_winner=int(top.indices[0]),selected=chosen,changed=s['winner_changed'],raw_top2_margin=float(top.values[0]-top.values[1]),raw_selected_logprob=float(lpraw[chosen]),sampled_selected_logprob=lp,raw_eos_logprob=float(lpraw[151645]),raw_entropy=float(-(probs*lpraw).sum()),head_shape=list(v['lm_head_input'].shape),head_dtype=str(v['lm_head_input'].dtype)))
  if cuda:
   g=torch.Generator(device='cuda:0');g.set_state(v['rng_before']);draw=int(torch.multinomial(torch.softmax(z.cuda().float()/p['temperature'],0),1,generator=g));assert draw==chosen;assert torch.equal(g.get_state().cpu(),v['rng_after'])
 assert torch.equal(previous,rng['after'])
 coordinate_ids=set(panel['coordinate_ids'])
 def complete(tokens):
  if len(tokens)<9 or tokens[-1]!=151649:return False
  b=len(tokens)-6
  if tokens[b]!=151648 or tokens[b-1]!=151647 or not all(t in coordinate_ids for t in tokens[b+1:b+5]):return False
  starts=[j for j,t in enumerate(tokens[:b-1]) if t==151646]
  return bool(starts and starts[-1]<b-2 and all(t not in (151646,151647,151648,151649,151645) for t in tokens[starts[-1]+1:b-1]))
 first_complete=next((off for off in caps['offsets'] if complete(ids[:off+1])),None)
 if p['complete_row']:assert first_complete==p['last_active_offset']
 else:assert first_complete is None and (p['draw_count']==32 or ids[p['last_active_offset']]==151645)
 assert all(s['native_winner_token_id']==s['token_id'] for s in raw['forced_replay'])
 return dict(receipt=bind(folder/'receipt.json'),draws=p['draw_count'],max_cpu_logprob_error=max(errors),cuda_rng_replay=cuda,companions_exact=True,decisions=decisions,pulse_raw_joint_logprob=sum(x['raw_selected_logprob'] for x in decisions))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--cell');ap.add_argument('--cuda',action='store_true');ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();folders=[R/'runtime/C'/a.cell] if a.cell else sorted((R/'runtime/C').glob('*'));checks=[check(f,a.cuda) for f in folders if (f/'raw.json').exists()];out=dict(status='passed',checks=checks,executions=len(checks),draws=sum(x['draws'] for x in checks),verifier=bind(__file__));a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status=out['status'],executions=len(checks),draws=out['draws'])))
