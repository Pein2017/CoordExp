"""CPU projection of captured decisions; all underlying vectors remain on disk."""
import json,hashlib,collections
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def family(t):return 'coordinate' if 151670<=t<=152669 else 'EOS' if t==151645 else 'wrapper' if t in [151646,151647,151648,151649] else 'category_or_other'
def field(history):
 if not history or history[-1]==151649:return 'row_admission_or_EOS'
 if history[-1]==151647:return 'box_open'
 last=max((j for j,x in enumerate(history) if x==151648),default=-1)
 if last>=0 and 151649 not in history[last:]:
  n=len(history)-last-1;return ['x1','y1','x2','y2','box_close'][n] if n<5 else 'malformed'
 return 'category_or_other'
result={};seen=set()
for e in read(R/'runtime-manifest.json')['cells']:
 folder=Path(e.get('reuse_from',str(Path(e['output_root'])/str(e['image_id'])/e['mode'])))
 if folder in seen or not(folder/'raw.json').exists():continue
 seen.add(folder);raw=read(folder/'raw.json');panel=read(Path(e.get('reuse_receipt_panel',e['panel'])['path']));t=panel['cases'][0]['target_position'];ids=raw['rows'][t]['token_ids'];sparse=torch.load(folder/'pulse-captures.pt',map_location='cpu',weights_only=True)['captures'];rows=[]
 for off,v in sparse.items():
  a,b=v['before_logits'],v['after_logits'];x,y=int(a.argmax()),int(b.argmax());lp=torch.log_softmax(a.float(),-1);top=a.topk(2).values
  rows.append(dict(offset=off,field=field(ids[:off]),history_sha256=hashlib.sha256(json.dumps(ids[:off],separators=(',',':')).encode()).hexdigest(),raw_winner=x,intervened_winner=y,actual_token=ids[off],raw_top2_margin=float(top[0]-top[1]),raw_logprob=float(lp[x]),intervened_choice_raw_logprob=float(lp[y]),raw_entropy=float(-(lp.exp()*lp).sum()),EOS_logprob=float(lp[151645]),head_shape=list(v['lm_head_input'].shape),head_dtype=str(v['lm_head_input'].dtype)))
 pulse=raw.get('sampling_pulse',raw.get('norm_pulse'))
 changes=collections.Counter(f'{family(x["native_winner_token_id"])}->{family(x["transformed_winner_token_id"])}' for x in pulse['steps'] if x['winner_changed'])
 result[str(folder.relative_to(R))]=dict(sparse=rows,pulse_change_families=dict(changes),tensor_path=str(folder/'pulse-captures.pt'))
(R/'fork-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cells=len(result),captured=sum(len(x['sparse']) for x in result.values()))))
