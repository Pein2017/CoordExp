import hashlib,json
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
p=read(R/'panel.json');d=read(R/'result.json');coeff=torch.load(R/'coefficients.pt',map_location='cpu',weights_only=True);assert torch.equal(coeff['factors'],coeff['median']/coeff['norms']);fork_count=0;baseline_sequences=0
for g in p['groups']:
 rec=read(R/f"runtime/{g['key']}-norm/receipt.json");identity=read(R/f"runtime/{g['key']}-identity/raw.json")
 for row,old in zip(identity['rows'],g['rows']):assert row['token_ids']==old['generated_token_ids'] and row['stop']==old['decode_stop_reason'];baseline_sequences+=1
 for iid,f in rec['first_forks'].items():
  path=Path(f['tensor']['path']);assert bind(path)==f['tensor'];t=torch.load(path,map_location='cpu',weights_only=True);before=t['before'];after=t['after'];assert torch.equal(before[:151670],after[:151670]) and torch.equal(before[152670:],after[152670:]);expected=(before[151670:152670].double()*coeff['factors']).to(before.dtype);assert torch.equal(expected,after[151670:152670]);assert int(before.argmax())==f['native_before_token'] and int(after.argmax())==f['treated_token'];assert before[151645]==after[151645];fork_count+=1
for iid,x in d['images'].items():
 a=set(x['baseline']['matches']['covered_owner_ids']);b=set(x['treated']['matches']['covered_owner_ids']);assert sorted(b-a)==x['gained_owner_ids'];assert sorted(a-b)==x['lost_incumbent_owner_ids'];assert sorted(a&b)==x['retained_incumbent_owner_ids']
result=dict(status='passed',baseline_sequences_exact=baseline_sequences,full_vocab_first_fork_tensors_verified=fork_count,all_noncoordinate_logits_exact=True,all_coordinate_transform_exact=True,owner_G_L_sets_recomputed=True,coefficient_formula_exact=True);(R/'saved-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
