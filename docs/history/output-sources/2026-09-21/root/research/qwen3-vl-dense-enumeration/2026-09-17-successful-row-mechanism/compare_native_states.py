import json
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
torch.set_num_threads(2)
s={k:torch.load(R/f'stage2/runtime/native-{k}/states.pt',map_location='cpu',weights_only=False) for k in ['S','F']}
target=s['S']['target_position'];assert s['S']['input_identity']==s['F']['input_identity'];result=[]
for off in ['63','67','68','76','103']:
 assert s['S']['mrope_positions'][off]['position_ids']==s['F']['mrope_positions'][off]['position_ids']
 for site in ['6','13','20','head']:
  vectors=[]
  for k in ['S','F']:
   c=s[k]['captures'][off];v=c['head_before'][target] if site=='head' else c['layers'][site]['before'][target];vectors.append(v.double())
  a,b=vectors;result.append(dict(offset=int(off),site=site,shape=list(a.shape),S_norm=float(a.norm()),F_norm=float(b.norm()),difference_norm=float((a-b).norm()),cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),position_identity=True,scope='same current format and common postrow prefix through67; later captures are route-specific histories'))
(R/'native-state-comparison.json').write_text(json.dumps({'status':'candidate_descriptive','comparisons':result,'not_claimed':'Norm/cosine differences alone do not identify a causal circuit or owner representation.'},indent=2)+'\n')
print([x for x in result if x['offset']==67])
