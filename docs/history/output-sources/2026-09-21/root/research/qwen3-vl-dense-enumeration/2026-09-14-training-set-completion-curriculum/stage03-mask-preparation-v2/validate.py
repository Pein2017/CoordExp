import json,hashlib,collections
from pathlib import Path
B=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum'); V1=B/'stage03-mask-preparation-v1'; O=B/'stage03-mask-preparation-v2'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
a=[json.loads(x) for x in open(V1/'decisions.jsonl')]; b=[json.loads(x) for x in open(O/'decisions.jsonl')]; assert len(a)==len(b)==195
assert [x['proposal_id'] for x in a]==[x['proposal_id'] for x in b]
assert collections.Counter(x['step'] for x in b)=={16:184,32:11}
v3=json.load(open(B/'target-owners-complete-v3.json')); targets=collections.defaultdict(set)
for r in v3['records']:
 if r.get('role') in ('gt_atomic','new','prior_non_gt'): targets[r['image_id']].add(r['owner_id'])
changes=[]
for x,y in zip(a,b):
 assert x['raw']==y['raw'] and x['effective_owner_id']==y['effective_owner_id'] and x['coverage_eligible_v3']==y['coverage_eligible_v3'] and x['admission_applied']==y['admission_applied'] and x['root_override_applied']==y['root_override_applied']
 if x['effective_direct_CE']!=y['effective_direct_CE']: changes.append((x,y))
 atomic=y['effective_owner_id'] in targets[y['image_id']]
 if y['effective_direct_CE']['bbox']=='positive': assert atomic and y['coverage_eligible_v3']
 if not atomic: assert y['effective_direct_CE']=={'bbox':'mask','description':'mask'}
 if y['effective_class']=='unknown': assert y['effective_direct_CE']['description']=='mask'
 if y['physical_status'] in ('unknown','false','repeat') or y['effective_extent']!='reasonable': assert y['effective_direct_CE']=={'bbox':'mask','description':'mask'}
assert len(changes)==1 and changes[0][0]['proposal_id']=='first-fit:step16:image-000000025274:p16'
assert changes[0][1]['effective_owner_id']=='900100025274' and changes[0][1]['effective_direct_CE']=={'bbox':'mask','description':'mask'}
rec=json.load(open(O/'receipt.json')); assert rec['source_v1']['decisions']['sha256']==sha(V1/'decisions.jsonl'); assert rec['invariants']['covered_missing_sets_unchanged']
print('PASS v2 rows=195 spans unchanged; only change=',changes[0][0]['proposal_id'],'bbox_positive=',sum(y['effective_direct_CE']['bbox']=='positive' for y in b),'desc_positive=',sum(y['effective_direct_CE']['description']=='positive' for y in b),'non_atomic fully masked=',sum(y['effective_owner_id'] not in targets[y['image_id']] for y in b))
