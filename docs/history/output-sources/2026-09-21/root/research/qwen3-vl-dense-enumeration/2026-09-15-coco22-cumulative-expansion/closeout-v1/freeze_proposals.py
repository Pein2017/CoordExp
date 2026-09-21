from pathlib import Path
import json, hashlib, collections
R=Path(__file__).resolve().parent.parent
O=R/'closeout-v1'
def bind(p):
 return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
raw={}; packets=[]
for p in sorted((R/'discovery-v2/review-packets-v2').glob('*/packet.json')):
 d=json.loads(p.read_text());packets.append(bind(p))
 for pol in d['policies']:
  for v in pol['proposals']:
   k=v['proposal_id'];assert k not in raw
   raw[k]={'proposal_id':k,'image_id':d['image_id'],'raw_status':v['status'],'status':'HOLD','reason':'no lead-closed disposition at user freeze','source_packet':str(p),'drawable':bool(v.get('individual_full_image_bbox_overlay'))}
assert len(raw)==821
ann=[json.loads(x) for x in (R/'annotations-v5/annotations.jsonl').read_text().splitlines()]
physical={u['stable_owner_id'] for row in ann for u in row['unlabeled']}
gt=json.loads((R/'gt-review-admission-v1/lead-admission.json').read_text())
# Only admitted existing GT supports a resolved physical mapping.
verified={str(g['coco_ann_id']) for g in gt['gt_owners'] if g['decision']=='verified_real_gt'}
byimage={x['image_id']:x for x in ann}
sol_map={'new-bowl-yellow':'coco22:335722:yellow-chip-bowl','new-donut-center-front':'coco22:335722:middle-plate-front-center-donut','new-donut-left-plate':'coco22:335722:middle-plate-left-donut','unclear-front-right-donut':'coco22:335722:middle-plate-front-right-donut'}
packages=[]; covered=set()
for i,f in [(335722,'sol'),(438671,'b'),(200288,'a'),(510122,'b'),(210584,'a')]:
 p=R/f'visual-review-v1/{f}-proposals/image-{i}/review.json';d=json.loads(p.read_text());packages.append(bind(p))
 for n,g in enumerate(d.get('proposal_groups',d.get('disposition_groups'))):
  ids=g.get('proposal_ids') or [d['raw_proposal_id_prefix']+x for x in g['proposal_id_suffixes']]
  gid=g.get('group_id',str(n));dis=g['disposition'];owner=g.get('physical_owner',g.get('linked_gt_coco_ann_id',g.get('physical_owner_key')))
  if i==335722 and gid in sol_map:owner=sol_map[gid];dis='lead_admitted_new_owner'
  if owner is not None:owner=str(owner)
  if owner and owner.startswith('gt:'):
   owner=str(byimage[i]['objects'][int(owner.split(':')[-1])]['coco_ann_id'])
  status='HOLD';reason='unresolved or unadmitted physical mapping'
  if dis in ('wrong_category','wrong_class_or_target'):
   status='PROCESSED';reason='persisted clear wrong-category disposition'
  elif (dis.startswith('existing_owner') or dis in ('distinct_new_owner_candidate','lead_admitted_new_owner')) and owner in verified|physical:
   status='PROCESSED';reason='mapped to existing verified GT or lead-admitted physical owner'
  elif dis in ('parser_dropped_degenerate_bbox','parser_dropped_repeated_tail'):
   status='PROCESSED';reason='definite malformed geometry/repeated parser tail; no owner promoted'
  if i==200288 and gid.endswith('tree-baseball-bat'):
   dis='wrong_category';status='PROCESSED';reason='lead correction: drawable tree patch, not unlocatable'
  for k in ids:
   assert k in raw and k not in covered;covered.add(k)
   raw[k].update(status=status,reason=reason,review=str(p),group=gid,disposition=dis,physical_owner=owner)
# Candidate-only packages are retained; no further visual adjudication occurs here.
remaining=[bind(p) for p in (R/'visual-review-v1').glob('*-proposals/image-*/review.json') if str(p) not in {x['path'] for x in packages}]
counts=collections.Counter(x['status'] for x in raw.values())
rows=sorted(raw.values(),key=lambda x:x['proposal_id'])
(O/'proposal-ledger.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
summary={'status':'lead-frozen','total_raw_proposals':len(raw),'counts':dict(counts),'lead_closed_package_raw_coverage':len(covered),'counts_by_image':{str(i):dict(collections.Counter(x['status'] for x in rows if x['image_id']==i)) for i in sorted({x['image_id'] for x in rows})},'definition':'PROCESSED means a persisted determinate disposition supported by prior lead-closed evidence; HOLD includes unreviewed, ambiguous, unlocatable, or candidate-only mappings. Parser accounting is not physical verification. All raw proposals accounted once; exhaustively reviewing them is not a launch gate.','frozen_annotations':bind(R/'annotations-v5/annotations.jsonl'),'ledger':bind(O/'proposal-ledger.jsonl'),'lead_closed_review_packages':packages,'retained_candidate_only_packages':remaining,'source_packets':packets}
(O/'proposal-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print({'total':len(raw),'counts':dict(counts),'covered_by_closed_packages':len(covered)})
