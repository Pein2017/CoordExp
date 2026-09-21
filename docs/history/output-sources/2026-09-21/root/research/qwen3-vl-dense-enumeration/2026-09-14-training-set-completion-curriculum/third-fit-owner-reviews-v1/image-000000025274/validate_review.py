#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=HERE.parents[1]
review=json.loads((HERE/'review.json').read_text())
packet=json.loads((B/'third-fit-review-packets-v1/image-000000025274/packet.json').read_text())
target=json.loads((B/'target-owners-complete-v4.json').read_text())
assert review['schema']=='third_fit_step32_owner_review.v1'
assert review['status']=='candidate_ready' and review['image_id']==25274 and review['checkpoint_step']==32
raw=packet['rendered']['raw_rows']; dec=review['decisions']
assert len(raw)==len(dec)==review['raw_row_count']==30
raw_ids=[r['prediction_id'] for r in raw]; dec_ids=[r['prediction_id'] for r in dec]
assert len(set(raw_ids))==30 and dec_ids==raw_ids
assert len(review['validation']['raw_prediction_ids_exact'])==30
assert review['validation']['raw_prediction_ids_exact']==raw_ids
fixed={x['owner_id'] for x in target['records'] if x['image_id']==25274}
assert len(fixed)==33 and review['summary']['target_owner_count']==33
covered={x['owner_id'] for x in dec if x['coverage_eligible']}
missing=set(review['summary']['missing_owner_ids'])
assert covered==set(review['summary']['covered_owner_ids'])
assert covered|missing==fixed and covered&missing==set()
assert review['summary']['covered_owner_count']==len(covered)==21
assert review['summary']['missing_owner_count']==len(missing)==12
assert sum(x['physical_status']=='repeat' for x in dec)==review['summary']['physical_repeat_row_count']==2
assert sum(x['physical_status']=='unknown' for x in dec)==review['summary']['physical_unknown_row_count']==1
assert sum(x['physical_status']=='invalid_output' for x in dec)==review['summary']['invalid_output_row_count']==1
assert sum(x['class']=='wrong' for x in dec)==review['summary']['class_wrong_row_count']==0
assert sum(x['class']=='unknown' for x in dec)==review['summary']['class_unknown_row_count']==1
for r,x in zip(raw,dec):
 assert x['generated_order']==r['generated_order'] and x['visual_group_id']==r['visual_group_id']
 assert x['raw_status']==r['status'] and x['coord_bins_1000']==r['coord_bins_1000']
 assert x['raw_source_row_sha256']==r['raw_source_row_sha256'] and x['raw_span_sha256']==r['raw_span_sha256']
 assert set(x['physical_status'] for _ in [0]) <= {'true_unique','repeat','false','unknown','invalid_output'}
 assert x['extent'] in {'reasonable','wrong','unknown'} and x['class'] in {'verified','wrong','unknown'}
 assert set(x['direct_CE'])=={'bbox','description'} and x['direct_CE']['bbox'] in {'positive','mask'} and x['direct_CE']['description'] in {'positive','mask'}
 assert all(Path(p).exists() for p in x['evidence_paths'])
 if x['coverage_eligible']:
  assert x['owner_id'] in fixed and x['physical_status'] in {'true_unique','repeat'} and x['extent']=='reasonable' and x['raw_status']=='parsed_valid'
  assert x['direct_CE']=={'bbox':'positive','description':'positive'}
 else: assert x['direct_CE']=={'bbox':'mask','description':'mask'}
# Every source/evidence manifest hash is current.
for p,meta in review['validation']['source_manifest'].items():
 q=Path(p); assert q.exists(); assert hashlib.sha256(q.read_bytes()).hexdigest()==meta['sha256']
# Review notes are one durable interpretation per raw group.
notes=[json.loads(x) for x in (HERE/'review-notes.jsonl').read_text().splitlines() if x.strip()]
assert len(notes)==30 and [x['prediction_id'] for x in notes]==raw_ids
assert all(Path(p).exists() for x in notes for p in x['viewed_evidence'])
assert review['validation']['review_notes_sha256']==hashlib.sha256((HERE/'review-notes.jsonl').read_bytes()).hexdigest()
assert review['validation']['coverage_partition_exact'] is True
print('PASS third-fit step32 image25274: raw=30 valid=29 invalid=1 covered=21 missing=12 repeats=2 unknown=1')
