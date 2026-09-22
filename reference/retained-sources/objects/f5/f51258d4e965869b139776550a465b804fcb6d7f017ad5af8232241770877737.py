import hashlib,json,pathlib
D=pathlib.Path(__file__).parent; B=D.parents[1]; review=json.loads((D/'review.json').read_text()); packet=json.loads((B/'second-fit-review-packets-v1/image-000000025274/packet.json').read_text())
assert review['schema']=='second_fit_step16_owner_review.v1' and review['status']=='candidate_ready'
assert [x['prediction_id'] for x in review['decisions']]==[f'p{i}' for i in range(29)]
assert review['raw_row_count']==29==packet['counts']['raw_row_count']
assert {x['visual_group_id'] for x in review['decisions']}=={x['visual_group_id'] for x in packet['rendered']['visual_groups']}
fixed={x['owner_id'] for x in packet['target_catalog_references'] if x.get('role')!='crowd'}
covered=set(review['summary']['covered_owner_ids']); missing=set(review['summary']['missing_owner_ids'])
assert covered|missing==fixed and not covered&missing and len(covered)==23 and len(missing)==9
for x in review['decisions']:
 assert x['direct_CE']['bbox'] in ('positive','mask') and x['direct_CE']['description'] in ('positive','mask')
 assert len(x['evidence_paths'])==4 and all(pathlib.Path(e['path']).exists() for e in x['evidence_paths'])
 if x['physical_status'] in ('repeat','unknown','invalid_output') or x.get('owner_role')=='crowd_group' or not x['coverage_eligible']:
  assert x['direct_CE']=={'bbox':'mask','description':'mask'}
 if x['coverage_eligible']: assert x['owner_id'] in fixed and x['physical_status']=='true_unique' and x['extent']=='reasonable'
assert all(not (x.get('owner_role')=='crowd_group' and x['direct_CE']['bbox']=='positive') for x in review['decisions'])
print(json.dumps({'validated':True,'raw_rows':29,'covered':len(covered),'missing':len(missing),'repeats':review['summary']['physical_repeat_row_count'],'unknown':review['summary']['physical_unknown_row_count']},sort_keys=True))
