#!/usr/bin/env python3
import copy,hashlib,json,os,shutil
from collections import Counter,defaultdict
from pathlib import Path
B=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum')
O=B/'stage03-mask-preparation-v1'; O.mkdir(parents=True,exist_ok=True)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()
def loadjson(p): return json.load(open(p))
# Inputs
Dpath=B/'first-fit-review-extraction-v1/decisions.jsonl'; decisions=[json.loads(x) for x in open(Dpath)]
assert len(decisions)==392 and len({r['proposal_id'] for r in decisions})==392
V3path=B/'target-owners-complete-v3.json'; v3=loadjson(V3path)
Apath=B/'first-fit-new-owner-admissions-v1/admissions.json'; admissions=loadjson(Apath)
Ppath=B/'stage03-single-owner-repair-v1/owner-plan.json'; plan=loadjson(Ppath)
fixed_roles={'gt_atomic','new','prior_non_gt'}
target_by_img=defaultdict(dict); target_order=defaultdict(list)
for r in v3['records']:
 if r.get('role') in fixed_roles:
  target_by_img[int(r['image_id'])][r['owner_id']]=r; target_order[int(r['image_id'])].append(r['owner_id'])
assert sum(len(x) for x in target_by_img.values())==228
# exact admission map by generated proposal; all eight are retained in source manifest, only selected rows are projected.
adm_by_pid={x['reference_proposal_id']:x for x in admissions['entries']}
# stage03 selection: all step16 rows of the nine append images, first six step32 rows of 323322, all step32 210457.
append_images=[int(x['image_id']) for x in plan['rows'] if x.get('action')=='append_absent_gt_before_eos_then_release']
assert len(append_images)==9 and len(set(append_images))==9
selected=[]
for r in decisions:
 im,st=r['image_id'],r['step']; order=r['generated_order']
 if st==16 and im in append_images: selected.append(r)
 elif im==323322 and st==32 and order<6: selected.append(r)
 elif im==210457 and st==32: selected.append(r)
assert len(selected)==184+6+5,(len(selected),Counter((r['step'],r['image_id']) for r in selected))
# packet exact raw row span and source packet binding checks
packet_cache={}; source_manifest=[]
for im in sorted(target_by_img):
 for st in (16,32):
  pp=B/f'first-fit-v1/plots-step-{st}/image-{im:012d}/packet.json'; assert pp.is_file(),pp
  pd=loadjson(pp); packet_cache[(im,st)]=pd
  source_manifest.append({'kind':'packet','image_id':im,'step':st,'path':str(pp),'sha256':sha(pp)})
# v3 admission lookup and effective fields
rows=[]
for src in selected:
 pid=src['proposal_id']; im=src['image_id']; st=src['step']; adm=adm_by_pid.get(pid)
 owner=src['owner_id']; source_owner=owner; extent=src['extent']; cls=src['class']; direct=copy.deepcopy(src['direct_CE']); physical=src['physical_status']
 # v3 owner admissions supersede pending local IDs for the exact reviewed proposal.
 if adm:
  owner=adm['owner_id']
  if adm.get('class_policy')=='mask_description': cls='unknown'
  # The clear-glass vessel admission explicitly preserves the held raw extent.
  if owner=='first-fit:new:219546:clear-glass-serving-vessel': extent='unknown'
 # Any unknown/false/repeated physical status or non-reasonable extent has no positive direct fields.
 if physical in ('unknown','false','repeat') or extent!='reasonable':
  direct={'bbox':'mask','description':'mask'}
 elif cls=='unknown':
  direct={'bbox':'positive','description':'mask'}
 elif cls=='wrong':
  direct={'bbox':'positive','description':'mask'}
 else:
  direct={'bbox':'positive','description':'positive'}
 in_v3=owner in target_by_img[im]
 eligible=bool(in_v3 and src['raw']['status']=='parsed_valid' and physical in ('true_unique','repeat') and extent=='reasonable')
 pd=packet_cache[(im,st)]; rr=next(x for x in pd['rendered']['raw_rows'] if x['prediction_id']==src['prediction_id'])
 rows.append({'schema':'training_set_completion.stage03_reviewed_prefix_decision.v1','image_id':im,'step':st,'generated_order':src['generated_order'],'proposal_id':pid,'prediction_id':src['prediction_id'],'raw':copy.deepcopy(src['raw']),'source_owner_id':source_owner,'effective_owner_id':owner,'target_role':target_by_img[im].get(owner,{}).get('role'),'physical_status':physical,'effective_extent':extent,'effective_class':cls,'effective_direct_CE':direct,'coverage_eligible_v3':eligible,'admission_applied':copy.deepcopy(adm) if adm else None,'root_override_applied':copy.deepcopy(src.get('root_override_applied')) if src.get('root_override_applied') else None,'reason':src.get('reason'),'source_decision':{'path':str(Dpath),'sha256':sha(Dpath),'line_proposal_id':pid},'source_packet':copy.deepcopy(src['source_packet']),'evidence':copy.deepcopy(src.get('evidence',[])),'packet_raw_row_status':rr.get('status')})
rows.sort(key=lambda r:(r['image_id'],r['step'],r['generated_order']))
# Exact selected spans and image summaries.
summaries=[]
for im in sorted(set(r['image_id'] for r in rows)):
 for st in sorted(set(r['step'] for r in rows if r['image_id']==im)):
  rs=[r for r in rows if r['image_id']==im and r['step']==st]; tgt=target_order[im]
  covered=[o for o in tgt if any(r['effective_owner_id']==o and r['coverage_eligible_v3'] for r in rs)]
  missing=[o for o in tgt if o not in covered]
  orders=[r['generated_order'] for r in rs]
  summaries.append({'image_id':im,'step':st,'selected_row_count':len(rs),'selected_generated_orders':orders,'selection_span':'step16_complete' if st==16 else ('step32_first6' if im==323322 else 'step32_complete'),'v3_target_owner_count':len(tgt),'covered_v3_owner_ids':covered,'missing_v3_owner_ids':missing,'covered_count':len(covered),'missing_count':len(missing),'coverage_eligible_row_count':sum(r['coverage_eligible_v3'] for r in rs),'raw_status_counts':dict(sorted(Counter(r['raw']['status'] for r in rs).items())),'physical_status_counts':dict(sorted(Counter(r['physical_status'] for r in rs).items())),'extent_counts':dict(sorted(Counter(r['effective_extent'] for r in rs).items())),'class_counts':dict(sorted(Counter(r['effective_class'] for r in rs).items())),'direct_CE_counts':{'bbox':dict(sorted(Counter(r['effective_direct_CE']['bbox'] for r in rs).items())),'description':dict(sorted(Counter(r['effective_direct_CE']['description'] for r in rs).items()))},'admitted_owner_ids_present':sorted({r['effective_owner_id'] for r in rs if r['admission_applied']}),'root_override_proposals':[r['proposal_id'] for r in rs if r['root_override_applied']]})
# GT absence check against step16 physical identity, plus exact v3 bins.
absence=[]
for pr in plan['rows']:
 if pr.get('action')!='append_absent_gt_before_eos_then_release': continue
 im=int(pr['image_id']); oid=pr['owner_id']; expected=pr['reference_bins']
 tr=target_by_img[im].get(oid); assert tr is not None,(im,oid)
 assert tr['reference_coord_bins_1000']==expected,(im,oid,tr['reference_coord_bins_1000'],expected)
 step16=[r for r in decisions if r['image_id']==im and r['step']==16]
 seen=[r['proposal_id'] for r in step16 if r.get('owner_id')==oid]
 absence.append({'image_id':im,'owner_id':oid,'category':pr.get('category'),'plan_reference_bins':expected,'v3_catalog_bins':tr['reference_coord_bins_1000'],'identity_absent_in_step16_review':pr.get('identity_absent_in_step16_review'),'step16_owner_rows':seen,'physically_absent_verified':len(seen)==0,'check':'physical identity absent from reviewed step16 owner IDs, independent of geometry matching'})
assert all(x['physically_absent_verified'] for x in absence)
# Source manifest includes all direct source hashes plus reviews/overrides referenced by canonical rows.
for rel,kind in [('first-fit-review-extraction-v1/decisions.jsonl','canonical_decisions'),('target-owners-complete-v3.json','target_catalog_v3'),('first-fit-new-owner-admissions-v1/admissions.json','v3_admissions'),('stage03-single-owner-repair-v1/owner-plan.json','stage03_owner_plan'),('first-fit-review-extraction-v1/holds_disagreements.json','prior_holds'),('first-fit-owner-reviews-v1/image-000000099937/root-overrides.json','root_override_99937'),('B/first-fit-owner-reviews-v1/image-000000323322/root-overrides.json','root_override_323322')]:
 p=B/rel; assert p.is_file(),p; source_manifest.append({'kind':kind,'path':str(p),'sha256':sha(p)})
for r in rows:
 for e in r['evidence']:
  if isinstance(e,dict) and e.get('path'): source_manifest.append({'kind':'evidence','path':e['path'],'sha256':e.get('sha256')})
source_manifest={ (x['path'],x.get('sha256')):x for x in source_manifest }
source_manifest=sorted(source_manifest.values(),key=lambda x:(x['kind'],x['path']))
# Outputs
with open(O/'decisions.jsonl','w') as f:
 for r in rows:f.write(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n')
json.dump({'schema':'training_set_completion.stage03_reviewed_prefix_summary.v1','status':'candidate_pending_root_training_use','target_version':str(V3path),'selected_projection':{'step16_append_images':append_images,'step32_image323322_generated_orders':[0,1,2,3,4,5],'step32_image210457':'complete'},'v3_target_owner_count':sum(len(x) for x in target_by_img.values()),'summaries':summaries,'aggregate':{'selected_rows':len(rows),'step16_rows':sum(r['step']==16 for r in rows),'step32_rows':sum(r['step']==32 for r in rows),'covered_owner_sum_by_step':{str(st):sum(x['covered_count'] for x in summaries if x['step']==st) for st in (16,32)},'missing_owner_sum_by_step':{str(st):sum(x['missing_count'] for x in summaries if x['step']==st) for st in (16,32)},'direct_bbox_positive_rows':sum(r['effective_direct_CE']['bbox']=='positive' for r in rows),'direct_description_positive_rows':sum(r['effective_direct_CE']['description']=='positive' for r in rows)},'gt_append_absence_checks':absence},open(O/'image_summary.json','w'),indent=2,sort_keys=True);open(O/'image_summary.json','a').write('\n')
json.dump({'schema':'training_set_completion.stage03_gt_append_absence_check.v1','target_version':str(V3path),'checks':absence},open(O/'gt_append_absence_check.json','w'),indent=2,sort_keys=True);open(O/'gt_append_absence_check.json','a').write('\n')
json.dump({'schema':'training_set_completion.stage03_proposal_decision_map.v1','selected_count':len(rows),'mapping':{r['proposal_id']:{'image_id':r['image_id'],'step':r['step'],'generated_order':r['generated_order'],'effective_owner_id':r['effective_owner_id'],'physical_status':r['physical_status'],'effective_extent':r['effective_extent'],'effective_class':r['effective_class'],'effective_direct_CE':r['effective_direct_CE'],'coverage_eligible_v3':r['coverage_eligible_v3']} for r in rows}},open(O/'proposal_decision_map.json','w'),indent=2,sort_keys=True);open(O/'proposal_decision_map.json','a').write('\n')
json.dump({'schema':'training_set_completion.stage03_source_manifest.v1','sources':source_manifest,'counts':{'source_entries':len(source_manifest),'evidence_entries':sum(x['kind']=='evidence' for x in source_manifest)},'source_hash_rule':'All source path/hash pairs are retained; canonical row evidence retains original review references.'},open(O/'source_manifest.json','w'),indent=2,sort_keys=True);open(O/'source_manifest.json','a').write('\n')
receipt={'schema':'training_set_completion.stage03_mask_preparation_receipt.v1','status':'candidate_ready','classification':'reviewed_prefix_projection_only_pending_root_training_use','artifact_root':str(O),'inputs':{'canonical_decisions':{'path':str(Dpath),'sha256':sha(Dpath)},'target_catalog_v3':{'path':str(V3path),'sha256':sha(V3path)},'v3_admissions':{'path':str(Apath),'sha256':sha(Apath)},'owner_plan':{'path':str(Ppath),'sha256':sha(Ppath)}},'selection':{'append_images':append_images,'step16_complete_rows':sum(r['step']==16 for r in rows),'step32_image323322_first6':sum(r['image_id']==323322 and r['step']==32 for r in rows),'step32_image210457_complete':sum(r['image_id']==210457 and r['step']==32 for r in rows)},'counts':{'selected_unique_proposals':len(rows),'selected_step16_rows':sum(r['step']==16 for r in rows),'selected_step32_rows':sum(r['step']==32 for r in rows),'v3_target_owner_denominator':228,'gt_append_absence_checks':len(absence),'gt_append_absence_all_pass':all(x['physically_absent_verified'] for x in absence),'admission_entries_total':len(admissions['entries']),'admission_entries_selected':sum(r['admission_applied'] is not None for r in rows),'root_overrides_selected':sum(r['root_override_applied'] is not None for r in rows)},'validation':{'proposal_id_uniqueness':'passed','packet_row_span_match':'passed','selected_prefix_span':'passed','target_v3_bins_against_owner_plan':'passed','step16_gt_physical_absence':'passed','no_positive_CE_for_unknown_false_repeat_or_wrong_unknown_extent':'passed','class_unknown_description_mask':'passed','clear_glass_219546_raw_extent_held_unknown':'passed'},'outputs':{'decisions_jsonl':str(O/'decisions.jsonl'),'proposal_decision_map':str(O/'proposal_decision_map.json'),'image_summary':str(O/'image_summary.json'),'gt_append_absence_check':str(O/'gt_append_absence_check.json'),'source_manifest':str(O/'source_manifest.json')},'limits':['This is a reviewed-prefix projection, not mask offsets, training data, or EOS acceptance.','Fresh suffix rows and forced/released rows remain outside this projection.','v3 admissions are applied only to their exact reviewed proposals; unselected admission entries remain in the source manifest.'],'command':'python3 '+str(O/'prepare.py')}
json.dump(receipt,open(O/'receipt.json','w'),indent=2,sort_keys=True);open(O/'receipt.json','a').write('\n')
print(json.dumps({'selected':len(rows),'step16':sum(r['step']==16 for r in rows),'step32':sum(r['step']==32 for r in rows),'absence':len(absence),'admission_selected':sum(r['admission_applied'] is not None for r in rows),'bbox_pos':sum(r['effective_direct_CE']['bbox']=='positive' for r in rows),'desc_pos':sum(r['effective_direct_CE']['description']=='positive' for r in rows)},sort_keys=True))
