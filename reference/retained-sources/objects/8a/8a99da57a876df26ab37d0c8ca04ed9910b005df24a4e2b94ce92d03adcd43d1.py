"""Normalize the snapshot of stage-01 single-image owner reviews."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Iterable

BASE=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum')
REVIEWS=BASE/'stage01-owner-reviews-v1'; PACKETS=BASE/'stage01-review-packets-v1'
SUPPORT=BASE/'stage01-support-seed-v1/candidate-support-ledger-v1.json'
ACQ=BASE/'stage01-acquisition-v1-retry1-config-batch2/rows.jsonl'
IMAGE_IDS=(25274,59571,99937,210457,219546,323322,351017,388795,417044,477415,528944)
OUT=Path(__file__).resolve().parent

def canon(x:Any)->bytes: return (json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def binding(p:Path)->dict[str,Any]:
 p=p.resolve(strict=True); return {'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size}
def write(p:Path,x:Any): p.write_bytes(canon(x))
def review_items(d:dict[str,Any])->list[dict[str,Any]]:
 out=[]
 def add(x:Any, key_hint=None):
  if not isinstance(x,dict): return
  if key_hint is not None and 'proposal_id' not in x: x=dict(x); x['proposal_id']=key_hint
  if x.get('proposal_id'): out.append(x)
 # Preserve each schema's native container shape while flattening only proposal records.
 if isinstance(d.get('decisions'),dict):
  for k,v in d['decisions'].items(): add(v,k)
 elif isinstance(d.get('decisions'),list):
  for x in d['decisions']: add(x)
 if isinstance(d.get('proposal_decisions'),list):
  for x in d['proposal_decisions']: add(x)
 if isinstance(d.get('proposals'),list):
  for x in d['proposals']: add(x)
 if isinstance(d.get('policies'),dict):
  for pol,payload in d['policies'].items():
   if isinstance(payload,dict) and isinstance(payload.get('proposal_reviews'),list):
    for x in payload['proposal_reviews']:
     y=dict(x); y.setdefault('policy',pol); add(y)
 if isinstance(d.get('policy_reviews'),list):
  for payload in d['policy_reviews']:
   if not isinstance(payload,dict) or not isinstance(payload.get('decisions'),list): continue
   for x in payload['decisions']:
    if isinstance(x,dict):
     y=dict(x); y.setdefault('policy',payload.get('policy')); y['_review_container']='policy_reviews'; add(y)
 direct_proposal_ids={x.get('proposal_id') for x in out if isinstance(x,dict) and x.get('proposal_id')}
 # Some completed reviews group identical proposals under a shared visual
 # decision. Expand each member to a row while retaining the complete group
 # as provenance on the flattened record.
 for key in ('proposal_decision_groups','identical_proposal_groups'):
  if isinstance(d.get(key),list):
   for group in d[key]:
    if not isinstance(group,dict): continue
    members=group.get('proposal_ids',group.get('members',[]))
    if not isinstance(members,list): continue
    for pid in members:
     # A policy decision is authoritative when the same proposal is also
     # mentioned by a summary group; the latter is provenance, not a second
     # semantic decision.
     if pid in direct_proposal_ids: continue
     x=dict(group); x.pop('proposal_ids',None); x.pop('members',None)
     x['proposal_id']=pid; x['review_group_key']=key; x['review_group']=group
     add(x)
 # Deduplicate only exact repeated containers; contradictory duplicate decisions remain a hold.
 seen=[]; result=[]
 for x in out:
  sig=canon(x)
  if sig not in seen: seen.append(sig); result.append(x)
 return result

def packet_proposals(packet:dict[str,Any])->dict[str,dict[str,Any]]:
 result={}
 for pol in packet.get('policies',[]):
  for p in pol.get('proposals',[]):
   x=dict(p); x['policy']=pol.get('policy',x.get('policy')); x['temperature']=pol.get('request',{}).get('temperature',x.get('temperature')); x['seed']=pol.get('request',{}).get('seed',x.get('seed')); x['policy_overlay_path']=pol.get('overlay_path'); result[x['proposal_id']]=x
 return result

def explicit_new_entries(d:dict[str,Any])->list[dict[str,Any]]:
 out=[]
 for key in ('newowners','new_owners','new_owner_catalog','stable_new_owners'):
  v=d.get(key)
  if isinstance(v,list):
   for x in v:
    if isinstance(x,dict) and x.get('owner_id') is not None:
     y=dict(x); y['_registry_field']=key; out.append(y)
 containers=[]
 if isinstance(d.get('decisions'),dict): containers.extend(d['decisions'].values())
 if isinstance(d.get('decisions'),list): containers.extend(d['decisions'])
 if isinstance(d.get('proposal_decisions'),list): containers.extend(d['proposal_decisions'])
 if isinstance(d.get('proposals'),list): containers.extend(d['proposals'])
 for x in containers:
  if isinstance(x,dict) and x.get('admission') in {'admit_candidate','admit_positive_support'} and x.get('owner_id') is not None:
   out.append({'owner_id':x['owner_id'],'proposal_id':x.get('proposal_id'),'_registry_field':'proposal_admission','raw_admission':x})
 return out

def main():
 support=json.loads(SUPPORT.read_text()); support_by={int(x['image_id']):x for x in support['images']}
 gt_ids={str(o['owner_id']) for im in support['images'] for o in im['gt_owners']}
 gt_ids_by_image={int(im['image_id']):{str(o['owner_id']) for o in im['gt_owners']} for im in support['images']}
 gt_atomic={str(o['owner_id']) for im in support['images'] for o in im['gt_owners'] if not o['is_crowd']}
 gt_crowd={str(o['owner_id']) for im in support['images'] for o in im['gt_owners'] if o['is_crowd']}
 prior=[]
 for im in support['images']:
  for x in im.get('reviewed_candidate_supports',[]):
   prior.append({'image_id':int(im['image_id']),**x})
 prior_ids={str(x.get('owner_id')) for x in prior}
 packet_by={i:PACKETS/f'image-{i:012d}/packet.json' for i in IMAGE_IDS}
 # One immutable snapshot of whatever review files exist now.
 review_paths={i:REVIEWS/f'image-{i:012d}/review.json' for i in IMAGE_IDS}
 owner_overrides_by_image={i:(json.loads((REVIEWS/f'image-{i:012d}/root-overrides.json').read_text()) if (REVIEWS/f'image-{i:012d}/root-overrides.json').is_file() else {}) for i in IMAGE_IDS}
 owner_alias_by_image={i:(owner_overrides_by_image[i].get('owner_aliases',{}) or {}) for i in IMAGE_IDS}
 present=[i for i,p in review_paths.items() if p.is_file()]; missing=[i for i in IMAGE_IDS if i not in present]
 all_rows=[]; per_image=[]; registry_entries=[]; orphan=[]; source_images=[]; route_summaries=[]; duplicate_review_ids=[]; contradictory_review_ids=[]
 for image_id in IMAGE_IDS:
  pp=packet_by[image_id]; packet=json.loads(pp.read_text()); pmap=packet_proposals(packet)
  rp=review_paths[image_id]; rd=json.loads(rp.read_text()) if rp.is_file() else None
  op=REVIEWS/f'image-{image_id:012d}/root-overrides.json'; od=json.loads(op.read_text()) if op.is_file() else None
  owner_aliases=(od or {}).get('owner_aliases',{})
  ritems=review_items(rd) if rd else []
  by_review={}; review_variants={}; dup=[]; contradictory=[]
  def decision_signature(x):
   return canon({k:x.get(k) for k in ('owner_id','physical_status','extent','class','direct_CE')})
  for x in ritems:
   pid=x['proposal_id']
   if pid in by_review:
    if decision_signature(x)==decision_signature(by_review[pid]): dup.append(pid)
    else: contradictory.append(pid)
    review_variants.setdefault(pid,[]).append(x)
   else:
    by_review[pid]=x; review_variants[pid]=[x]
  duplicate_review_ids.extend((image_id,pid) for pid in dup)
  contradictory_review_ids.extend((image_id,pid) for pid in sorted(set(contradictory)))
  # Schemas that summarize explicit new-owner catalogs or held groups still
  # carry proposal membership; materialize those memberships as review rows
  # without inventing decisions for other proposals.
  derived={}
  held_ids=(od or {}).get('held_new_owner_ids',{})
  for x in explicit_new_entries(rd or {}):
   members=list(x.get('proposal_ids') or ([x.get('canonical_proposal_id')] if x.get('canonical_proposal_id') else []))
   if not members:
    wanted=x.get('bbox_coord_bins_1000'); wanted_px=x.get('bbox_pixel_xyxy')
    members=[pid for pid,pp0 in pmap.items() if (wanted is not None and pp0.get('coord_bins_1000')==wanted) or (wanted_px is not None and pp0.get('bbox_pixel_xyxy')==wanted_px)]
   if not members: continue
   oid=str(x['owner_id']); held_reason=held_ids.get(oid)
   cls=x.get('category_name',x.get('category',x.get('class_policy')))
   cls_status=x.get('class_status',x.get('category_status','verified'))
   for pid in members:
    if not pid or pid in by_review: continue
    derived[pid]={'proposal_id':pid,'owner_id':None if held_reason else oid,'physical_status':'unknown' if held_reason else 'true_unique','extent':'unknown' if held_reason else 'reasonable','class':'unknown' if held_reason or cls_status not in {'verified','correct'} else 'verified','direct_CE':'mask' if held_reason or cls_status not in {'verified','correct'} else 'positive','evidence':x.get('evidence',x.get('evidence_crop_path')),'reason':held_reason or x.get('reason','Explicit reviewer new-owner catalog member.')}
  for group in (rd or {}).get('groups_not_admitted_as_atomic_owners',[]):
   if isinstance(group,dict):
    for pid in group.get('members',[]):
     if pid not in by_review and pid not in derived: derived[pid]={'proposal_id':pid,'owner_id':None,'physical_status':'unknown','extent':'unknown','class':'unknown','direct_CE':'mask','reason':group.get('reason','Group not admitted as atomic owner.')}
  for group in (rd or {}).get('identical_proposal_groups',[]):
   if not isinstance(group,dict): continue
   decision=str(group.get('decision',''))
   import re
   m=re.search(r'same existing GT owner ([A-Za-z0-9:_-]+)',decision)
   for pid in group.get('members',[]):
    if pid in by_review or pid in derived: continue
    derived[pid]={'proposal_id':pid,'owner_id':m.group(1) if m else None,'physical_status':'true_unique' if m else 'unknown','extent':'reasonable' if m else 'unknown','class':'verified' if m else 'unknown','direct_CE':'positive' if m else 'mask','reason':decision}
  by_review.update(derived)
  for pid in by_review:
   if pid not in pmap: orphan.append({'image_id':image_id,'proposal_id':pid,'raw_review_decision':by_review[pid]})
  for pid,pr in sorted(pmap.items(), key=lambda kv:(kv[1].get('policy',''),kv[1].get('generated_order',99999),kv[0])):
   rv=by_review.get(pid)
   row={
    'proposal_id':pid,'image_id':image_id,'request_id':pr.get('source_request_id'),'policy':pr.get('policy'),'temperature':pr.get('temperature'),'seed':pr.get('seed'),'generated_order':pr.get('generated_order'),
    'parser_status':pr.get('status'),'parser_drop_reason':pr.get('drop_reason'),'parser_drop_code':pr.get('drop_code'),'coord_bins_1000':pr.get('coord_bins_1000'),'bbox_pixel_xyxy':pr.get('bbox_pixel_xyxy'),'description_predicted':pr.get('description'),'nearest_gt_owner_id':pr.get('nearest_gt_owner_id'),'nearest_gt_iou':pr.get('nearest_gt_iou'),'review_flags_packet':pr.get('review_flags',[]),
    'reviewed':bool(rv),'reviewed_owner_id':rv.get('owner_id') if rv else None,'physical_status':rv.get('physical_status') if rv else 'unknown','extent':rv.get('extent') if rv else 'unknown','class':rv.get('class') if rv else 'unknown','direct_CE':rv.get('direct_CE') if rv else 'mask','review_reason':rv.get('reason',rv.get('note',rv.get('review_note'))) if rv else 'No review decision present in the immutable snapshot; unknown/unreviewed.',
    'review_evidence':rv.get('evidence',rv.get('visual_evidence',rv.get('evidence_paths'))) if rv else [],
    'source':{'packet_path':str(pp.resolve()),'packet_sha256':sha(pp),'acquisition_line_number_1_based':pr.get('source_line_number_1_based'),'acquisition_line_sha256':pr.get('source_line_sha256'),'raw_span_sha256':pr.get('raw_span_sha256'),'image_path':packet['image']['path'],'image_sha256':packet['image']['sha256']},
   'raw_review_decision':rv,
   'review_decision_variants':review_variants.get(pid,[]) if len(review_variants.get(pid,[]))>1 else [],
   'review_decision_contradiction':pid in contradictory,
   }
   # Keep the actual raw parser-invalid status independent from a reviewer's semantic invalid label.
   row['raw_geometry_invalid']=row['parser_drop_reason']=='geometry_invalid'
   row['root_owner_id']=owner_aliases.get(row['reviewed_owner_id'],row['reviewed_owner_id']) if row['reviewed_owner_id'] is not None else None
   # ExistingGT:<id> is a mechanical prefix used by some review files. Strip
   # it only when the bare ID is a GT owner on this same image.
   if isinstance(row['root_owner_id'],str) and row['root_owner_id'].startswith('existingGT:'):
    bare=row['root_owner_id'].split(':',1)[1]
    if bare in gt_ids_by_image[image_id]: row['root_owner_id']=bare; row['root_existing_gt_prefix_normalized']=True
   row['root_override_applied']=row['reviewed_owner_id'] is not None and row['root_owner_id']!=row['reviewed_owner_id']
   row['effective_extent']=row['extent']; row['effective_class']=row['class']; row['effective_direct_CE']=row['direct_CE']
   if isinstance(row['direct_CE'],dict): row['effective_description_CE']=row['direct_CE'].get('description','mask'); row['effective_bbox_CE']=row['direct_CE'].get('bbox','mask')
   else: row['effective_description_CE']=row['direct_CE']; row['effective_bbox_CE']=row['direct_CE']
   owner_override=(od or {}).get('owner_decision_overrides',{}).get(row['root_owner_id'],{})
   if owner_override:
    row['effective_extent']=owner_override.get('extent',row['effective_extent']); row['effective_class']=owner_override.get('class',row['effective_class']); row['effective_direct_CE']=owner_override.get('direct_CE',row['effective_direct_CE']); row['root_review_override']=owner_override
    if 'class' in owner_override and owner_override['class'] not in {'verified','correct'}: row['effective_description_CE']='mask'
   held_reason=(od or {}).get('held_new_owner_ids',{}).get(row['root_owner_id']) if isinstance((od or {}).get('held_new_owner_ids',{}),dict) else None
   if held_reason is not None:
    row['root_owner_held']=True; row['root_owner_hold_reason']=held_reason; row['effective_direct_CE']='mask'; row['effective_description_CE']='mask'; row['effective_bbox_CE']='mask'
   all_rows.append(row)
  for x in explicit_new_entries(rd or {}):
   if str(x['owner_id']) not in prior_ids:
    registry_entries.append({'image_id':image_id,'owner_id':str(x['owner_id']),'explicit_registry_field':x['_registry_field'],'raw_entry':x})
  source_images.append({'image_id':image_id,'packet':binding(pp),'review':binding(rp) if rp.is_file() else None,'root_override':binding(op) if op.is_file() else None,'review_status':'present' if rp.is_file() else 'missing','review_schema':rd.get('schema') if rd else None,'review_status_value':rd.get('status') if rd else None,'source_image':packet['image']})
  if rd:
   for key in ('selected_route','best_coherent_existing_route','route_assessment','eos_supervision','route_recommendation','route_eos_admissibility'):
    if key in rd: route_summaries.append({'image_id':image_id,'field':key,'value':rd[key]})
  per_image.append({'image_id':image_id,'packet_proposal_count':len(pmap),'review_decision_count':len(ritems),'reviewed_proposal_count':len(set(by_review)&set(pmap)),'unreviewed_proposal_count':len(set(pmap)-set(by_review)),'review_status':'present' if rd else 'missing','review_schema':rd.get('schema') if rd else None,'proposal_rows_path':str((OUT/'proposal-rows'/f'image-{image_id:012d}.jsonl').resolve())})
 # Recompute same-owner repeats within each actual policy after root alias rulings.
 for image_id in IMAGE_IDS:
  groups={}
  for row in (r for r in all_rows if r['image_id']==image_id and r['root_owner_id'] is not None):
   if row['physical_status'] not in {'true_unique','repeat'}: continue
   groups.setdefault((row['policy'],row['root_owner_id']),[]).append(row)
  for members in groups.values():
   for row in members[1:]: row['root_repeat_after_alias']=True
   if members: members[0]['root_repeat_after_alias']=False
 # Write one JSONL per image; rows preserve full review object for expansion.
 (OUT/'proposal-rows').mkdir(exist_ok=True)
 for image_id in IMAGE_IDS:
  rows=[x for x in all_rows if x['image_id']==image_id]
  path=OUT/'proposal-rows'/f'image-{image_id:012d}.jsonl'
  path.write_bytes(b''.join(canon(x) for x in rows))
 # Explicit new registry entries only. Merge repeated declarations by exact owner ID without resolving aliases.
 registry=[]; seen={}
 for x in registry_entries:
  oid=owner_alias_by_image.get(x['image_id'],{}).get(x['owner_id'],x['owner_id'])
  x=dict(x); x['owner_id_before_root_override']=x['owner_id']; x['owner_id']=oid
  bucket=seen.setdefault(oid,{'owner_id':oid,'declarations':[]}); bucket['declarations'].append(x)
 for oid,x in sorted(seen.items()):
  members=[r for r in all_rows if r['root_owner_id']==oid]
  physical={r['physical_status'] for r in members}; extent={r['effective_extent'] for r in members}; classes={r['effective_class'] for r in members}
  # direct_CE may be a structured bbox/description object in group reviews;
  # canonical strings keep the registry deterministic without discarding it.
  ce={json.dumps(r['effective_direct_CE'],sort_keys=True,separators=(',',':'),ensure_ascii=False) for r in members}
  row_eligible=[r for r in members if r['physical_status'] in {'true_unique','repeat'} and r['effective_extent']=='reasonable' and r['parser_status']=='parsed_valid' and not r.get('root_owner_held')]
  row_variation_fields=[k for k,v in (('physical_status',physical),('extent',extent),('class',classes)) if len(v)>1]
  same_proposal_conflict=any(r.get('review_decision_contradiction') for r in members)
  x.update({'member_proposal_ids':[r['proposal_id'] for r in members],'physical_statuses':sorted(physical),'extents':sorted(extent),'classes':sorted(classes),'effective_description_CE_values':sorted({str(r['effective_description_CE']) for r in members}),'direct_CE_values':[json.loads(v) for v in sorted(ce)],'coverage_row_evidence_count':len(row_eligible),'coverage_eligible_candidate':bool(row_eligible),'row_variation':bool(row_variation_fields),'row_variation_fields':row_variation_fields,'conflict_hold':same_proposal_conflict})
  # Held catalog entries remain explicit evidence, but are excluded from the
  # admitted target-owner denominator. Root acceptance is computed from the
  # source catalogs and held-owner override records, never from manual totals.
  x['root_held']=any(
   str(dec.get('owner_id_before_root_override',dec.get('owner_id'))) in
   ((owner_overrides_by_image.get(int(dec['image_id']),{}).get('held_new_owner_ids',{}) or {}))
   for dec in x['declarations'])
  x['admitted_target_candidate']=not x['root_held']
  registry.append(x)
 # Preserve packet/support/acquisition provenance and all GT owners by role.
 admitted_new=[x for x in registry if x['admitted_target_candidate']]
 held_new=[x for x in registry if x['root_held']]
 target={'gt_atomic':[{'image_id':int(im['image_id']),'owner_id':str(o['owner_id']),'category_name':o['category_name'],'bbox_coord_bins_1000':o['bbox_coord_canvas']['coord_bins_1000'],'is_crowd':False} for im in support['images'] for o in im['gt_owners'] if not o['is_crowd']], 'gt_crowd':[{'image_id':int(im['image_id']),'owner_id':str(o['owner_id']),'category_name':o['category_name'],'bbox_coord_bins_1000':o['bbox_coord_canvas']['coord_bins_1000'],'is_crowd':True} for im in support['images'] for o in im['gt_owners'] if o['is_crowd']], 'prior_reviewed_supports':prior,'explicit_new_owner_registry':registry,'admitted_new_owner_registry':admitted_new,'held_explicit_new_owner_candidates':held_new,'computed_counts':{'gt_atomic':len(gt_atomic),'gt_crowd':len(gt_crowd),'prior_reviewed_supports':len(prior),'prior_non_gt':sum(str(x.get('owner_id')) not in gt_ids for x in prior),'new_explicit_declared':len(registry),'new_admitted_target_candidates':len(admitted_new),'new_coverage_eligible_candidates':sum(x['coverage_eligible_candidate'] for x in admitted_new),'atomic_candidate_total':len(gt_atomic)+sum(str(x.get('owner_id')) not in gt_ids for x in prior)+len(admitted_new),'atomic_coverage_eligible_total':len(gt_atomic)+sum(str(x.get('owner_id')) not in gt_ids for x in prior)+sum(x['coverage_eligible_candidate'] for x in admitted_new)},'rule':'Coverage candidate requires owner_id, physical_status true_unique/repeat, and extent reasonable; class errors remain separate. Admitted target-owner count includes explicit root-accepted owners even when a reviewed emitted box is wrong extent; such rows retain coverage_eligible_candidate=false.'}
 source={'schema':'training_set_completion.stage01_review_extraction.source_manifest.v2','snapshot_policy':'Single snapshot of review.json files present at extraction start; missing files are not polled or inferred.','acquisition':binding(ACQ),'support_ledger':binding(SUPPORT),'packets_and_reviews':source_images,'expected_image_ids':list(IMAGE_IDS),'completed_review_image_ids':present,'missing_review_image_ids':missing,'partial_review_image_ids':[i for i in present if next(x for x in per_image if x['image_id']==i)['unreviewed_proposal_count']>0]}
 # Diagnostics distinguish raw parser invalid from review semantic invalid labels.
 diag={'schema':'training_set_completion.stage01_review_extraction.diagnostic_counts.v2','expected_images':len(IMAGE_IDS),'review_files_present':len(present),'review_files_missing':len(missing),'present_image_ids':present,'missing_image_ids':missing,'packet_proposal_rows_all_images':len(all_rows),'reviewed_proposal_rows':sum(r['reviewed'] for r in all_rows),'unreviewed_proposal_rows':sum(not r['reviewed'] for r in all_rows),'raw_geometry_invalid_rows':sum(r['raw_geometry_invalid'] for r in all_rows),'review_semantic_invalid_rows':sum(r['physical_status']=='invalid' for r in all_rows),'review_confirmed_false_rows':sum(r['physical_status']=='false' for r in all_rows),'review_unknown_rows':sum(r['physical_status']=='unknown' for r in all_rows),'root_recomputed_repeat_rows':sum(r.get('root_repeat_after_alias',False) for r in all_rows),'explicit_new_owner_registry_count':len(registry),'admitted_new_owner_registry_count':len(admitted_new),'held_explicit_new_owner_count':len(held_new),'gt_atomic_count':len(gt_atomic),'gt_crowd_count':len(gt_crowd),'prior_reviewed_support_count':len(prior),'orphan_review_decision_count':len(orphan),'duplicate_review_decision_ids':sorted(set(f'{i}:{p}' for i,p in duplicate_review_ids)),'same_proposal_contradictory_review_ids':sorted(set(f'{i}:{p}' for i,p in contradictory_review_ids)),'alias_or_unresolved_owner_id_count':0,'alias_or_unresolved_owner_ids':[],'contradiction_hold_count':sum(x['conflict_hold'] for x in registry),'row_variation_owner_count':sum(x['row_variation'] for x in registry),'row_variation_owner_ids':[x['owner_id'] for x in registry if x['row_variation']],'coverage_row_evidence_total':sum(x['coverage_row_evidence_count'] for x in registry),'image_breakdown':per_image,'coverage_rule':'Coverage requires one same owner row with owner_id, physical_status in {true_unique,repeat}, effective extent reasonable, and parser_status parsed_valid. No cross-row set combination; no natural EOS/pass inferred here.','raw_invalid_vs_review_invalid_note':'raw_geometry_invalid_rows comes only from packet parser status; reviewer semantic invalid/false labels are separate and never overwrite it. Root-held 59571 owners retain identity evidence but all effective CE is masked and they are excluded from the admitted denominator.'}
 # unresolved aliases and non-GT owner labels are held, never automatically mapped.
 aliases=[]
 for r in all_rows:
  oid=r['root_owner_id']
  if oid and oid not in gt_ids and oid not in {x['owner_id'] for x in registry} and oid not in {str(x.get('owner_id')) for x in prior}:
   aliases.append(oid)
 aliases=sorted(set(aliases)); diag['alias_or_unresolved_owner_id_count']=len(aliases);diag['alias_or_unresolved_owner_ids']=aliases
 manifest={'schema':'training_set_completion.stage01_review_extraction.v2','status':'candidate_extraction_pending_root_visual_acceptance','source_manifest_path':str((OUT/'source-manifest.json').resolve()),'proposal_rows_dir':str((OUT/'proposal-rows').resolve()),'target_candidate_registry_path':str((OUT/'target-candidate-registry.json').resolve()),'diagnostic_counts_path':str((OUT/'diagnostic-counts.json').resolve()),'completed_image_ids':present,'missing_image_ids':missing,'partial_image_ids':source['partial_review_image_ids'],'root_rulings_needed':['210457 is partial: unspecified raw proposals remain unknown/unreviewed.','Owner aliases/prefixes and conflicting declarations remain holds.','Natural EOS/pass and after-masking routes are not accepted by extraction.'],'orphan_review_decisions':orphan,'route_summaries':route_summaries}
 write(OUT/'target-candidate-registry.json',target);write(OUT/'source-manifest.json',source);write(OUT/'diagnostic-counts.json',diag);write(OUT/'extraction-receipt.json',manifest)
 print(json.dumps({'status':manifest['status'],'completed':present,'missing':missing,'partial':source['partial_review_image_ids'],'rows':len(all_rows),'raw_geometry_invalid':diag['raw_geometry_invalid_rows'],'review_invalid':diag['review_semantic_invalid_rows'],'new_registry':len(registry),'orphans':len(orphan),'aliases':len(aliases)},indent=2))
if __name__=='__main__': main()
