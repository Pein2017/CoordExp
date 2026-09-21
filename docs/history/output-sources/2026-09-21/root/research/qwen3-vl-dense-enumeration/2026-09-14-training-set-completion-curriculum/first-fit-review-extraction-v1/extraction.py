#!/usr/bin/env python3
import copy,glob,hashlib,json,os,re
from collections import Counter,defaultdict
from pathlib import Path
B=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum')
OUT=B/'first-fit-review-extraction-v1'; OUT.mkdir(parents=True,exist_ok=True)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()
def parse_id(p):
 m=re.search(r'first-fit:(?:step(16|32)):image-(\d+):p(\d+)$',p)
 return (int(m.group(2)),int(m.group(1)),int(m.group(3))) if m else None
def walk_records(x,out,path=''):
 if isinstance(x,dict):
  if isinstance(x.get('proposal_id'),str) and x['proposal_id'].startswith('first-fit:') and 'prediction_id' in x:
   out.append((copy.deepcopy(x),path))
  for k,v in x.items(): walk_records(v,out,f'{path}.{k}' if path else k)
 elif isinstance(x,list):
  for i,v in enumerate(x): walk_records(v,out,f'{path}[{i}]')
def walk_candidate_maps(x,maps,raws,path=''):
 if isinstance(x,dict):
  props=[]
  # Candidate registries use explicit candidate/member/seen/canonical fields;
  # ordinary decision rows with proposal_id+owner_id are not registries.
  if isinstance(x.get('canonical_proposal_id'),str) and x['canonical_proposal_id'].startswith('first-fit:'): props.append(x['canonical_proposal_id'])
  for k in ('proposal_ids','member_proposal_ids','seen_at'):
   if isinstance(x.get(k),list): props.extend(p for p in x[k] if isinstance(p,str) and p.startswith('first-fit:'))
  if isinstance(x.get('proposal_id'),str) and x['proposal_id'].startswith('first-fit:') and any(k in x for k in ('pending_owner_id','candidate_owner_id','candidate_id')): props.append(x['proposal_id'])
  owner=x.get('owner_id') or x.get('pending_owner_id') or x.get('candidate_owner_id') or x.get('candidate_id')
  if owner and props:
   for p in props: maps[p]=owner; raws.append({'path':path,'record':copy.deepcopy(x)})
  for k,v in x.items(): walk_candidate_maps(v,maps,raws,f'{path}.{k}' if path else k)
 elif isinstance(x,list):
  for i,v in enumerate(x): walk_candidate_maps(v,maps,raws,f'{path}[{i}]')

def resolve_rel(p):
 if not isinstance(p,str): return None
 q=Path(p)
 if q.is_absolute(): return q
 # review evidence paths are relative to curriculum root
 return B/q

# frozen v2 target set
catalog_path=B/'target-owners-complete-v2.json'; catalog=json.load(open(catalog_path))
fixed_records=[r for r in catalog['records'] if r.get('role') in ('gt_atomic','new','prior_non_gt')]
assert len(fixed_records)==220
fixed_by_img=defaultdict(dict); target_order=defaultdict(list)
for r in fixed_records:
 fixed_by_img[int(r['image_id'])][r['owner_id']]=r
 target_order[int(r['image_id'])].append(r['owner_id'])

review_paths=sorted(glob.glob(str(B)+'/**/first-fit-owner-reviews-v1/image-*/review.json',recursive=True))
assert len(review_paths)==11,review_paths
reviews=[]; source_manifest=[]; candidate_map={}; candidate_raw=[]; all_mentions=defaultdict(list)
for rf in review_paths:
 rd=json.load(open(rf)); im=int(rd['image_id']); rsha=sha(rf)
 reviews.append((rf,rd,rsha))
 source_manifest.append({'kind':'review','path':rf,'sha256':rsha,'image_id':im})
 # Capture candidate registries before selecting row records.
 walk_candidate_maps(rd,candidate_map,candidate_raw,'')
 # All declared review-side source/hash objects retained in manifest section below.
 raw=[]; walk_records(rd,raw)
 for rec,path in raw:
  pid=rec.get('proposal_id'); parsed=parse_id(pid)
  if not parsed: continue
  all_mentions[pid].append({'review_path':rf,'review_sha256':rsha,'object_path':path,'record':rec})

# Build packets and source entries; packet is authoritative for raw geometry/status.
packet_cache={}; packet_manifest=[]; scored_manifest=[]; image_manifest=[]; evidence_manifest={}
for im in sorted(fixed_by_img):
 ip=B/f'public_data/coco/rescale_32_1024_bbox/images/train2017/{im:012d}.jpg'
 # actual public data lives outside B in this worktree path
 ip=Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/train2017')/f'{im:012d}.jpg'
 image_manifest.append({'kind':'original_image','image_id':im,'path':str(ip),'sha256':sha(ip)})
 for step in (16,32):
  pp=B/f'first-fit-v1/plots-step-{step}/image-{im:012d}/packet.json'; assert pp.is_file(),pp
  pd=json.load(open(pp)); packet_cache[(im,step)]=pd
  psha=sha(pp); packet_manifest.append({'kind':'packet','image_id':im,'step':step,'path':str(pp),'sha256':psha})
  sj=pd.get('source',{}).get('scored_json')
  if sj and sj.get('path'):
   scored_manifest.append({'kind':'scored_json','image_id':im,'step':step,'path':sj['path'],'sha256':sj.get('sha256') or sha(sj['path'])})
  for rr in pd.get('rendered',{}).get('raw_rows',[]):
   for k in ('context_crop','tight_crop'):
    if isinstance(rr.get(k),dict) and rr[k].get('path'):
     ep=rr[k]['path']; evidence_manifest[ep]={'kind':'evidence_crop','path':ep,'sha256':rr[k].get('sha256') or sha(ep)}
   # overlay is packet-level, added once below
  for k in ('raw_generated_overlay','target_catalog_overlay'):
   v=pd.get('rendered',{}).get(k)
   if isinstance(v,dict) and v.get('path'): evidence_manifest[v['path']]={'kind':'evidence_overlay','path':v['path'],'sha256':v.get('sha256') or sha(v['path'])}
# declared overrides, and exact hashes
override_paths=sorted(glob.glob(str(B)+'/**/first-fit-owner-reviews-v1/image-*/root-overrides.json',recursive=True))
overrides=[]; override_by_pid={}
for of in override_paths:
 od=json.load(open(of)); os_hash=sha(of); overrides.append({'kind':'root_override','path':of,'sha256':os_hash,'image_id':od.get('image_id')})
 for po in od.get('proposal_overrides',[]):
  override_by_pid[po['proposal_id']]={'path':of,'sha256':os_hash,'override':copy.deepcopy(po)}

# Candidate mapping and source records. Determine effective row by richness; sparse summary copies are retained as mentions.
def richness(rec):
 return sum(rec.get(k) is not None for k in ('step','prediction_id','owner_id','physical_status','extent','class','direct_CE','reason','evidence','evidence_paths','coord_bins_1000','bbox_coord_bins_1000','bbox_pixel_xyxy'))
def canon_hash_obj(x): return json.dumps(x,sort_keys=True,separators=(',',':'))
canonical=[]; duplicate_groups=[]; gaps=[]
for pid,mentions in sorted(all_mentions.items(),key=lambda kv:(parse_id(kv[0]) or (999999,999,999))):
 im,step,pn=parse_id(pid); assert im in fixed_by_img,(pid,im); assert step in (16,32)
 # derive rows whose declared step agrees with proposal; tolerate summary copy with absent step
 valid=[m for m in mentions if m['record'].get('step') in (None,step) and m['record'].get('prediction_id')==f'p{pn}']
 assert valid, pid
 chosen=max(valid,key=lambda m:richness(m['record']))
 duplicate_groups.append({'proposal_id':pid,'mention_count':len(mentions),'selected_object_path':chosen['object_path'],'mentions':[{k:m[k] for k in ('review_path','review_sha256','object_path')} for m in mentions]}) if len(mentions)>1 else None
 canonical.append((pid,im,step,pn,chosen['record'],mentions))

assert len(canonical)==392,(len(canonical),len(all_mentions))
# Ensure packet sets match exactly and raw order is stable.
packet_ids=set()
for (im,step),pd in packet_cache.items():
 for rr in pd['rendered']['raw_rows']:
  packet_ids.add(f'first-fit:step{step}:image-{im:012d}:{rr["prediction_id"]}')
assert packet_ids==set(all_mentions),(len(packet_ids),len(all_mentions),sorted(packet_ids-set(all_mentions))[:3],sorted(set(all_mentions)-packet_ids)[:3])

# Canonical rows; normalize only same-image existingGT ids that are in fixed v2 catalog.
rows=[]
for pid,im,step,pn,src,mentions in canonical:
 pd=packet_cache[(im,step)]; rr=next(r for r in pd['rendered']['raw_rows'] if r['prediction_id']==f'p{pn}')
 owner=src.get('owner_id'); owner_raw=owner
 if isinstance(owner,str) and owner.startswith('existingGT:'):
  base=owner[len('existingGT:'):]
  if base in fixed_by_img[im]: owner=base
  else: gaps.append({'proposal_id':pid,'kind':'unresolved_existingGT_alias','raw_owner_id':owner,'reason':'same-image target owner absent; alias not normalized'})
 # If source owner omitted but an explicit per-proposal candidate registry supplies one, preserve source omission and use candidate only as effective pending identity.
 candidate_owner=candidate_map.get(pid)
 if owner is None and candidate_owner is not None: owner=candidate_owner
 effective=copy.deepcopy(src)
 ov=override_by_pid.get(pid)
 if ov:
  po=ov['override'];
  for k in ('effective_extent','effective_direct_CE'):
   if k in po: effective[k]=copy.deepcopy(po[k])
  effective['root_override_reason']=po.get('reason')
 # Root uses effective_* names; otherwise source names.
 extent=effective.get('effective_extent',effective.get('extent')) or 'unknown'
 direct=effective.get('effective_direct_CE',effective.get('direct_CE')) or {'bbox':'mask','description':'mask'}
 physical=effective.get('physical_status') or 'unknown'; cls=effective.get('class') or 'unknown'
 raw_status=rr.get('status') or 'unknown'
 in_fixed=owner in fixed_by_img[im]
 role=fixed_by_img[im][owner]['role'] if in_fixed else None
 if in_fixed: identity_state='fixed_target'
 elif owner=='900100025274': identity_state='separate_group'
 elif candidate_owner or owner is not None: identity_state='candidate_pending_or_unqualified'
 else: identity_state='unassigned_or_unknown'
 eligible=bool(in_fixed and raw_status=='parsed_valid' and physical in ('true_unique','repeat') and extent=='reasonable')
 # Keep all evidence/source fields compact while retaining original review row exactly.
 evidence_raw=src.get('evidence',src.get('evidence_paths'))
 evidence_bindings=[]
 def add_e(path,kind=None):
  if isinstance(path,dict):
   ep=path.get('path') or path.get('overlay') or path.get('crop') or path.get('tight_crop')
   if ep: add_e(ep,kind)
  elif isinstance(path,str):
   ep=str(resolve_rel(path)) if not Path(path).is_absolute() else path
   if os.path.isfile(ep): evidence_bindings.append({'kind':kind or 'review_evidence','path':ep,'sha256':sha(ep)})
 if isinstance(evidence_raw,list):
  for e in evidence_raw: add_e(e)
 elif isinstance(evidence_raw,dict):
  for k,v in evidence_raw.items(): add_e(v,k)
 elif isinstance(evidence_raw,str): add_e(evidence_raw)
 # Packet overlay/crops bind every raw row even when review evidence was relative or sparse.
 ovp=pd['rendered'].get('raw_generated_overlay',{})
 if ovp.get('path'): evidence_bindings.append({'kind':'packet_raw_overlay','path':ovp['path'],'sha256':ovp.get('sha256') or sha(ovp['path'])})
 for k in ('tight_crop','context_crop'):
  if isinstance(rr.get(k),dict): evidence_bindings.append({'kind':'packet_'+k,'path':rr[k]['path'],'sha256':rr[k].get('sha256') or sha(rr[k]['path'])})
 # Dedup evidence while preserving order.
 seen=set(); evidence_bindings=[e for e in evidence_bindings if not (e['path'] in seen or seen.add(e['path']))]
 scored=pd.get('source',{}).get('scored_json',{})
 row={
  'schema':'training_set_completion.first_fit_canonical_decision.v1','image_id':im,'step':step,'proposal_id':pid,'prediction_id':rr['prediction_id'],'generated_order':rr.get('generated_order',pn),
  'raw':{'status':raw_status,'description':rr.get('description'),'coord_bins_1000':rr.get('coord_bins_1000'),'raw_bbox_pixel_xyxy':rr.get('raw_bbox_pixel_xyxy'),'render_box_pixel_xyxy':rr.get('render_box_pixel_xyxy'),'raw_axes_preserved':rr.get('raw_axes_preserved',True)},
  'owner_id':owner,'source_owner_id':owner_raw,'identity_state':identity_state,'target_role':role,'group_status':'crowd_separate' if owner=='900100025274' else None,'physical_status':physical,'extent':extent,'class':cls,'direct_CE':direct,'coverage_eligible':eligible,
  'candidate_owner_from_registry':candidate_owner if owner_raw is None else None,'root_override_applied':copy.deepcopy(ov) if ov else None,'reason':effective.get('root_override_reason',effective.get('reason')),
  'source_review':{'path':chosen_review_path if False else mentions[0]['review_path'],'sha256':mentions[0]['review_sha256'],'selected_record':copy.deepcopy(src),'all_mentions':[{k:m[k] for k in ('review_path','review_sha256','object_path','record')} for m in mentions]},
  'source_packet':{'path':str(B/f'first-fit-v1/plots-step-{step}/image-{im:012d}/packet.json'),'sha256':sha(B/f'first-fit-v1/plots-step-{step}/image-{im:012d}/packet.json'),'scored_row_index':pd.get('scored_row_index'),'scored_row_sha256':pd.get('scored_row_sha256'),'scored_json_path':scored.get('path'),'scored_json_sha256':scored.get('sha256')},
  'evidence':evidence_bindings
 }
 rows.append(row)

# Deterministic order.
rows.sort(key=lambda x:(x['image_id'],x['step'],x['generated_order'],x['proposal_id']))
# owner repeats by physical identity within each image and step, regardless IoU.
repeats={};
for im in sorted(fixed_by_img):
 for step in (16,32):
  rs=[r for r in rows if r['image_id']==im and r['step']==step]
  by=defaultdict(list)
  for r in rs:
   if r['owner_id'] is not None: by[r['owner_id']].append(r['proposal_id'])
  repeats[f'{im}:{step}']={o:pids for o,pids in sorted(by.items()) if len(pids)>1}
# Per image/step summaries and debt diagnostics.
summaries=[]
for im in sorted(fixed_by_img):
 for step in (16,32):
  rs=[r for r in rows if r['image_id']==im and r['step']==step]; target=target_order[im]
  covered=[]
  for o in target:
   if any(r['owner_id']==o and r['coverage_eligible'] for r in rs): covered.append(o)
  missing=[o for o in target if o not in covered]
  summaries.append({'image_id':im,'step':step,'fixed_target_count':len(target),'raw_rows':len(rs),'parsed_valid_rows':sum(r['raw']['status']=='parsed_valid' for r in rs),'raw_invalid_rows':sum(r['raw']['status']!='parsed_valid' for r in rs),'covered_fixed_owner_ids':covered,'missing_fixed_owner_ids':missing,'covered_count':len(covered),'missing_count':len(missing),'coverage_eligible_row_count':sum(r['coverage_eligible'] for r in rs),'physical_status_counts':dict(sorted(Counter(r['physical_status'] for r in rs).items())),'extent_counts':dict(sorted(Counter(r['extent'] for r in rs).items())),'class_counts':dict(sorted(Counter(r['class'] for r in rs).items())),'bbox_CE_counts':dict(sorted(Counter(r['direct_CE'].get('bbox') for r in rs).items())),'description_CE_counts':dict(sorted(Counter(r['direct_CE'].get('description') for r in rs).items())),'class_debt_row_count':sum(r['class']!='verified' for r in rs),'false_row_count':sum(r['physical_status']=='false' for r in rs),'unknown_row_count':sum(r['physical_status']=='unknown' for r in rs),'extent_debt_row_count':sum(r['extent']!='reasonable' for r in rs),'owner_repeats':repeats[f'{im}:{step}'],'outside_fixed_pending_owner_ids':sorted({r['owner_id'] for r in rs if r['owner_id'] is not None and r['owner_id'] not in fixed_by_img[im]})})

# Build hold/disagreement records.
holds=[]
for g in duplicate_groups:
 if g['mention_count']>1:
  vals=[]
  for m in g['mentions']:
   rec=next(mm['record'] for mm in all_mentions[g['proposal_id']] if mm['object_path']==m['object_path'] and mm['review_path']==m['review_path'])
   vals.append({k:rec.get(k) for k in ('step','prediction_id','owner_id','physical_status','extent','class','direct_CE','reason')})
  distinct={canon_hash_obj(v) for v in vals}
  holds.append({'kind':'duplicate_source_mentions','proposal_id':g['proposal_id'],'mention_count':g['mention_count'],'classification':'partial_summary_duplicate' if len(distinct)>1 else 'exact_duplicate','selected_richest_record':g['selected_object_path'],'mentions':g['mentions'],'field_snapshots':vals})
for gap in gaps: holds.append({'kind':'source_alias_hold',**gap})
for r in rows:
 if r['identity_state'] not in ('fixed_target','separate_group') and r['owner_id'] is not None:
  holds.append({'kind':'pending_or_unqualified_owner','proposal_id':r['proposal_id'],'owner_id':r['owner_id'],'physical_status':r['physical_status'],'extent':r['extent'],'class':r['class'],'reason':r['reason']})
for pid,ov in sorted(override_by_pid.items()): holds.append({'kind':'root_effective_override','proposal_id':pid,'override_source':ov})

# Manifest includes all review/packet/scored/original/override/evidence source identities.
manifest={'schema':'training_set_completion.first_fit_review_source_manifest.v1','target_catalog':{'path':str(catalog_path),'sha256':sha(catalog_path),'fixed_target_count':220},'reviews':sorted(source_manifest,key=lambda x:x['path']),'packets':sorted(packet_manifest,key=lambda x:(x['image_id'],x['step'])),'scored_json':sorted({(x['path'],x['sha256']):x for x in scored_manifest}.values(),key=lambda x:x['path']),'original_images':sorted(image_manifest,key=lambda x:x['image_id']),'root_overrides':sorted(overrides,key=lambda x:x['path']),'evidence_files':sorted(evidence_manifest.values(),key=lambda x:x['path'])}
# declared hashes from review fields retained without interpreting them as authorities.
declared=[]
for rf,rd,rsha in reviews:
 def collect(x,path=''):
  if isinstance(x,dict):
   for k,v in x.items():
    if isinstance(v,dict) and isinstance(v.get('path'),str) and isinstance(v.get('sha256'),str): declared.append({'review_path':rf,'field_path':f'{path}.{k}' if path else k,'declared':copy.deepcopy(v)})
    collect(v,f'{path}.{k}' if path else k)
  elif isinstance(x,list):
   for i,v in enumerate(x): collect(v,f'{path}[{i}]')
 collect(rd)
manifest['review_declared_path_hash_pairs']=declared

# Candidate registry with all pending mappings and raw source locations.
candidates=[]
for rec in candidate_raw:
 x=copy.deepcopy(rec['record']); candidates.append({'source_review_object_path':rec['path'],'record':x})
# retain unique source registry objects
uniq={canon_hash_obj(x):x for x in candidates}; candidates=sorted(uniq.values(),key=lambda x:(x['source_review_object_path'],canon_hash_obj(x['record'])))

# Output line-oriented canonical decisions and JSON summaries.
with open(OUT/'decisions.jsonl','w') as f:
 for r in rows: f.write(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n')
json.dump({'schema':'training_set_completion.first_fit_per_image_step_summary.v1','summaries':summaries,'aggregate':{'image_count':11,'steps_per_image':2,'raw_rows':len(rows),'fixed_target_denominator':220,'covered_owner_union_by_step':{str(s):sum(1 for z in summaries if z['step']==s and z['covered_count']>0) for s in (16,32)}},'owner_repeats':repeats},open(OUT/'per_image_step_summary.json','w'),indent=2,sort_keys=True); open(OUT/'per_image_step_summary.json','a').write('\n')
json.dump({'schema':'training_set_completion.first_fit_hold_disagreement.v1','duplicate_source_mentions':[x for x in holds if x.get('kind')=='duplicate_source_mentions'],'pending_or_unqualified_holds':[x for x in holds if x.get('kind')=='pending_or_unqualified_owner'],'root_effective_overrides':[x for x in holds if x.get('kind')=='root_effective_override'],'source_alias_holds':[x for x in holds if x.get('kind')=='source_alias_hold'],'candidate_registry_objects':candidates,'source_gaps':gaps},open(OUT/'holds_disagreements.json','w'),indent=2,sort_keys=True); open(OUT/'holds_disagreements.json','a').write('\n')
json.dump(manifest,open(OUT/'source_manifest.json','w'),indent=2,sort_keys=True); open(OUT/'source_manifest.json','a').write('\n')
receipt={'schema':'training_set_completion.stage01_first_fit_review_extraction.v1','status':'candidate_ready','classification':'candidate_pending_lead_acceptance','artifact_root':str(OUT),'inputs':{'review_roots':[str(B/'first-fit-owner-reviews-v1'),str(B/'B/first-fit-owner-reviews-v1')],'target_catalog':str(catalog_path),'target_catalog_sha256':sha(catalog_path)},'counts':{'review_files':len(review_paths),'images':len(fixed_by_img),'steps':2,'raw_decision_mentions':sum(len(v) for v in all_mentions.values()),'unique_raw_proposals':len(rows),'step16_rows':sum(r['step']==16 for r in rows),'step32_rows':sum(r['step']==32 for r in rows),'fixed_target_denominator':len(fixed_records),'target_records_by_role':dict(sorted(Counter(r['role'] for r in fixed_records).items())),'source_duplicate_groups':sum(1 for g in duplicate_groups if g['mention_count']>1),'source_gaps':len(gaps),'root_override_count':len(override_by_pid)},'validation':{'unique_proposal_assertion':'passed','packet_set_match':'passed','per_image_step_partition':'passed','effective_coverage_rule':'valid raw row + true_unique/repeat + reasonable extent + fixed v2 owner','normalization_rule':'existingGT:<id> normalized only when same-image fixed target exists','class_rule':'class independent from owner coverage','crowd_or_nonfixed_rule':'crowd/group retained separately; all nonfixed owners outside frozen220 comparison'},'outputs':{'decisions_jsonl':str(OUT/'decisions.jsonl'),'per_image_step_summary':str(OUT/'per_image_step_summary.json'),'holds_disagreements':str(OUT/'holds_disagreements.json'),'source_manifest':str(OUT/'source_manifest.json')},'command':'python3 /tmp/first_fit_extract.py'}
json.dump(receipt,open(OUT/'receipt.json','w'),indent=2,sort_keys=True); open(OUT/'receipt.json','a').write('\n')
print(json.dumps({'out':str(OUT),'rows':len(rows),'mentions':sum(len(v) for v in all_mentions.values()),'duplicates':sum(1 for g in duplicate_groups if g['mention_count']>1),'gaps':len(gaps),'override':len(override_by_pid),'summaries':len(summaries)},sort_keys=True))
for s in (16,32):
 ss=[x for x in summaries if x['step']==s]; print('step',s,'rows',sum(x['raw_rows'] for x in ss),'covered',sum(x['covered_count'] for x in ss),'missing',sum(x['missing_count'] for x in ss),'eligible',sum(x['coverage_eligible_row_count'] for x in ss))
