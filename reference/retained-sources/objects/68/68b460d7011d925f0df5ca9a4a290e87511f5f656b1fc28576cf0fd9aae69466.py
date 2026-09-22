from __future__ import annotations
import hashlib, importlib.util, json, shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image

B=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum')
SEL=B/'second-fit-selectors-v1'; PACK=B/'second-fit-review-packets-v1'
COHORT=(25274,59571,99937,210457,219546,323322,351017,388795,417044,477415,528944)
ACQ=B/'stage01-acquisition-v1-retry1-config-batch2/manifest.json'
TARGET=B/'target-owners-complete-v3.json'
STAGE2=B/'stage02-refresh-selectors-v1/scored-t0.json'
RENDER=B/'B/first-fit-visualization-preparation-v1/render_readback.py'

def canonical(v): return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p).resolve(strict=True); return {'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size}
def dump(p,v): Path(p).write_bytes(canonical(v))
def sig(row):
 # Exact viewing signature: token span identity plus parsed category/axes and raw status.
 raw=row.get('raw') if isinstance(row.get('raw'),dict) else {}
 return (row.get('status'),row.get('drop_reason'),row.get('description'),row.get('raw_span_sha256'),
         tuple(row.get('coord_bins_1000') or ()),tuple(row.get('bbox_pixel_xyxy') or ()),
         raw.get('raw_span_text'))
def metric(row):
 m=row['class_agnostic_iou_matching']
 return {'raw_row_count':row['raw_row_count'],'valid_prediction_count':row['valid_prediction_count'],
  'raw_row_debt_count':row['raw_row_debt_count'],'token_count':row['token_count'],
  'natural_eos':row['natural_eos'],'capped':row['capped'],'capped_by_limit':row.get('capped_by_limit'),
  'cap_debt':row['cap_debt'],'eos_debt':row['eos_debt'],'excess_token_count':row.get('excess_token_count'),
  'strict_pairwise_iou_gt_0_95_pair_count':len(row.get('pairwise_iou_gt_0.95') or []),
  'thresholds': {t:{'matches':len(m[t]['matches']),
    'missing_reference_owner_ids':list(m[t]['missing_reference_owner_ids']),
    'unmatched_prediction_ids':[x if isinstance(x,str) else x['prediction_id'] for x in m[t]['unmatched_predictions']]}
    for t in ('0.5','0.8')}}
def comparison():
 stages=json.loads(STAGE2.read_text())
 second={s:json.loads((SEL/f'scored-step-{s}.json').read_text()) for s in (16,32,64)}
 ids=list(COHORT)
 if stages['readback_image_ids']!=ids and set(stages['readback_image_ids'])!=set(ids): raise ValueError('stage02 image coverage mismatch')
 if stages['sources']['target_owners']['sha256'] != bind(TARGET)['sha256']: raise ValueError('stage02 target hash does not match v3')
 st_by={int(r['image_id']):r for r in stages['rows']}
 allrows=[]
 for s,d in second.items():
  got=[int(r['image_id']) for r in d['rows']]
  if got!=ids and set(got)!=set(ids): raise ValueError(f'second-fit {s} image coverage mismatch')
  for r in d['rows']:
   i=int(r['image_id']); a=metric(r); b=metric(st_by[i])
   # Deltas stay geometric and explicitly exclude physical/owner claims.
   delta={'raw_row_count':a['raw_row_count']-b['raw_row_count'],
    'valid_prediction_count':a['valid_prediction_count']-b['valid_prediction_count'],
    'raw_row_debt_count':a['raw_row_debt_count']-b['raw_row_debt_count'],
    'cap_debt':a['cap_debt']-b['cap_debt'],'eos_debt':a['eos_debt']-b['eos_debt']}
   for t in ('0.5','0.8'):
    delta[f'matches_{t}']=a['thresholds'][t]['matches']-b['thresholds'][t]['matches']
    delta[f'missing_{t}']=len(a['thresholds'][t]['missing_reference_owner_ids'])-len(b['thresholds'][t]['missing_reference_owner_ids'])
    delta[f'unmatched_{t}']=len(a['thresholds'][t]['unmatched_prediction_ids'])-len(b['thresholds'][t]['unmatched_prediction_ids'])
   allrows.append({'image_id':i,'second_fit_step':s,'second_fit':a,'stage02_refresh_parent16':b,'geometric_delta':delta})
 out={'schema':'training_set_completion.second_fit_geometric_selector_comparison.v1','status':'diagnostic_only_pending_owner_review',
  'coordinate_semantics':{'pred_matching_field':'coord_bins_1000','reference_field':'reference_coord_bins_1000','domain':[0,1000],
   'processed_pixel_bbox':'retained_for_rendering_only','pixel_bbox_not_compared_to_normalized_references':True},
  'cohort_image_ids':ids,'sources':{'stage02_refresh_scored':bind(STAGE2),'target_catalog_v3':bind(TARGET),
   'second_fit_scored':{str(s):bind(SEL/f'scored-step-{s}.json') for s in (16,32,64)}},
  'rows':allrows,
  'metric_scope':'Class-agnostic IoU matching, raw/valid/drop/cap/EOS debt, and pairwise strict-repeat diagnostics only; no physical identity or acceptance labels.',
  'raw_rows_preserved_in_source_selectors':True}
 dump(SEL/'geometric-comparison-to-stage02.json',out)
 # compact receipt aggregate, computed from rows
 agg={}
 for s in (16,32,64):
  rs=[x for x in allrows if x['second_fit_step']==s]
  agg[str(s)]={'raw_row_count':sum(x['second_fit']['raw_row_count'] for x in rs),
   'valid_prediction_count':sum(x['second_fit']['valid_prediction_count'] for x in rs),
   'raw_row_debt_count':sum(x['second_fit']['raw_row_debt_count'] for x in rs),
   'cap_debt':sum(x['second_fit']['cap_debt'] for x in rs),'eos_debt':sum(x['second_fit']['eos_debt'] for x in rs),
   'strict_pairwise_iou_gt_0_95_pair_count':sum(x['second_fit']['strict_pairwise_iou_gt_0_95_pair_count'] for x in rs),
   'matches':{t:sum(x['second_fit']['thresholds'][t]['matches'] for x in rs) for t in ('0.5','0.8')}}
 receipt={'schema':'training_set_completion.second_fit_selector_receipt.v1','status':'diagnostic_only_pending_owner_review',
  'selector_artifacts':{str(s):bind(SEL/f'scored-step-{s}.json') for s in (16,32,64)},'comparison_artifact':bind(SEL/'geometric-comparison-to-stage02.json'),
  'stage02_parent16_comparison':bind(STAGE2),'target_catalog_v3':bind(TARGET),'cohort_image_ids':ids,
  'missing_image_ids':[],'aggregate_geometric_counts':agg,'raw_rows_retained':True,
  'physical_acceptance_reference':bind(B/'parent16-v3-physical-ledger-v2/root-acceptance.json'),
  'physical_acceptance_used_for_selector_metrics':False}
 receipt['receipt_sha256']=hashlib.sha256(canonical(receipt)).hexdigest(); dump(SEL/'receipt.json',receipt)
 return second

def render(second):
 # import established CPU drawing primitives, avoiding its per-row crop explosion.
 spec=importlib.util.spec_from_file_location('rr',RENDER); rr=importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
 acq=json.loads(ACQ.read_text()); target=json.loads(TARGET.read_text())
 acq_by={int(x['image_id']):x for x in acq['records']}
 target_by=defaultdict(list)
 for ref in target['records']: target_by[int(ref['image_id'])].append(dict(ref))
 step=second[16]; step_by={int(x['image_id']):x for x in step['rows']}
 if set(step_by)!=set(COHORT): raise ValueError('step16 packet image coverage mismatch')
 if PACK.exists() and any(PACK.iterdir()): raise ValueError(f'refusing nonempty packet root {PACK}')
 PACK.mkdir(parents=True,exist_ok=True)
 packet_summaries=[]
 for image_id in COHORT:
  scored=step_by[image_id]; rec=acq_by[image_id]
  image_path=Path(rec['case']['image_path']).resolve(strict=True)
  with Image.open(image_path) as im: image=im.convert('RGB')
  if [image.width,image.height] != [int(rec['case']['image_width']),int(rec['case']['image_height'])]: raise ValueError(f'dimensions {image_id}')
  refs=[]
  for ref in target_by[image_id]:
   r=dict(ref); r['bbox_pixel_xyxy']=rr.bins_to_pixels(ref['reference_coord_bins_1000'],image.width,image.height); refs.append(r)
  rows=[]
  for x in scored['raw_rows']:
   y=dict(x); y['_raw_box_pixel_xyxy']=rr.raw_box(y,image.width,image.height); rows.append(y)
  rows.sort(key=lambda x:(int(x.get('generated_order',10**9)),str(x.get('prediction_id',''))))
  d=PACK/f'image-{image_id:012d}'; c=d/'crops'; c.mkdir(parents=True)
  orig=d/'original.jpg'; shutil.copyfile(image_path,orig)
  tgt=d/'target-catalog-overlay.png'; raw=d/'raw-generated-overlay.png'
  rr.render_target_overlay(image,refs,tgt,image_id); rr.render_raw_overlay(image,rows,raw,image_id)
  groups=defaultdict(list)
  for x in rows: groups[sig(x)].append(x)
  group_rows=[]
  # exact group order follows first generated order; deterministic and readable.
  ordered=sorted(groups.items(),key=lambda kv:(int(kv[1][0].get('generated_order',10**9)),str(kv[1][0].get('prediction_id',''))))
  row_to_group={}
  for gi,(signature,members) in enumerate(ordered):
   gid=f'g{gi:04d}'; rep=members[0]; tight=c/f'{gid}-tight.png'; context=c/f'{gid}-context.png'
   tf=rr.render_crop(image,rep,tight,8,'group-tight'); cf=rr.render_crop(image,rep,context,64,'group-context')
   member_ids=[str(x.get('prediction_id')) for x in members]
   for x in members: row_to_group[str(x.get('prediction_id'))]=gid
   group_rows.append({'visual_group_id':gid,'member_prediction_ids':member_ids,'member_count':len(members),
    'signature_fields':{'status':rep.get('status'),'drop_reason':rep.get('drop_reason'),'description':rep.get('description'),
      'raw_span_sha256':rep.get('raw_span_sha256'),'coord_bins_1000':rep.get('coord_bins_1000'),'bbox_pixel_xyxy':rep.get('bbox_pixel_xyxy')},
    'representative_prediction_id':str(rep.get('prediction_id')),
    'tight_crop':{'path':str(tight.resolve()),'frame_pixel_xyxy':list(tf),'sha256':sha(tight)},
    'context_crop':{'path':str(context.resolve()),'frame_pixel_xyxy':list(cf),'sha256':sha(context)}})
  rendered=[]
  for x in rows:
   pid=str(x.get('prediction_id')); gid=row_to_group[pid]; g=next(z for z in group_rows if z['visual_group_id']==gid)
   rendered.append({'prediction_id':pid,'generated_order':x.get('generated_order'),'status':x.get('status'),'drop_reason':x.get('drop_reason'),
    'description':x.get('description'),'coord_bins_1000':x.get('coord_bins_1000'),'raw_bbox_pixel_xyxy':x.get('bbox_pixel_xyxy'),
    'render_box_pixel_xyxy':x.get('_raw_box_pixel_xyxy'),'raw_span_sha256':x.get('raw_span_sha256'),'visual_group_id':gid,
    'shared_crop_paths':{'tight':g['tight_crop']['path'],'context':g['context_crop']['path']},
    'raw_source_row_sha256':hashlib.sha256(canonical(x)).hexdigest()})
  packet={'schema':'training_set_completion.second_fit_review_packet.v1','status':'diagnostic_visualization_only',
   'image_id':image_id,'dose_step':16,'route_id':scored.get('route_id'),'scored_row_sha256':hashlib.sha256(canonical(scored)).hexdigest(),
   'source':{'scored_json':bind(SEL/'scored-step-16.json'),'acquisition_manifest':bind(ACQ),'target_catalog_v3':bind(TARGET),
    'original_image':bind(image_path),'physical_acceptance_reference':bind(B/'parent16-v3-physical-ledger-v2/root-acceptance.json'),
    'dimensions':[image.width,image.height]},
   'target_catalog_references':[{'owner_id':r['owner_id'],'role':r.get('role'),'category':r.get('category'),'reference_coord_bins_1000':r['reference_coord_bins_1000'],'bbox_pixel_xyxy':r['bbox_pixel_xyxy']} for r in refs],
   'rendered':{'original_image':{'path':str(orig.resolve()),'sha256':sha(orig)},'target_catalog_overlay':{'path':str(tgt.resolve()),'sha256':sha(tgt)},
    'raw_generated_overlay':{'path':str(raw.resolve()),'sha256':sha(raw)},'visual_groups':group_rows,'raw_rows':rendered},
   'counts':{'target_reference_count':len(refs),'raw_row_count':len(rows),'valid_prediction_count':sum(x.get('status')=='parsed_valid' for x in rows),
    'raw_invalid_or_dropped_count':sum(x.get('status')!='parsed_valid' for x in rows),'visual_group_count':len(group_rows)},
   'acceptance_boundary':{'no_owner_or_class_decisions':True,'no_physical_acceptance_applied':True,'all_raw_rows_preserved':True,
    'grouping_is_visualization_only':True,'invalid_rows_visible':True,'processed_pixel_bbox_rendered_as_pixel_axes':True,
    'normalized_coord_bins_are_the_matching_canvas':True}}
  pp=d/'packet.json'; dump(pp,packet)
  packet_summaries.append({'image_id':image_id,'packet':str(pp.resolve()),'raw_row_count':len(rows),'valid_prediction_count':sum(x.get('status')=='parsed_valid' for x in rows),'invalid_or_dropped_count':sum(x.get('status')!='parsed_valid' for x in rows),'visual_group_count':len(group_rows),'original_image':str(orig.resolve()),'target_overlay':str(tgt.resolve()),'raw_overlay':str(raw.resolve())})
 # Cross-dose exact groups preserve all member raw rows, and metrics remain in selectors unchanged.
 cross={}
 for image_id in COHORT:
  by=defaultdict(list)
  for dose in (16,32,64):
   row=next(x for x in second[dose]['rows'] if int(x['image_id'])==image_id)
   for x in row['raw_rows']:
    by[sig(x)].append({'dose_step':dose,'prediction_id':x.get('prediction_id'),'generated_order':x.get('generated_order'),
      'status':x.get('status'),'drop_reason':x.get('drop_reason'),'description':x.get('description'),
      'coord_bins_1000':x.get('coord_bins_1000'),'bbox_pixel_xyxy':x.get('bbox_pixel_xyxy'),'raw_span_sha256':x.get('raw_span_sha256')})
  gs=[]
  for gi,(signature,members) in enumerate(sorted(by.items(),key=lambda kv:(int(kv[1][0].get('dose_step',10**9)),int(kv[1][0].get('generated_order',10**9)),str(kv[1][0].get('prediction_id',''))))):
   if len(members)>1: gs.append({'exact_signature_group_id':f'g{gi:05d}','member_count':len(members),'dose_steps':sorted({m['dose_step'] for m in members}),'members':members,
     'deduplication_for_metrics':False})
  cross[str(image_id)]={'raw_row_counts_by_step':{str(d):next(x for x in second[d]['rows'] if int(x['image_id'])==image_id)['raw_row_count'] for d in (16,32,64)},
   'visual_exact_repeat_group_count':len(gs),'visual_exact_repeat_groups':gs,
   'strict_pairwise_iou_gt_0_95_pairs_by_step':{str(d):next(x for x in second[d]['rows'] if int(x['image_id'])==image_id)['pairwise_iou_gt_0.95'] for d in (16,32,64)}}
 repeat={'schema':'training_set_completion.second_fit_exact_repeat_groups.v1','status':'diagnostic_visualization_only',
  'source_selectors':{str(d):bind(SEL/f'scored-step-{d}.json') for d in (16,32,64)},'cohort_image_ids':list(COHORT),
  'images':cross,'grouping_semantics':'Exact full-row token span/category/coordinate/bbox/status signatures for viewing only; every member remains a raw row and no acceptance deduplication occurs.'}
 dump(PACK/'repeat-groups-across-doses.json',repeat)
 rec={'schema':'training_set_completion.second_fit_review_packet_receipt.v1','status':'diagnostic_visualization_ready',
  'source_selectors':{str(d):bind(SEL/f'scored-step-{d}.json') for d in (16,32,64)},'target_catalog_v3':bind(TARGET),
  'acquisition_manifest':bind(ACQ),'physical_acceptance_reference':bind(B/'parent16-v3-physical-ledger-v2/root-acceptance.json'),
  'cohort_image_ids':list(COHORT),'rendered_step':16,'rendered_image_ids':list(COHORT),'missing_image_ids':[],
  'packets':packet_summaries,'repeat_groups':bind(PACK/'repeat-groups-across-doses.json'),
  'all_raw_rows_preserved_in_selector_sources':True,'no_physical_acceptance_or_deduplication':True}
 rec['receipt_sha256']=hashlib.sha256(canonical(rec)).hexdigest(); dump(PACK/'receipt.json',rec)
 return rec

if __name__=='__main__':
 second=comparison(); rec=render(second); print(json.dumps({'selector_receipt':str(SEL/'receipt.json'),'packet_receipt':str(PACK/'receipt.json'),'packets':len(rec['packets'])},indent=2))
