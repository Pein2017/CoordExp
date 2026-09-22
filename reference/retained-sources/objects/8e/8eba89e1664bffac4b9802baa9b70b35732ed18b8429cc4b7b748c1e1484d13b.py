import json,hashlib,importlib.util,shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
B=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum'); S=B/'third-fit-selectors-v1'; O=B/'third-fit-review-packets-v1'; ACQ=B/'stage01-acquisition-v1-retry1-config-batch2/manifest.json'; TARGET=B/'target-owners-complete-v4.json'; PARENT=B/'parent16-v4-physical-ledger-v1/ledger.json'; COHORT=(25274,59571,99937,210457,219546,323322,351017,388795,417044,477415,528944)
RPATH=B/'B/first-fit-visualization-preparation-v1/render_readback.py'; sp=importlib.util.spec_from_file_location('rr',RPATH);rr=importlib.util.module_from_spec(sp);sp.loader.exec_module(rr)
def c(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size}
def sig(x):
 raw=x.get('raw') if isinstance(x.get('raw'),dict) else {}
 return (x.get('status'),x.get('drop_reason'),x.get('description'),x.get('raw_span_sha256'),tuple(x.get('coord_bins_1000') or ()),tuple(x.get('bbox_pixel_xyxy') or ()),raw.get('raw_span_text'))
acq=json.load(open(ACQ));target=json.load(open(TARGET));parent=json.load(open(PARENT)); a={int(x['image_id']):x for x in acq['records']};t=defaultdict(list)
for x in target['records']:t[int(x['image_id'])].append(dict(x))
if O.exists() and any(O.iterdir()):raise SystemExit(f'nonempty output {O}')
O.mkdir(parents=True)
sc=json.load(open(S/'scored-step-32.json')); by={int(x['image_id']):x for x in sc['rows']}; packets=[]
for iid in COHORT:
 row=by[iid]; rec=a[iid]; ip=Path(rec['case']['image_path']).resolve(strict=True)
 with Image.open(ip) as im:image=im.convert('RGB')
 if list(image.size)!=[int(rec['case']['image_width']),int(rec['case']['image_height'])]:raise ValueError(iid)
 refs=[]
 for z in t[iid]:y=dict(z);y['bbox_pixel_xyxy']=rr.bins_to_pixels(z['reference_coord_bins_1000'],image.width,image.height);refs.append(y)
 raw=[]
 for z in row['raw_rows']:
  y=dict(z);y['_raw_box_pixel_xyxy']=rr.raw_box(y,image.width,image.height);raw.append(y)
 raw.sort(key=lambda z:(int(z.get('generated_order',10**9)),str(z.get('prediction_id',''))))
 d=O/f'image-{iid:012d}'; crops=d/'crops';crops.mkdir(parents=True); orig=d/'original.jpg';shutil.copyfile(ip,orig);to=d/'target-catalog-overlay.png';ro=d/'raw-generated-overlay.png';rr.render_target_overlay(image,refs,to,iid);rr.render_raw_overlay(image,raw,ro,iid)
 groups=defaultdict(list)
 for z in raw:groups[sig(z)].append(z)
 gs=[]; rowgid={}
 for gi,(sg,members) in enumerate(sorted(groups.items(),key=lambda kv:int(kv[1][0].get('generated_order',10**9)))):
  gid=f'g{gi:04d}';rep=members[0];tp=crops/f'{gid}-tight.png';cp=crops/f'{gid}-context.png';tf=rr.render_crop(image,rep,tp,8,'group-tight');cf=rr.render_crop(image,rep,cp,64,'group-context')
  for z in members:rowgid[str(z['prediction_id'])]=gid
  gs.append({'visual_group_id':gid,'member_prediction_ids':[str(z['prediction_id']) for z in members],'member_count':len(members),'signature_fields':{'status':rep.get('status'),'drop_reason':rep.get('drop_reason'),'description':rep.get('description'),'raw_span_sha256':rep.get('raw_span_sha256'),'coord_bins_1000':rep.get('coord_bins_1000'),'bbox_pixel_xyxy':rep.get('bbox_pixel_xyxy')},'tight_crop':{'path':str(tp.resolve()),'frame_pixel_xyxy':list(tf),'sha256':sha(tp)},'context_crop':{'path':str(cp.resolve()),'frame_pixel_xyxy':list(cf),'sha256':sha(cp)}})
 rendered=[]
 for z in raw:
  gid=rowgid[str(z['prediction_id'])];g=next(q for q in gs if q['visual_group_id']==gid);rendered.append({'prediction_id':z['prediction_id'],'generated_order':z.get('generated_order'),'status':z.get('status'),'drop_reason':z.get('drop_reason'),'description':z.get('description'),'coord_bins_1000':z.get('coord_bins_1000'),'raw_bbox_pixel_xyxy':z.get('bbox_pixel_xyxy'),'render_box_pixel_xyxy':z.get('_raw_box_pixel_xyxy'),'raw_span_sha256':z.get('raw_span_sha256'),'visual_group_id':gid,'shared_crop_paths':{'tight':g['tight_crop']['path'],'context':g['context_crop']['path']},'raw_source_row_sha256':hashlib.sha256(c(z)).hexdigest()})
 packet={'schema':'training_set_completion.third_fit_step32_review_packet.v1','status':'diagnostic_visualization_only','image_id':iid,'checkpoint_step':32,'route_id':row.get('route_id'),'scored_row_sha256':hashlib.sha256(c(row)).hexdigest(),'source':{'scored_json':bind(S/'scored-step-32.json'),'acquisition_manifest':bind(ACQ),'target_catalog_v4':bind(TARGET),'parent_v4_physical_ledger':bind(PARENT),'original_image':bind(ip),'dimensions':[image.width,image.height]},'target_catalog_references':[{'owner_id':z['owner_id'],'role':z.get('role'),'category':z.get('category'),'reference_coord_bins_1000':z['reference_coord_bins_1000'],'bbox_pixel_xyxy':z['bbox_pixel_xyxy']} for z in refs],'rendered':{'original_image':{'path':str(orig.resolve()),'sha256':sha(orig)},'target_catalog_overlay':{'path':str(to.resolve()),'sha256':sha(to)},'raw_generated_overlay':{'path':str(ro.resolve()),'sha256':sha(ro)},'visual_groups':gs,'raw_rows':rendered},'counts':{'target_reference_count':len(refs),'raw_row_count':len(raw),'valid_prediction_count':sum(z.get('status')=='parsed_valid' for z in raw),'raw_invalid_or_dropped_count':sum(z.get('status')!='parsed_valid' for z in raw),'visual_group_count':len(gs)},'acceptance_boundary':{'no_physical_review':True,'no_owner_or_class_decisions':True,'all_raw_rows_preserved':True,'grouping_visualization_only':True,'invalid_rows_visible':True,'processed_pixel_bbox_rendered_only':True,'matching_canvas_normalized_bins_1000':True}}
 (d/'packet.json').write_bytes(c(packet));packets.append({'image_id':iid,'packet':str((d/'packet.json').resolve()),'raw_row_count':len(raw),'valid_prediction_count':packet['counts']['valid_prediction_count'],'raw_invalid_or_dropped_count':packet['counts']['raw_invalid_or_dropped_count'],'visual_group_count':len(gs),'original_image':str(orig.resolve()),'raw_overlay':str(ro.resolve()),'target_overlay':str(to.resolve())})
# Compact receipt: selector sources carry all 33 metrics; packets are only step32.
rec={'schema':'training_set_completion.third_fit_review_packet_receipt.v1','status':'diagnostic_visualization_ready','selector_sources':{str(s):bind(S/f'scored-step-{s}.json') for s in (16,32,64)},'target_catalog_v4':bind(TARGET),'parent_v4_physical_ledger':bind(PARENT),'acquisition_manifest':bind(ACQ),'cohort_image_ids':list(COHORT),'rendered_checkpoint_step':32,'rendered_image_ids':list(COHORT),'missing_image_ids':[],'packets':packets,'raw_rows_preserved_in_selectors_and_packets':True,'no_physical_review_or_catalog_mutation':True}
rec['receipt_sha256']=hashlib.sha256(c(rec)).hexdigest();(O/'receipt.json').write_bytes(c(rec));print(json.dumps({'receipt':str(O/'receipt.json'),'packets':len(packets),'raw_step32':sum(x['raw_row_count'] for x in packets),'groups_step32':sum(x['visual_group_count'] for x in packets)},indent=2))
