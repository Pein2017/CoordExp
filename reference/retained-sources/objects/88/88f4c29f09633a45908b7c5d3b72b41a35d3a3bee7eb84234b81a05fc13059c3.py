"""Frozen <=4 events/refined image: first proxy per model, first paired loss else gain."""
from pathlib import Path
import json,argparse
from PIL import Image,ImageDraw
from probes.training_set_completion.readback_selectors import iou_xyxy as m_iou
a=argparse.ArgumentParser();a.add_argument('--input',default='reduction.json');args=a.parse_args();R=Path(__file__).parent;p=json.loads((R/'panel.json').read_text());d=json.loads((R/args.input).read_text());events=[];out=R/'physical-review';out.mkdir(exist_ok=True)
for iid in map(str,p['refined_source_order']):
 if not all(iid in d['cells'][c] for c in p['conditions']):continue
 case=next(c for c in p['refined_cases'] if str(c['input_record']['image_id'])==iid);bank=p['refined_banks'][iid];image_events=[]
 for model in ['tied','untied']:
  original=d['cells'][model+'-original'][iid]['views']['refined'];treated=d['cells'][model+'-normalized'][iid]['views']['refined'];repeat=original['first_strict_repeat']
  if repeat:
   rows=original['valid_predictions'];cur=next(x for x in rows if x['prediction_id']==repeat['prediction_id']);earlier=[x for x in rows if x['generated_order']<cur['generated_order']];prior=max(earlier,key=lambda x: m_iou(cur['coord_bins_1000'],x['coord_bins_1000']))
   image_events.append(dict(kind='first_repeat_proxy',model=model,rows=[dict(label='earlier',box=prior['coord_bins_1000'],row=prior['generated_order'],desc=prior['description']),dict(label='repeat',box=cur['coord_bins_1000'],row=cur['generated_order'],desc=cur['description'])],physical_status='HOLD_pending_review'))
  old=set(original['matches']['covered_owner_ids']);new=set(treated['matches']['covered_owner_ids']);loss=sorted(old-new);gain=sorted(new-old)
  if loss or gain:
   owner=(loss or gain)[0];o=next(o for o in bank if o['owner_id']==owner);boxes=[dict(label='trusted',box=o['reference_coord_bins_1000'],row=None,desc=o['description'])]
   for name,z in [('original',original),('normalized',treated)]:
    matches=[m for m in z['matches']['matches'] if str(m['reference_owner_id'])==owner]
    if matches:
     row=next(v for v in z['valid_predictions'] if v['prediction_id']==matches[0]['prediction_id']);boxes.append(dict(label=name,box=row['coord_bins_1000'],row=row['generated_order'],desc=row['description']))
    elif z['valid_predictions']:
     row=max(z['valid_predictions'],key=lambda v:m_iou(v['coord_bins_1000'],o['reference_coord_bins_1000']));overlap=m_iou(row['coord_bins_1000'],o['reference_coord_bins_1000'])
     if overlap>0:boxes.append(dict(label=name+' nearest-IoU proxy',box=row['coord_bins_1000'],row=row['generated_order'],desc=row['description'],iou=overlap))
   image_events.append(dict(kind='first_known_loss' if loss else 'first_known_gain',model=model,owner_id=owner,rows=boxes,physical_status='HOLD_pending_review',limit='Absent IoU match does not establish physical disappearance; changed geometry or assignment can remain unresolved.'))
 for j,event in enumerate(image_events):
  im=Image.open(case['image_path']).convert('RGB');draw=ImageDraw.Draw(im);colors=['cyan','red','yellow']
  for k,row in enumerate(event['rows']):
   b=row['box'];draw.rectangle([b[0]*im.width/999,b[1]*im.height/999,b[2]*im.width/999,b[3]*im.height/999],outline=colors[k],width=3);draw.text((8,8+16*k),f"{colors[k]} {row['label']} row{row['row']} {row['desc']}",fill=colors[k],stroke_width=1)
  name=f'{iid}-{j}-{event["model"]}-{event["kind"]}.png';im.save(out/name);boxes=[row['box'] for row in event['rows']];x1=min(b[0] for b in boxes)*im.width/999;y1=min(b[1] for b in boxes)*im.height/999;x2=max(b[2] for b in boxes)*im.width/999;y2=max(b[3] for b in boxes)*im.height/999;pad=max(30,(x2-x1)*.3,(y2-y1)*.3);crop=im.crop((max(0,x1-pad),max(0,y1-pad),min(im.width,x2+pad),min(im.height,y2+pad)));crop.thumbnail((600,600));crop.save(out/name.replace('.png','-context.png'));event['context']=str(out/name.replace('.png','-context.png'));event.update(image_id=int(iid),image_path=case['image_path'],plot=str(out/name));events.append(event)
assert len(events)<=72
(out/'events.json').write_text(json.dumps(dict(selection='first repeat proxy original per model; first known loss else first gain sorted owner ID per model',events=events),indent=2)+'\n');print(len(events))
