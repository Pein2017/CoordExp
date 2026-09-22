"""One-off CPU FP census and seeded visual review packet; never rewrites labels."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from probes.dora_owner_learning.reward_rows import _gt_objects, _pred_objects
from src.artifacts import publish_json_exclusive
from src.data.geometry import iou_xyxy
from src.vis import render_gt_vs_prediction

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
OUT = ROOT/'2026-09-09-fp-visual-distribution'
SOURCE = ROOT/'2026-09-09-round1-greedy-realization'
RUN = SOURCE/'cold/source256-rloo-round1-train256-natural-v1'
DRAW_FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
SEED = 20260909
QUOTAS = {'strict_repeat': 10, 'same_category_near_GT': 20,
          'other_category_high_overlap': 14, 'weak_GT_relation': 20}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, values):
    with path.open('x') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True)+'\n')


def padded_panel(im, size=(520, 520)):
    result = Image.new('RGB', size, 'white')
    scale = min(size[0]/im.width, size[1]/im.height)
    scaled = im.resize((max(1, round(im.width*scale)), max(1, round(im.height*scale))))
    xy = ((size[0]-scaled.width)//2, (size[1]-scaled.height)//2)
    result.paste(scaled, xy)
    return result


def render_crop(item, gt):
    with Image.open(item['image_path']) as source:
        image = source.convert('RGB')
    assert image.size == (item['width'], item['height'])
    x1,y1,x2,y2 = item['bbox']
    cx,cy = (x1+x2)/2,(y1+y2)/2
    halfx,halfy = max(80,(x2-x1)*.85),max(80,(y2-y1)*.85)
    region = (max(0,int(cx-halfx)),max(0,int(cy-halfy)),
              min(image.width,int(cx+halfx)+1),min(image.height,int(cy+halfy)+1))
    raw = image.crop(region)
    annotated = raw.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.truetype(DRAW_FONT, max(10,round(raw.width/55)))
    def clipped(box):
        a,b,c,d = box
        if c<=region[0] or a>=region[2] or d<=region[1] or b>=region[3]:
            return None
        return (max(0,a-region[0]),max(0,b-region[1]),
                min(raw.width-1,c-region[0]),min(raw.height-1,d-region[1]))
    for k,(category,box) in enumerate(gt):
        b = clipped(box)
        if b is not None:
            draw.rectangle(b,outline=(35,130,255),width=max(1,raw.width//350))
            draw.text((b[0]+1,b[1]+1),f'G{k} {category}',font=font,fill=(35,130,255))
    if item['repeat_parent'] is not None and item['repeat_iou']>.95:
        draw.rectangle(clipped(item['repeat_parent_bbox']),outline=(200,20,190),width=max(2,raw.width//200))
    draw.rectangle(clipped(item['bbox']),outline=(240,20,20),width=max(2,raw.width//200))
    canvas = Image.new('RGB',(1060,600),'white')
    cd = ImageDraw.Draw(canvas)
    titlefont = ImageFont.truetype(DRAW_FONT,18)
    cd.text((10,8),f"{item['case_id']} | prediction: {item['category']} | {item['stratum']}",font=titlefont,fill='black')
    cd.text((10,34),f"GT IoU same={item['best_same_iou']:.3f} any={item['best_any_iou']:.3f} | repeat={item['repeat_iou']:.3f}",font=titlefont,fill='black')
    canvas.paste(padded_panel(raw),(10,65))
    canvas.paste(padded_panel(annotated),(540,65))
    path = OUT/'crops'/f"{item['image_id']}_p{item['pred_index']:03d}.png"
    canvas.save(path)
    item['crop_region_xyxy'] = list(region)
    item['crop_path'] = str(path)


def main():
    cases_path = SOURCE/'analysis-v1/cases.json'
    raw_path = RUN/'gt_vs_pred.jsonl'
    cases = {c['example_id']: c for c in json.loads(cases_path.read_text())}
    raw = {r['row_id']:r for r in map(json.loads,raw_path.read_text().splitlines())}
    assert len(raw)==len(cases)==256 and raw.keys()==cases.keys()
    assert not (OUT/'inventory.jsonl').exists()
    for directory in ('crops','batches','reviews'):
        (OUT/directory).mkdir(exist_ok=False)
    inventory=[]
    gt_by_image={}
    row_counts=[]
    capped=[]
    for eid,row in raw.items():
        c=cases[eid]
        pred,invalid=_pred_objects(row)
        assert invalid==0 and len(pred)==len(row['pred'])
        gt=_gt_objects(row,row_id=eid)
        gt_by_image[eid]=gt
        matched={m['pred_index'] for m in c['post']['50']['matches']}
        assert len(pred)-len(matched)==c['post']['50']['fp']
        row_counts.append({'example_id':eid,'image_id':c['image_id'],'fp':c['post']['50']['fp'],
                           'tp':len(matched),'repeats':c['post']['strict_repeats'],
                           'cap':c['post']['cap'],'image_path':row['image_path']})
        if c['post']['cap']: capped.append(eid)
        for j,(cat,box) in enumerate(pred):
            if j in matched: continue
            overlaps=sorted(((iou_xyxy(box,b),k,gcat) for k,(gcat,b) in enumerate(gt)),reverse=True)
            best_any=max((v for v,k,gcat in overlaps),default=0.)
            best_same=max((v for v,k,gcat in overlaps if gcat==cat),default=0.)
            repeat_iou,parent=max(((iou_xyxy(box,b),k) for k,(_,b) in enumerate(pred[:j])),default=(0.,None))
            if repeat_iou>.95: stratum='strict_repeat'
            elif best_same>=.1: stratum='same_category_near_GT'
            elif best_any>=.5: stratum='other_category_high_overlap'
            else: stratum='weak_GT_relation'
            inventory.append({'case_id':f"{c['image_id']}:p{j}",'example_id':eid,
                'image_id':c['image_id'],'pred_index':j,'category':cat,'bbox':list(box),
                'width':row['image_width'],'height':row['image_height'],'image_path':row['image_path'],
                'stratum':stratum,'best_same_iou':best_same,'best_any_iou':best_any,
                'nearby_gt':[{'index':k,'category':gcat,'iou':v,'bbox':list(gt[k][1])} for v,k,gcat in overlaps[:5]],
                'repeat_iou':repeat_iou,'repeat_parent':parent,
                'repeat_parent_bbox':list(pred[parent][1]) if parent is not None else None,
                'repeat_parent_category':pred[parent][0] if parent is not None else None,
                'repeat_parent_is_matched':parent in matched,'image_capped':bool(c['post']['cap'])})
    assert len(inventory)==1094 and sum(x['stratum']=='strict_repeat' for x in inventory)==484
    assert sum(x['image_capped'] for x in inventory)==511
    rng=random.Random(SEED)
    selected=[]
    strata=Counter(x['stratum'] for x in inventory)
    for stratum,n in QUOTAS.items():
        population=sorted((x for x in inventory if x['stratum']==stratum),key=lambda x:x['case_id'])
        assert len(population)>=n
        for x in rng.sample(population,n):
            selected.append(dict(x,population_size=len(population),stratum_sample_size=n,
                                 sample_weight=len(population)/n))
    assert len(selected)==64 and len({x['case_id'] for x in selected})==64
    selected_images=sorted({x['example_id'] for x in selected}|set(capped))
    vis=render_gt_vs_prediction(RUN,OUT/'native-overviews',row_ids=selected_images,
                               duplicate_iou_threshold=.95)
    assert len(vis.image_paths)==len(selected_images)
    overviews=dict(zip(selected_images,map(str,vis.image_paths)))
    for x in selected:
        x['overview_path']=overviews[x['example_id']]
        render_crop(x,gt_by_image[x['example_id']])
    root_images={x['example_id'] for x in selected if x['stratum']=='strict_repeat'}|set(capped)
    batches={key:[] for key in ('root','a','b','c')}
    by_image=defaultdict(list)
    for x in selected: by_image[x['example_id']].append(x)
    for eid,items in sorted(by_image.items(),key=lambda p:(-len(p[1]),p[0])):
        owner='root' if eid in root_images else min(('a','b','c'),key=lambda k:len(batches[k]))
        batches[owner].extend(items)
    for owner,items in batches.items():
        write_jsonl(OUT/'batches'/f'{owner}.jsonl',items)
    write_jsonl(OUT/'inventory.jsonl',inventory)
    write_jsonl(OUT/'sample.jsonl',selected)
    publish_json_exclusive(OUT/'summary.json',{
        'schema_version':'fp_visual_audit_packet.v1','scope':'Source256 RLOO round1 train256 FP50',
        'source_run':str(RUN),'source_files':{str(raw_path):sha(raw_path),str(cases_path):sha(cases_path)},
        'script_sha256':sha(__file__),'images':256,'gt':1955,'tp':1262,'fp':1094,
        'images_with_fp':sum(x['fp']>0 for x in row_counts),'fp_in_capped_images':511,
        'strict_repeat_fp':484,'strata':dict(strata),'sample_seed':SEED,'sample_quotas':QUOTAS,
        'sample_size':64,'sample_image_count':len(by_image),'overview_image_count':len(selected_images),
        'batches':{k:len(v) for k,v in batches.items()},'category_counts':dict(Counter(x['category'] for x in inventory)),
        'image_counts':row_counts,'capped_image_overviews':{eid:overviews[eid] for eid in capped},
        'interpretation':'Global matcher owns FP membership; native overview matching colors are reference-only. Crops preserve raw pixels and show the target red, GT blue, nearest strict repeat magenta. Visual judgments are candidate evidence, not ground truth.',
        'sampling':'Uniform random sampling without replacement separately within each deterministic stratum; every other-category-high-overlap FP is included. Unknown judgments stay in the sample. No visual or outcome-based replacement.'})
    print(json.dumps({'fp':1094,'strata':dict(strata),'samples':len(selected),'sample_images':len(by_image),
                      'batch_sizes':{k:len(v) for k,v in batches.items()},'capped_contexts':len(capped)}))


if __name__=='__main__':
    main()
