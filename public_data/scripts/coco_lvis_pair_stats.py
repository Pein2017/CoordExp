"""COCO/LVIS shared-image pair statistics; image and instance denominators remain distinct.

The three original split comparisons and crowd exclusion are preserved. Full raw
annotation processing is explicit; importing this module performs no reads.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import orjson



def iou_and_cover(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = a[2] * a[3]; bb = b[2] * b[3]
    return inter / (aa+bb-inter) if aa+bb-inter else 0.0, inter / aa if aa else 0.0, inter / bb if bb else 0.0

def analyze(coco_name: str, lvis_name: str, *, data_root: Path):
    coco = orjson.loads((data_root/'coco/raw/annotations'/coco_name).read_bytes())
    lvis = orjson.loads((data_root/'lvis/raw/annotations'/lvis_name).read_bytes())
    ci = {int(x['id']) for x in coco['images']}
    li = {int(x['id']) for x in lvis['images']}
    shared = ci & li
    cb = defaultdict(list); lb = defaultdict(list)
    crowd = 0
    for a in coco['annotations']:
        if a['image_id'] in shared:
            if a.get('iscrowd', 0): crowd += 1
            else: cb[a['image_id']].append((int(a['category_id']), a['bbox']))
    for a in lvis['annotations']:
        if a['image_id'] in shared:
            lb[a['image_id']].append((int(a['category_id']), a['bbox']))
    linst = Counter(); limg = Counter(); cimg = Counter(); pair_img=Counter(); pair_inst_with_c=Counter()
    overlap = defaultdict(lambda: [0,0,0,0,0,0,0,0])
    # IoU >= .1/.3/.5/.75; LVIS and COCO coverage >= .5; sum max IoU; sum max LVIS coverage.
    for imgid in shared:
        ca=cb[imgid]; la=lb[imgid]
        c_by_cat=defaultdict(list)
        for c,b in ca: c_by_cat[c].append(b)
        lcats={l for l,_ in la}
        for c in c_by_cat: cimg[c]+=1
        for l in lcats:
            limg[l]+=1
            for c in c_by_cat: pair_img[(l,c)]+=1
        for l,a in la:
            linst[l]+=1
            for c,boxes in c_by_cat.items():
                pair_inst_with_c[(l,c)] += 1
                vals = [iou_and_cover(a,b) for b in boxes]
                bi=max(v[0] for v in vals)
                bc=max(v[1] for v in vals)
                cc=max(v[2] for v in vals)
                z=overlap[(l,c)]
                z[0]+=bi>=.1; z[1]+=bi>=.3; z[2]+=bi>=.5; z[3]+=bi>=.75
                z[4]+=bc>=.5; z[5]+=cc>=.5; z[6]+=bi; z[7]+=bc
    rows=[]
    for (l,c),n in pair_inst_with_c.items():
        z=overlap[(l,c)]
        rows.append({'l':l,'c':c,'l_instances':linst[l],'l_images':limg[l],
            'c_images':cimg[c], 'images_both':pair_img[(l,c)], 'instances_with_c_image':n,
            'iou01':z[0], 'iou03':z[1], 'iou05':z[2], 'iou075':z[3],
            'l_cover05':z[4], 'c_cover05':z[5], 'mean_best_iou_when_c_image':round(z[6]/n,5),
            'mean_best_l_cover_when_c_image':round(z[7]/n,5)})
    return {'coco_file':coco_name,'lvis_file':lvis_name,'coco_images':len(ci),'lvis_images':len(li),
        'shared_images':len(shared),'shared_coco_instances_non_crowd':sum(map(len,cb.values())),
        'shared_lvis_instances':sum(linst.values()), 'coco_crowd_excluded':crowd,
        'lvis_categories':{str(x['id']):{'name':x['name'],'synonyms':x.get('synonyms',[]),'def':x.get('def',''), 'frequency':x.get('frequency','')} for x in lvis['categories']},
        'coco_categories':{str(x['id']):x['name'] for x in coco['categories']},
        'rows':rows}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; no overwrite is permitted')
    results = [analyze(c, l, data_root=args.data_root) for c, l in (
        ('instances_train2017.json', 'lvis_v1_train.json'),
        ('instances_train2017.json', 'lvis_v1_val.json'),
        ('instances_val2017.json', 'lvis_v1_val.json'))]
    with args.output.open('xb') as stream:
        stream.write(orjson.dumps(results))


if __name__ == '__main__':
    main()
