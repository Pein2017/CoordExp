"""Reuse the already installed Co-DETR image API; no GT or candidates enter it."""
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image

REPO=Path('/data/CoordExp/external/Co-DETR')
CONFIG=REPO/'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py'
WEIGHTS=REPO/'models/co_dino_5scale_vit_large_coco.pth'
HELPER=REPO/'tools/codetr_infer_human_refined12.py'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.

def decide(candidate,predictions):
    valid=[p for p in predictions if p['score']>=.5]
    same=[p for p in valid if p['category']==candidate['category']]
    best=max(same,key=lambda p:iou(p['bbox_xyxy'],candidate['bbox'])) if same else None
    overlap=iou(best['bbox_xyxy'],candidate['bbox']) if best else 0.
    other=[p for p in valid if p['category']!=candidate['category']]
    alt=max(other,key=lambda p:iou(p['bbox_xyxy'],candidate['bbox'])) if other else None
    alt_iou=iou(alt['bbox_xyxy'],candidate['bbox']) if alt else 0.
    accepted=overlap>=.70
    return {'decision':'accept' if accepted else 'unknown','entity_category_support':'supported_candidate' if overlap>=.25 else 'unknown','localization':'agrees' if accepted else ('disagrees' if overlap>=.25 else 'unknown'),'category_conflict_flag':alt_iou>=.5,'candidate_iou':overlap,'selected':best,'alternative_category_iou':alt_iou,'alternative':alt,'scope':'Co-DETR corroboration, not GT; non-detection is unknown; no cross-prediction duplicate ruling'}

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    source,out=Path(sys.argv[1]),Path(sys.argv[2]);out.mkdir(exist_ok=False)
    raw=json.loads(source.read_text()) if source.suffix=='.json' else [json.loads(l) for l in source.read_text().splitlines()]
    rows=[r.get('source_sample',r) for r in raw]
    unique={}
    for r in rows:
        unique.setdefault(r['image_path'],r)
    import torch,mmcv,mmdet
    spec=importlib.util.spec_from_file_location('codetr_existing_runner',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    start=time.perf_counter()
    frozen={'source':str(source),'source_sha256':sha(source),'config':str(CONFIG),'config_sha256':sha(CONFIG),'weights':str(WEIGHTS),'weights_sha256':sha(WEIGHTS),'helper':str(HELPER),'helper_sha256':sha(HELPER),'code_sha256':sha(__file__),'cases':len(rows),'images':len(unique),'rule':'Co-DETR same class score >= .50 and candidate IoU >= .70 -> accept, otherwise unknown','entity_support_rule':'same class score >= .50 and candidate IoU >= .25; separate from localization','inference':'official init_detector/inference_detector; original RGB image, no candidate category/coordinates or GT supplied','torch':torch.__version__,'mmcv':mmcv.__version__,'mmdet':mmdet.__version__,'python':sys.version,'gpu':'0','duplicate_scope':'strict IoU>.95 handled separately; weaker cross-prediction duplicate identity not evaluated'}
    (out/'config.json').write_text(json.dumps(frozen,indent=2)+'\n')
    torch.set_num_threads(4)
    model=helper.init_detector(str(CONFIG),str(WEIGHTS),device='cuda:0')
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    (out/'effective-config.py').write_text(model.cfg.pretty_text)
    init=time.perf_counter()-start; predictions={};timings=[]
    with (out/'image-predictions.jsonl').open('x') as f:
        for path,row in unique.items():
            with Image.open(path) as im:
                assert im.size==(row['width'],row['height'])
            t=time.perf_counter()
            result=helper.inference_detector(model,path)
            torch.cuda.synchronize()
            dt=time.perf_counter()-t
            preds=helper.collect_predictions(result,tuple(model.CLASSES),.5)
            rec={'image_path':path,'image_id':row['image_id'],'image_sha256':sha(path),'width':row['width'],'height':row['height'],'wall_seconds':dt,'predictions':preds}
            f.write(json.dumps(rec)+'\n');f.flush();predictions[path]=preds;timings.append(dt)
    decisions=[dict(case_id=r['case_id'],**decide(r,predictions[r['image_path']])) for r in rows]
    (out/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    wall=time.perf_counter()-start
    receipt={'status':'complete','cases':len(rows),'images':len(unique),'wall_seconds':wall,'initialization_seconds':init,'allocated_gpu_hours':wall/3600,'max_memory_allocated_bytes':torch.cuda.max_memory_allocated(),'image_latencies':timings,'hot_mean_seconds_per_image':sum(timings[1:])/max(1,len(timings)-1),'amortized_seconds_per_candidate':sum(timings)/len(rows),'timing_scope':'sequential image forwards; images shared by candidates reused; batch throughput not measured','raw_predictions_sha256':sha(out/'image-predictions.jsonl'),'decisions_sha256':sha(out/'decisions.json'),'config_sha256':sha(out/'config.json')}
    (out/'runtime.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
