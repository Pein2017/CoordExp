import contextlib,io,json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
root=Path('/data/CoordExp/outputs/infra_base')
runs=[]
for folder,model,norm,budget in [('untied-axis-val200-3084-20260918','untied+axis',False,3084),('untied-axis-val200-normalized-20260918','untied+axis',True,3084),('tied-val200-normalized-20260918','tied',True,3084),('untied-axis-val200-20260918','untied+axis',False,512)]:
 for rp in ['rp100','rp110']:runs.append((root/folder/rp/'evaluation',model,norm,budget,1.0 if rp=='rp100' else 1.1))
runs.append((Path('/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/inference/four-coordinate-xy-val200/evaluation'),'tied',False,512,1.1))
results=[]
for d,model,norm,budget,rp in runs:
 m=json.loads((d/'metrics.json').read_text())
 with contextlib.redirect_stdout(io.StringIO()):
  gt=COCO(str(d/'coco_gt.json'));dt=gt.loadRes(str(d/'coco_predictions.json'));e=COCOeval(gt,dt,'bbox');e.evaluate();e.accumulate();e.summarize()
 assert abs(float(e.stats[0])-m['bbox_AP'])<1e-10
 idx=int(np.flatnonzero(np.isclose(e.params.iouThrs,.5))[0]);eligible=tp=0
 for row in e.evalImgs:
  if row is None or row['aRng']!=e.params.areaRng[0] or row['maxDet']!=100:continue
  valid=np.logical_not(row['gtIgnore']);eligible+=int(valid.sum());tp+=int(((row['gtMatches'][idx]>0)&valid).sum())
 assert eligible==1600
 results.append(dict(model=model,normalized=norm,rp=rp,max_new_tokens=budget,mAP=m['bbox_AP'],FN50=eligible-tp,TP50=tp,GT=eligible,evaluation_dir=str(d)))
results.sort(key=lambda x:x['mAP'],reverse=True)
out=Path(__file__).parent/'summary.json';out.write_text(json.dumps({'fn_policy':'COCOeval bbox, category-aware, IoU .50, area all, maxDets100 per image/category, no additional score threshold','results':results},indent=2))
for r in results:print({k:v for k,v in r.items() if k!='evaluation_dir'})
