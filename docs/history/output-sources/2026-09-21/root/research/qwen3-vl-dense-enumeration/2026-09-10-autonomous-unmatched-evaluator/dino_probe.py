"""Task-local two-view independent detector consistency; raw predictions retained."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models/grounding-dino-tiny'
REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def iou(a, b):
    inter = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.

def select(candidate, detections):
    eligible = [d for d in detections if d['score'] >= .30]
    if not eligible:
        return None
    return max(eligible, key=lambda d:iou(candidate['bbox'],d['bbox']))

def decide(candidate, views):
    selected = [select(candidate, v['detections']) for v in views]
    overlaps = [iou(candidate['bbox'],d['bbox']) if d else 0 for d in selected]
    pair = iou(selected[0]['bbox'],selected[1]['bbox']) if all(selected) else 0
    accepted = min(overlaps) >= .70 and pair >= .70
    return {'decision':'accept' if accepted else 'unknown', 'entity_category':'supported_candidate' if accepted else 'unknown', 'geometry':'acceptable_candidate' if accepted else 'unknown', 'candidate_ious':overlaps, 'pair_iou':pair, 'selected':selected}

def main():
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    out.mkdir(exist_ok=False)
    data = json.loads(source.read_text()) if source.suffix=='.json' else [json.loads(l) for l in source.read_text().splitlines()]
    rows = [r.get('source_sample',r) for r in data]
    config = {'source':str(source), 'source_sha256':sha(source), 'model':str(MODEL), 'model_revision':REVISION, 'weights_sha256':sha(MODEL/'model.safetensors'), 'dtype':'float32', 'views':['full','padded_context'], 'context':'candidate center, 2x candidate width/height, minimum 128px; no candidate edges drawn', 'text':'lowercase candidate category followed by a dot', 'postprocess_score_threshold':.10, 'text_threshold':.25, 'rule':'best candidate-IoU among detections with score >= .30 in EACH view; both candidate IoUs >= .70 and cross-view IoU >= .70 -> accept, else unknown', 'code_sha256':sha(__file__)}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    start=time.perf_counter()
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    torch.set_num_threads(4)
    proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(MODEL,local_files_only=True).to('cuda').eval()
    init=time.perf_counter()-start
    results=[]
    with (out/'responses.jsonl').open('x') as f:
        for row in rows:
            im=Image.open(row['image_path']).convert('RGB');w,h=im.size
            assert im.size==(row['width'],row['height'])
            x1,y1,x2,y2=row['bbox']; cx,cy=(x1+x2)/2,(y1+y2)/2
            cw,ch=min(w,max(128,round(2*(x2-x1)))),min(h,max(128,round(2*(y2-y1))))
            left=min(w-cw,max(0,round(cx-cw/2)));top=min(h-ch,max(0,round(cy-ch/2)))
            views=[]; t=time.perf_counter()
            for form,win in [('full',[0,0,w,h]),('padded_context',[left,top,left+cw,top+ch])]:
                crop=im.crop(win)
                inputs=proc(images=crop,text=row['category'].lower()+'.',return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    outputs=model(**inputs)
                post=proc.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=.10,text_threshold=.25,target_sizes=[crop.size[::-1]])[0]
                ds=[]
                for box,score,label in zip(post['boxes'].cpu().tolist(),post['scores'].cpu().tolist(),post['text_labels']):
                    mapped=[box[0]+win[0],box[1]+win[1],box[2]+win[0],box[3]+win[1]]
                    assert all(__import__('math').isfinite(v) for v in mapped+[score])
                    ds.append({'bbox':mapped,'score':score,'label':label})
                views.append({'form':form,'window':win,'detections':ds})
            torch.cuda.synchronize()
            result={'case_id':row['case_id'],'source_image_sha256':sha(row['image_path']),'views':views,'wall_seconds':time.perf_counter()-t,**decide(row,views)}
            f.write(json.dumps(result)+'\n');f.flush();results.append(result)
    wall=time.perf_counter()-start
    receipt={'status':'complete','cases':len(results),'responses':2*len(results),'initialization_seconds':init,'wall_seconds':wall,'allocated_gpu_hours':wall/3600,'gpu':'0','torch':torch.__version__,'transformers':transformers.__version__,'max_memory_allocated_bytes':torch.cuda.max_memory_allocated(),'timing_scope':'sequential per candidate, two forwards, includes processing; allocation elapsed not kernel time','amortized_hot_seconds':sum(r['wall_seconds'] for r in results[1:])/max(1,len(results)-1),'config_sha256':sha(out/'config.json'),'responses_sha256':sha(out/'responses.jsonl'),'code_sha256':sha(__file__)}
    (out/'runtime.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':
    main()
