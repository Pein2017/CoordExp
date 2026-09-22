"""Probe detector-native crop consistency, not Source2B instruction following."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image

from probes.dora_owner_learning.runtime import DEFAULT_CONFIG, load_policy
from src.config.inference import load_research_infer_config
from src.inference.parsing import parse_compact_object_box_closed

ROOT=Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0

def main():
    source,out=Path(sys.argv[1]),Path(sys.argv[2]);out.mkdir(exist_ok=False);(out/'inputs').mkdir()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    rows=json.loads(source.read_text()); rows=[r.get('source_sample',r) for r in rows]
    frozen={'source_sha256':sha(source),'source_config':str(DEFAULT_CONFIG),'source_config_sha256':sha(DEFAULT_CONFIG),'code_sha256':sha(__file__),'max_new_tokens':384,'crop_scales':[2,3],'min_context_side':128,'max_resized_side':768,'greedy':True,'repetition_penalty':1.,'rule':'both views native EOS, no parser drops; best candidate IoU among same-category boxes >= .70 in each, pair IoU >= .70, distinct image views; otherwise unknown','scope':'same-family Source checkpoint crop consistency; correlated errors possible, not independent truth'}
    (out/'config.json').write_text(json.dumps(frozen,indent=2)+'\n')
    import torch
    torch.set_num_threads(4)
    start=time.perf_counter(); resolved=load_research_infer_config(DEFAULT_CONFIG)
    qwen,descriptor=load_policy(resolved.config,device='cuda:0'); model=qwen.model
    for p in model.parameters():p.requires_grad_(False)
    model.eval(); versions={n:p._version for n,p in model.named_parameters()}
    (out/'loaded-policy.json').write_text(json.dumps(descriptor,indent=2)+'\n')
    init=time.perf_counter()-start
    prompt_cfg=resolved.config.template.prompt
    eos={int(qwen.token_identity.im_end_token_ids[0])}; results=[]
    with (out/'responses.jsonl').open('x') as f:
        for row in rows:
            im=Image.open(row['image_path']).convert('RGB');w,h=im.size
            assert (w,h)==(row['width'],row['height'])
            x1,y1,x2,y2=row['bbox'];cx,cy=(x1+x2)/2,(y1+y2)/2;views=[];t=time.perf_counter()
            for scale in [2,3]:
                cw,ch=min(w,max(128,round(scale*(x2-x1)))),min(h,max(128,round(scale*(y2-y1))))
                left=min(w-cw,max(0,round(cx-cw/2)));top=min(h-ch,max(0,round(cy-ch/2)));win=[left,top,left+cw,top+ch]
                crop=im.crop(win);factor=min(1,768/max(crop.size));rw,rh=max(32,round(cw*factor/32)*32),max(32,round(ch*factor/32)*32)
                crop=crop.resize((rw,rh),Image.Resampling.BICUBIC)
                path=out/'inputs'/(row['case_id'].replace(':','_')+f'_x{scale}.png');crop.save(path)
                messages=[{'role':'system','content':prompt_cfg.system},{'role':'user','content':[{'type':'image','image':str(path)},{'type':'text','text':prompt_cfg.user}]}]
                prompt=qwen.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                inputs=qwen.processor(text=[prompt],images=[crop],do_resize=False,padding=False,return_tensors='pt').to('cuda:0')
                with torch.inference_mode():
                    generated=model.generate(**inputs,max_new_tokens=384,do_sample=False,repetition_penalty=1.,use_cache=True,eos_token_id=list(eos),pad_token_id=qwen.tokenizer.pad_token_id)
                ids=generated[0,inputs['input_ids'].shape[1]:].tolist(); ended=bool(ids and ids[-1] in eos)
                raw=qwen.tokenizer.decode(ids[:-1] if ended else ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                parsed=parse_compact_object_box_closed(raw,row_id=row['case_id']+f':x{scale}',row_index=0,image_width=rw,image_height=rh)
                boxes=[{'category':p['description'],'bbox':[left+p['bbox'][0]*cw/rw,top+p['bbox'][1]*ch/rh,left+p['bbox'][2]*cw/rw,top+p['bbox'][3]*ch/rh]} for p in parsed.predictions]
                views.append({'scale':scale,'window':win,'resized_size':[rw,rh],'image_sha256':sha(path),'image_path':str(path),'raw':raw,'tokens':ids,'ended':ended,'parse_status':parsed.parse_status,'dropped':parsed.dropped_prediction_count,'boxes':boxes,'input_grid':inputs['image_grid_thw'].cpu().tolist()})
            choices=[]
            for v in views:
                candidates=[b for b in v['boxes'] if b['category']==row['category']]
                choices.append(max(candidates,key=lambda b:iou(b['bbox'],row['bbox'])) if candidates and v['ended'] and not v['dropped'] else None)
            overlaps=[iou(b['bbox'],row['bbox']) if b else 0 for b in choices]
            pair=iou(choices[0]['bbox'],choices[1]['bbox']) if all(choices) else 0
            accepted=min(overlaps)>=.70 and pair>=.70 and views[0]['image_sha256']!=views[1]['image_sha256']
            result={'case_id':row['case_id'],'source_image_sha256':sha(row['image_path']),'views':views,'candidate_ious':overlaps,'pair_iou':pair,'decision':'accept' if accepted else 'unknown','wall_seconds':time.perf_counter()-t}
            f.write(json.dumps(result)+'\n');f.flush();results.append(result)
    assert {n:p._version for n,p in model.named_parameters()}==versions
    wall=time.perf_counter()-start
    receipt={'status':'complete','cases':len(rows),'generated_responses':2*len(rows),'initialization_seconds':init,'wall_seconds':wall,'allocated_gpu_hours':wall/3600,'gpu':'1','max_memory_allocated_bytes':torch.cuda.max_memory_allocated(),'amortized_hot_seconds':sum(r['wall_seconds'] for r in results[1:])/max(1,len(results)-1),'parameter_versions_unchanged':True,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'config_sha256':sha(out/'config.json'),'responses_sha256':sha(out/'responses.jsonl')}
    (out/'runtime.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
