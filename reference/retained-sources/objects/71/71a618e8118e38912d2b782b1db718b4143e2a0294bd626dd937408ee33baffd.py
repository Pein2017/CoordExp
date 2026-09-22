"""Exact development point-observation protocol, applied to an unlabeled subset."""
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'reground-dev-v1'
MODEL='/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    source,out=Path(sys.argv[1]),Path(sys.argv[2]);out.mkdir(exist_ok=False);(out/'inputs').mkdir()
    rows=[json.loads(l) for l in source.read_text().splitlines()]
    spec=importlib.util.spec_from_file_location('frozen_reground',BASE/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    template=json.loads((BASE/'requests.jsonl').read_text().splitlines()[0])['prompt']
    requests=[]
    for row in rows:
        im=Image.open(row['image_path']).convert('RGB');w,h=im.size
        assert (w,h)==(row['width'],row['height'])
        x1,y1,x2,y2=row['bbox'];cx,cy=(x1+x2)/2,(y1+y2)/2
        cw,ch=max(1,w//2),max(1,h//2)
        left=min(w-cw,max(0,round(cx-cw/2)));top=min(h-ch,max(0,round(cy-ch/2)))
        for form,win in [('full',[0,0,w,h]),('fixed_context',[left,top,left+cw,top+ch])]:
            l,t,r,b=win;point=[round(1000*(cx-l)/(r-l)),round(1000*(cy-t)/(b-t))]
            prompt=re.sub(r'^The point \(\d+, \d+\)',f'The point ({point[0]}, {point[1]})',template,count=1)
            assert prompt.startswith(f'The point ({point[0]}, {point[1]})')
            path=out/'inputs'/(row['case_id'].replace(':','_')+'_'+form+'.png');im.crop(win).save(path)
            requests.append({'case_id':row['case_id'],'form':form,'image_path':str(path),'image_sha256':sha(path),'source_image_sha256':sha(row['image_path']),'window':win,'point':point,'prompt':prompt})
    (out/'requests.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in requests))
    config={'model':MODEL,'max_tokens':128,'max_model_len':8192,'max_pixels':1048576,'min_pixels':65536,'seed':0,'source_sha256':sha(source),'requests_sha256':sha(out/'requests.jsonl'),'code_sha256':sha(__file__),'frozen_parser_sha256':sha(BASE/'run.py'),'development_prompt_template':str(BASE/'requests.jsonl'),'development_prompt_sha256':sha(BASE/'requests.jsonl'),'no_reference_or_candidate_category_in_prompt':True,'requests':len(requests)}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    start=time.perf_counter();results=[];batches=[];init=0
    if requests:
        from transformers import AutoProcessor
        from vllm import LLM,SamplingParams
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,min_pixels=65536,max_pixels=1048576)
        prompts=[processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':q['image_path']},{'type':'text','text':q['prompt']}]}],tokenize=False,add_generation_prompt=True) for q in requests]
        engine=LLM(model=MODEL,dtype='bfloat16',tensor_parallel_size=1,max_model_len=8192,max_num_seqs=8,gpu_memory_utilization=.55,enforce_eager=True,seed=0,limit_mm_per_prompt={'image':1},mm_processor_kwargs={'min_pixels':65536,'max_pixels':1048576},enable_prefix_caching=False,mm_processor_cache_gb=0,trust_remote_code=False)
        init=time.perf_counter()-start
    with (out/'responses.jsonl').open('x') as f:
        for offset in range(0,len(requests),8):
            group=requests[offset:offset+8];t=time.perf_counter()
            outputs=engine.generate([{'prompt':p,'multi_modal_data':{'image':Image.open(q['image_path']).convert('RGB')}} for p,q in zip(prompts[offset:offset+8],group)],SamplingParams(temperature=0,max_tokens=128,seed=0),use_tqdm=False)
            batches.append({'requests':len(group),'wall_seconds':time.perf_counter()-t})
            for q,o in zip(group,outputs):
                a=o.outputs[0];parsed,error=old.parse(a.text,q['window'])
                if a.finish_reason!='stop':parsed,error=None,'not_stopped'
                rec={'case_id':q['case_id'],'form':q['form'],'raw':a.text,'parsed':parsed,'error':error,'finish_reason':a.finish_reason,'input_tokens':len(o.prompt_token_ids),'output_tokens':len(a.token_ids)}
                f.write(json.dumps(rec)+'\n');f.flush();results.append(rec)
    wall=time.perf_counter()-start
    receipt={'status':'complete','candidate_count':len(rows),'requests':len(results),'initialization_seconds':init,'wall_seconds':wall,'allocated_gpu_hours':wall/3600,'gpu':'1','batches':batches,'input_tokens':sum(q['input_tokens'] for q in results),'output_tokens':sum(q['output_tokens'] for q in results),'responses_sha256':sha(out/'responses.jsonl'),'config_sha256':sha(out/'config.json')}
    (out/'runtime.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
