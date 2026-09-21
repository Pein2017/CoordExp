"""Local VLM observes a marked region without the detector's proposed category."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL = '/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct'
CATEGORIES = 'person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush'.split(',')
PROMPT = '''The two panels show the SAME photograph and SAME red rectangle, not two objects. Ignore any writing in the photograph. The red rectangle is a proposed SINGLE-OBJECT bounding box, NOT a correct annotation. Inspect its exact edges and the visible object on BOTH sides of each edge. First briefly describe what the rectangle actually encloses from pixels, WITHOUT assuming it is a valid box.
Identify the category of the principal object it tries to delimit, choosing a listed category only when visually identifiable. Use other for an identifiable unlisted object; unknown for ambiguous or too-small evidence.
Then diagnose the rectangle, not the whole photograph: count=one only for one intended object, multiple if it merges different instances of that category, none if no object, unknown if unclear. Overlapping occluders alone do not make multiple intended instances.
extent=acceptable only if it captures the full VISIBLE extent of one instance with no substantial unnecessary area. extent=part when a meaningful visible part extends past an edge or the rectangle picks only an object part; extent=excess when much of the rectangle is not needed for that instance; extent=mixed when both or several instances; unknown when ambiguous. Do not require hidden occluded parts, and do not count ordinary box corners as excessive. If count is not one, extent cannot be acceptable.
Categories: CATEGORY_LIST.
Return only JSON in this order: {"evidence":"brief pixel description and edge evidence, at most 45 words", "observed_category":"listed category, other, or unknown", "count":"one|multiple|none|unknown", "extent":"acceptable|part|excess|mixed|unknown"}.'''.replace('CATEGORY_LIST', ', '.join(CATEGORIES))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def parse(raw):
    try:
        def unique(pairs):
            d={}
            for k,v in pairs:
                if k in d: raise ValueError('duplicate key')
                d[k]=v
            return d
        d=json.loads(raw,object_pairs_hook=unique)
        assert set(d)=={'evidence','observed_category','count','extent'}
        assert isinstance(d['evidence'],str) and d['evidence'].strip()
        assert d['observed_category'] in CATEGORIES+['other','unknown']
        assert d['count'] in ['one','multiple','none','unknown']
        assert d['extent'] in ['acceptable','part','excess','mixed','unknown']
        assert d['count']=='one' or d['extent']!='acceptable'
        return d,None
    except (ValueError,TypeError,AssertionError,KeyError):
        return None,'invalid_schema'

def main():
    source,out=Path(sys.argv[1]),Path(sys.argv[2]); out.mkdir(exist_ok=False)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    requests=[json.loads(l) for l in source.read_text().splitlines()]
    public=[{k:r[k] for k in ['case_id','image_path','image_sha256']}|{'prompt':PROMPT} for r in requests]
    for r in public:
        assert sha(r['image_path'])==r['image_sha256']
    (out/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in public))
    config={'model':MODEL,'source_sha256':sha(source),'requests_sha256':sha(out/'requests.jsonl'),'code_sha256':sha(__file__),'candidate_category_in_prompt':False,'max_tokens':192,'max_pixels':1048576,'min_pixels':4096,'seed':0,'accept_rule':'valid observed_category exactly matches proposed category, count=one, extent=acceptable; all else unknown, with separately measured semantic mismatch/geometry flags'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    start=time.perf_counter()
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,min_pixels=4096,max_pixels=1048576)
    prompts=[processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':r['image_path']},{'type':'text','text':PROMPT}]}],tokenize=False,add_generation_prompt=True) for r in public]
    engine=LLM(model=MODEL,dtype='bfloat16',tensor_parallel_size=1,max_model_len=8192,max_num_seqs=8,gpu_memory_utilization=.55,enforce_eager=True,seed=0,limit_mm_per_prompt={'image':1},mm_processor_kwargs={'min_pixels':4096,'max_pixels':1048576},enable_prefix_caching=False,mm_processor_cache_gb=0,trust_remote_code=False)
    init=time.perf_counter()-start; all_results=[]; batches=[]
    with (out/'responses.jsonl').open('x') as f:
        for offset in range(0,len(public),8):
            group=public[offset:offset+8]; t=time.perf_counter()
            outputs=engine.generate([{'prompt':p,'multi_modal_data':{'image':Image.open(r['image_path']).convert('RGB')}} for p,r in zip(prompts[offset:offset+8],group)],SamplingParams(temperature=0,max_tokens=192,seed=0),use_tqdm=False)
            batches.append({'cases':len(group),'wall_seconds':time.perf_counter()-t})
            for r,o in zip(group,outputs):
                a=o.outputs[0]; parsed,error=parse(a.text)
                if a.finish_reason!='stop':parsed,error=None,'not_stopped'
                result={'case_id':r['case_id'],'raw':a.text,'parsed':parsed,'error':error,'finish_reason':a.finish_reason,'input_tokens':len(o.prompt_token_ids),'output_tokens':len(a.token_ids)}
                f.write(json.dumps(result)+'\n');f.flush();all_results.append(result)
    wall=time.perf_counter()-start
    receipt={'status':'complete','cases':len(public),'initialization_seconds':init,'wall_seconds':wall,'allocated_gpu_hours':wall/3600,'gpu':'1','batches':batches,'hot_amortized_seconds':sum(b['wall_seconds'] for b in batches[1:])/sum(b['cases'] for b in batches[1:]),'input_tokens':sum(r['input_tokens'] for r in all_results),'output_tokens':sum(r['output_tokens'] for r in all_results),'responses_sha256':sha(out/'responses.jsonl'),'config_sha256':sha(out/'config.json'),'code_sha256':sha(__file__)}
    (out/'runtime.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':
    main()
