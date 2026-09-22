import os, json, time, hashlib, re, importlib.metadata
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/final-v2/reviewed-cases.json')
MODEL = '/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-8B-Instruct'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
def parse(raw, window):
    try:
        def unique(pairs):
            d={}
            for k,v in pairs:
                if k in d: raise ValueError('duplicate key')
                d[k]=v
            return d
        x=json.loads(raw,object_pairs_hook=unique)
        assert set(x)=={'status','category','bbox'}
        assert x['status'] in ['resolved','ambiguous','none']
        assert isinstance(x['category'],str)
        if x['status']!='resolved': return None,'unresolved'
        b=x['bbox']; assert isinstance(b,list) and len(b)==4
        assert all(type(v) in [int,float] and 0<=v<=1000 for v in b)
        assert b[0]<b[2] and b[1]<b[3]
        assert x['category'].strip() and x['category'].lower() not in ['unknown','none']
        l,t,r,bot=window
        return {'category':x['category'].lower().strip(),'bbox':[l+b[0]/1000*(r-l),t+b[1]/1000*(bot-t),l+b[2]/1000*(r-l),t+b[3]/1000*(bot-t)]},None
    except (ValueError,TypeError,AssertionError,KeyError): return None,'invalid_schema'
def decide(candidate, a, b):
    if not a or not b: return {'decision':'unknown','reason':'unresolved_or_invalid'}
    pair=iou(a['bbox'],b['bbox']); scores=[iou(candidate['bbox'],x['bbox']) for x in [a,b]]
    result={'decision':'unknown','reason':'cross_view_or_candidate_disagreement','pair_iou':pair,'candidate_ious':scores}
    if a['category']!=b['category'] or pair<.75:return result
    if a['category']==candidate['category'].lower().strip() and min(scores)>=.75:
        result.update(decision='accept',reason='two_view_category_and_box_agreement')
    elif max(scores)<=.5 or a['category']!=candidate['category'].lower().strip():
        result.update(decision='repair_proposal',reason='two_view_agree_candidate_disagrees',proposal=a)
    return result

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    start=time.perf_counter(); rows=json.loads(SOURCE.read_text())
    (ROOT/'inputs').mkdir(exist_ok=True)
    requests=[]; images=[]
    for row in rows:
        s=row['source_sample']; im=Image.open(s['image_path']).convert('RGB'); w,h=im.size
        assert [w,h]==[s['width'],s['height']]
        cx=(s['bbox'][0]+s['bbox'][2])/2; cy=(s['bbox'][1]+s['bbox'][3])/2
        cw=max(1,w//2); ch=max(1,h//2)
        left=min(w-cw,max(0,round(cx-cw/2))); top=min(h-ch,max(0,round(cy-ch/2)))
        for form,win in [('full',[0,0,w,h]),('fixed_context',[left,top,left+cw,top+ch])]:
            l,t,r,bot=win; path=ROOT/'inputs'/(row['case_id'].replace(':','_')+'_'+form+'.png')
            im.crop(win).save(path)
            point=[round(1000*(cx-l)/(r-l)),round(1000*(cy-t)/(bot-t))]
            prompt=(f'The point ({point[0]}, {point[1]}) uses coordinates normalized to 0-1000 in this image. '
                'Identify the single whole physical object at that point and its tight visible bounding box. '
                'Infer its common object category independently. Do not box a collection or just a part. '
                'If the point lies on an occluder, between multiple plausible objects, or a part whose whole object cannot be resolved, use ambiguous. '
                'If no object is identifiable use none. If the object extends outside this image use ambiguous. '
                'Return only JSON: {"status":"resolved|ambiguous|none","category":"short common category or unknown",'
                '"bbox":[x1,y1,x2,y2]}. Box coordinates must be normalized 0-1000 for THIS image. '
                'Use null for bbox when not resolved.')
            requests.append({'case_id':row['case_id'],'form':form,'image_path':str(path),'image_sha256':sha(path),'source_image_sha256':sha(s['image_path']),'window':win,'point':point,'prompt':prompt})
            images.append(Image.open(path).convert('RGB'))
    (ROOT/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests))
    config={'model':MODEL,'max_tokens':128,'max_model_len':8192,'max_pixels':1048576,'min_pixels':65536,'seed':0,'rule':'exact lowercase category agreement and pair IoU>=0.75; accept iff candidate category matches and both IoUs>=0.75; repair proposal iff both IoUs<=0.5 or category differs; rest unknown','request_count':len(requests),'source_sha256':sha(SOURCE),'code_sha256':sha(__file__),'requests_sha256':sha(ROOT/'requests.jsonl')}
    dump(ROOT/'config.json',config)
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,min_pixels=65536,max_pixels=1048576)
    prompts=[processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':q['image_path']},{'type':'text','text':q['prompt']}]}],tokenize=False,add_generation_prompt=True) for q in requests]
    model=LLM(model=MODEL,dtype='bfloat16',tensor_parallel_size=1,max_model_len=8192,max_num_seqs=8,gpu_memory_utilization=.8,enforce_eager=True,seed=0,limit_mm_per_prompt={'image':1},mm_processor_kwargs={'min_pixels':65536,'max_pixels':1048576},enable_prefix_caching=False,mm_processor_cache_gb=0,trust_remote_code=False)
    init=time.perf_counter()-start; all_out=[]; batches=[]
    # First real case closes schema and coordinate-transform seam before the population.
    groups=[list(range(2))]+[list(range(i,min(i+8,len(requests)))) for i in range(2,len(requests),8)]
    with (ROOT/'responses.jsonl').open('x') as f:
        for indices in groups:
            t=time.perf_counter()
            outputs=model.generate([{'prompt':prompts[i],'multi_modal_data':{'image':images[i]}} for i in indices],SamplingParams(temperature=0,max_tokens=128,seed=0),use_tqdm=False)
            batches.append({'indices':indices,'wall_seconds':time.perf_counter()-t})
            for i,o in zip(indices,outputs):
                raw=o.outputs[0]; parsed,error=parse(raw.text,requests[i]['window'])
                if raw.finish_reason!='stop':parsed,error=None,'not_stopped'
                result={'case_id':requests[i]['case_id'],'form':requests[i]['form'],'raw':raw.text,'parsed':parsed,'error':error,'finish_reason':raw.finish_reason,'input_tokens':len(o.prompt_token_ids),'output_tokens':len(raw.token_ids)}
                f.write(json.dumps(result)+'\n');f.flush();all_out.append(result)
            if indices==[0,1]:
                dump(ROOT/'first-case-receipt.json',{'outputs':all_out,'coordinate_contract':'normalized relative to each provided image, mapped to original pixel coordinates; crop is fixed half-image, independent of candidate edges'})
                if not any(x['parsed'] for x in all_out):raise RuntimeError('first-case schema/coordinate smoke failed; stop')
    decisions=[]
    for i,row in enumerate(rows):
        a,b=all_out[2*i:2*i+2]
        decisions.append({'case_id':row['case_id'],**decide(row['source_sample'],a['parsed'],b['parsed']),'reference':{k:row[k] for k in ['category','entity','geometry']}})
    dump(ROOT/'decisions.json',decisions)
    wall=time.perf_counter()-start
    dump(ROOT/'runtime.json',{'status':'complete','wall_seconds':wall,'initialization_seconds':init,'allocated_gpu_hours':wall/3600,'gpu':'1','responses':len(all_out),'input_tokens':sum(x['input_tokens'] for x in all_out),'output_tokens':sum(x['output_tokens'] for x in all_out),'batches':batches,'versions':{p:importlib.metadata.version(p) for p in ['vllm','transformers','torch','Pillow']},'config_sha256':sha(ROOT/'config.json'),'responses_sha256':sha(ROOT/'responses.jsonl'),'code_sha256':sha(__file__),'cost_scope':'single allocated GPU elapsed time, not active kernel time; no API calls'})
if __name__=='__main__':main()
