"""Real media/prompt and frozen call-budget admission; no model loads."""
from pathlib import Path
from probes.owner_successor_scale import data as d
from src.qwen.runtime_loading import QwenLoadOptions,load_qwen_components_from_options

p=d.ROOT/'slice-packet-v2.json'
packet=d.e.read(p);pool=d.e.read(packet['pool']['path'])
assert d.e.binding(packet['pool']['path'])==packet['pool']
assert d.e.binding(Path(d.__file__))==packet['producer']
assert len(pool['image_ids'])==len(set(pool['image_ids']))==4096
assert not set(pool['image_ids']) & set(pool['excluded_ids'])
assert len(pool['reused_ids'])==330 and len(pool['new_ids'])==3766
assert len(packet['call_ids'])==len(set(packet['call_ids']))==4
qwen=load_qwen_components_from_options(QwenLoadOptions(base_model=pool['config']['model']['base_model'],dtype='fp32',attn_implementation='sdpa',load_model=False))
qwen.processor.image_processor.do_resize=False
items={x['image_id']:x for x in pool['items']}
projections=[]
for image_id in packet['image_ids']:
    item=items[image_id]
    frozen,batch=d.materialize(item,pool['config'],qwen,device='cpu')
    assert frozen['image_id']==image_id
    projections.append({'image_id':image_id,'source':item['source'],'prompt_tokens':len(batch.prompt_token_ids[0]),'media_sha256':batch.media_sha256[0],'grid':list(batch.image_grids[0])})
    if item['source']=='reused_n16':
        jobs,_=d.nominate(frozen,item['natural'],qwen.tokenizer)
        expected=[j for j in packet['conditional_jobs'] if j['image_id']==image_id]
        assert all(j in jobs for j in expected)
        assert all(len(j['h_token_ids'])+len(j['c_token_ids'])+j['remaining_budget']==3084 for j in expected)
d.e.publish(d.ROOT/'slice-cpu-check-v2.json',{'status':'passed','packet':d.e.binding(p),'producer':packet['producer'],'pool_images':4096,'reused_rows':330,'new_rows':3766,'call_count':4,'projections':projections,'model_loads':0,'model_forwards':0})
print('CPU real4 media projections, frozen4096/330reuse/3766new and exact4 calls passed')
