"""Repair only two donor record envelopes; preserve every scientific input."""
from pathlib import Path
import copy
import json
import os
import sys

ROOT=Path(__file__).parent
sys.path.insert(0,'/data/CoordExp/.worktrees/research-probes')
from probes.dora_owner_learning.candidate_opportunity import file_hash
from src.data.examples import raw_example_from_jsonl_row
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
from src.qwen.native import prepare_native_inputs
from probes.source_rweak_row_cross.run import build_requests

if __name__=='__main__':
    old=json.loads((ROOT/'packet.json').read_text());new=copy.deepcopy(old)
    source=Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/train.coord.jsonl')
    rows={}
    for line in source.open():
        row=json.loads(line)
        if row['image_id'] in (3125,3514):rows[row['image_id']]=row
        if len(rows)==2:break
    jsonl=Path(old['config']['data']['input_jsonl']);failures=[]
    for rank in (3,7):
        previous=old['cases'][rank]['donor_case'];record=previous['input_record']
        try:raw_example_from_jsonl_row(record,jsonl_path=jsonl,row_number=previous['row_index']+1,raw_line=json.dumps(record))
        except Exception as error:failures.append(dict(rank=rank,before_error=repr(error)))
        else:raise AssertionError('expected old donor-envelope failure absent')
        donor=new['cases'][rank]['donor_case'];record=copy.deepcopy(rows[new['cases'][rank]['donor_image_id']])
        record['images']=[os.path.relpath(donor['image_path'],jsonl.parent)]
        donor['input_record']=record
        raw=raw_example_from_jsonl_row(record,jsonl_path=jsonl,row_number=donor['row_index']+1,raw_line=json.dumps(record))
        assert str(raw.example_id)==donor['row_id'] and str(raw.image.path)==donor['image_path']
        unchanged=copy.deepcopy(new['cases'][rank]);unchanged['donor_case']['input_record']=previous['input_record']
        assert unchanged==old['cases'][rank]
    assert all(new['cases'][i]==old['cases'][i] for i in (0,1,2,4,5,6))
    # Exercise the actual native caller on CPU, including prompt and image tensors.
    qwen=load_qwen_components_from_options(QwenLoadOptions(base_model=old['model']['base_model_path'],
        dtype='fp32',attn_implementation='sdpa',patch_embed_linearization='enabled',load_model=False))
    import torch
    checked=[]
    for rank in (3,7):
        case=new['cases'][rank];requests,_=build_requests(qwen,new['config'],[case['donor_case']])
        batch=prepare_native_inputs(qwen.processor,requests,device=torch.device('cpu'),record_media_identity=True)
        assert list(batch.prompt_token_ids[0])==case['prompt_token_ids']
        assert list(batch.image_grids[0])==case['source_case']['image_plan']['observed_image_grid_thw']
        checked.append(dict(rank=rank,request_id=batch.request_ids[0],grid=list(batch.image_grids[0]),media_sha256=batch.media_sha256[0]))
    new['technical_retry_of']=dict(packet_sha256=file_hash(ROOT/'packet.json'),failed_ranks=[3,7],
        reason='Only donor input envelopes: relative image reference and existing canonical coord-token GT schema; same image bytes/prompt/grid/prefix/jobs/model')
    new['source_files'].update({str(source):file_hash(source),str(Path(__file__)):file_hash(__file__),
        str(ROOT/'packet.json'):file_hash(ROOT/'packet.json')})
    with (ROOT/'packet-repair-01.json').open('x') as out:json.dump(new,out,indent=2,sort_keys=True);out.write('\n')
    receipt=dict(before_failures=failures,after_cpu_native_requests=checked,model_loads=0,model_forwards=0,
        unchanged_scientific_inputs=True,packet_sha256=file_hash(ROOT/'packet-repair-01.json'))
    with (ROOT/'repair-preflight.json').open('x') as out:json.dump(receipt,out,indent=2,sort_keys=True);out.write('\n')
    print(json.dumps(receipt,indent=2))
