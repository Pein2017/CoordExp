"""CPU checks of the fixed-group inference caller and sealed readback boundary."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from probes.full_label_fit import coord_norm as c
from src.qwen.generation import ContinuationResult
from src.qwen.native import NativeRequest


def test_sealed_request_coverage_and_input_identity():
    inputs = {i:dict(image_id=i, request_id=f'{i}:greedy:0', prompt_token_ids=[1],
                    media_sha256='media', image_grid_thw=[1,2,2]) for i in range(18)}
    producer = dict(kind='fixed_checkpoint_coord_norm_inference', condition='off')
    rows = [c.online.seal(dict(row, token_ids=[9], text='x', generated_tokens=1,
                             stop_reason='im_end'), producer) for row in inputs.values()]
    c.validate_records(rows, inputs, producer)
    bad = copy.deepcopy(rows);bad[0]['prompt_token_ids']=[2]
    bad[0]=c.online.seal(bad[0],producer)
    with pytest.raises(AssertionError):c.validate_records(bad,inputs,producer)
    bad=copy.deepcopy(rows);bad[0]['request_id']=bad[1]['request_id']
    bad[0]=c.online.seal(bad[0],producer)
    with pytest.raises(AssertionError):c.validate_records(bad,inputs,producer)
    with pytest.raises(AssertionError):c.validate_records(rows[:-1],inputs,producer)


@pytest.mark.parametrize('rank',[0,7])
def test_actual_run_has_two_fixed_group_reads_and_no_model_or_optimizer(tmp_path, monkeypatch, rank):
    import torch
    from src.qwen import vllm_rollout as v
    groups=[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13,14],[15,16,17]]
    inputs={i:dict(image_id=i,request_id=f'{i}:greedy:0',prompt_token_ids=[1],
                  media_sha256='m',image_grid_thw=[1,2,2]) for i in range(18)}
    contract=dict(groups=groups,weight_identity='fixed',base_model='/base',
        base_config_sha256='cfg',tokenizer_sha256='tok',coordinate_ids=list(range(1000)),
        checkpoint='/checkpoint',conditions=['off','median'],source={'revision':'cpu'})
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract))
    q=SimpleNamespace(model=None,base_model_path=Path('/base'),base_config_sha256='cfg',
        tokenizer_sha256='tok',tokenizer=SimpleNamespace(pad_token_id=0,
          convert_tokens_to_ids=lambda x:int(x[8:-2]) if x.startswith('<|coord_') else 1001,
          decode=lambda *a,**kw:'<|im_end|>'),to_artifact_dict=lambda:{'load_model':False})
    calls=[]
    class Engine:
        def __init__(self,**kwargs):
            self.receipts=[];calls.append(('init',kwargs));self.mode=None
            self.device_request={};self.startup={}
        def __enter__(self):return self
        def __exit__(self,*args):calls.append(('closed',))
        def configure_coordinate_output_norm(self,mode,ids,**kwargs):
            self.mode=mode;calls.append(('policy',mode));return {'mode':mode,'identity':'fixed'}
        def generate(self,requests,**kwargs):
            calls.append(('generate',[r.request_id for r in requests]))
            self.receipts.append({'coordinate_output_norm':dict(mode=self.mode,identity='fixed',calls=1,coordinate_tokens=1000,coordinate_ids=list(range(1000)),first_call=dict(scaling_active=self.mode=='median',non_coordinate_unchanged=True,changed_coordinates=1),norm_min=1.,norm_max=2.,median_norm=1.,factor_min=.5,factor_max=1.)})
            return tuple(ContinuationResult(r.request_id,(1001,),'im_end',
                SimpleNamespace(raw_logprobs=(-.1,),processed_logprobs=(-.1,))) for r in requests)
    def gather(target,value):
        if isinstance(value,dict):target[:]=[{}]*8;return
        producer=value[0]['producer'];target[:]=[]
        for r,group in enumerate(groups):
            target.append([c.online.seal(dict(inputs[i],token_ids=[1001],text='<|im_end|>',
                generated_tokens=1,stop_reason='im_end'),producer) for i in group])
    monkeypatch.setenv('RANK',str(rank));monkeypatch.setenv('LOCAL_RANK',str(rank));monkeypatch.setenv('WORLD_SIZE','8')
    with patch.object(c,'validate_contract',return_value=inputs), \
         patch('probes.rollout_row_credit.frontend',return_value=q), \
         patch.object(c.online,'vllm_requests',side_effect=lambda q,inp,group:[NativeRequest(inp[i]['request_id'],'','') for i in group]), \
         patch.object(v,'VllmDoraRollout',Engine),patch.object(v,'validate_device_assignments'), \
         patch('torch.cuda.set_device'),patch('torch.distributed.init_process_group'), \
         patch('torch.distributed.destroy_process_group'),patch('torch.distributed.barrier'), \
         patch('torch.distributed.all_gather_object',side_effect=gather):
        c.run(path,tmp_path/'native')
    wanted=[inputs[i]['request_id'] for i in groups[rank]]
    assert [call[1] for call in calls if call[0]=='generate']==[wanted,wanted]
    assert [call[1] for call in calls if call[0]=='policy']==['off','median']
    assert sum(call[0]=='init' for call in calls)==1 and calls[-1][0]=='closed'
    complete=c.load(tmp_path/f'native/rank-{rank}/complete.json')
    assert complete['HF_forwards']==complete['optimizer_steps']==0


def test_policy_witness_rejects_wrong_identity_ids_and_inactive_treatment():
    contract=dict(weight_identity='fixed',coordinate_ids=list(range(1000)))
    policy=dict(mode='median',identity='fixed',calls=1,coordinate_tokens=1000,
        coordinate_ids=list(range(1000)),norm_min=1.,norm_max=2.,median_norm=1.,
        factor_min=.5,factor_max=1.,first_call=dict(scaling_active=True,
        non_coordinate_unchanged=True,changed_coordinates=1))
    c.validate_policy(policy,'median',contract)
    for mutate in (lambda p:p.update(identity='other'), lambda p:p['coordinate_ids'].reverse(),
                   lambda p:p['first_call'].update(scaling_active=False),lambda p:p.update(factor_min=0)):
        wrong=copy.deepcopy(policy);mutate(wrong)
        with pytest.raises(AssertionError):c.validate_policy(wrong,'median',contract)


def test_actual_persisted_score_rejects_resigned_wrong_group(tmp_path):
    # Exercise the real saved-output scorer and consumer, without a model load.
    inputs={r['image_id']:r for r in c.load(c.online.INPUTS)}
    image_ids=sorted(inputs);groups=[image_ids[r::8] for r in range(8)]
    label=Path(__file__).resolve().parents[2]/'research/experiments/2026-10-02-full-label-self-rollout-fit/inputs/full-labels.json'
    contract=dict(image_ids=image_ids,groups=groups,weight_identity='fixed',source='cpu',
                  coordinate_ids=list(range(1000)),conditions=['off','median'],label_path=str(label),reference_run=str(tmp_path/'reference'))
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract));output=tmp_path/'native'
    reference=tmp_path/'reference/rollout-16'
    for condition in ['off','median','reference']:
        root=reference if condition=='reference' else output/condition
        producer=dict(kind='fixed_checkpoint_coord_norm_inference',condition=condition,
            weight_identity='fixed',source='cpu',contract_sha256=c.online.p.digest(path))
        for rank,group in enumerate(groups):
            directory=root/f'rank-{rank}';directory.mkdir(parents=True)
            artifacts={}
            for index,i in enumerate(group):
                row=c.online.seal(dict(inputs[i],token_ids=[9],text='<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_0|><|coord_0|><|coord_999|><|coord_999|><|box_end|><|im_end|>',generated_tokens=1,
                    stop_reason='im_end',generation_rank=rank,generation_batch_index=index),producer)
                if condition=='reference':row=c.online.seal(dict(row,arm='greedy'),producer)
                c.write(directory/f'{i}.json',row);artifacts[f'{i}.json']=c.online.p.digest(directory/f'{i}.json')
            policy=dict(mode=condition,identity='fixed',calls=1,coordinate_tokens=1000,
                coordinate_ids=list(range(1000)),norm_min=1.,norm_max=2.,median_norm=1.,factor_min=.5,factor_max=1.,
                first_call=dict(scaling_active=condition=='median',non_coordinate_unchanged=True,changed_coordinates=1))
            c.write(directory/'complete.json',dict(status='complete',image_ids=group,producer=producer,policy=policy,artifacts=artifacts))
    for rank in range(8):
        directory=output/f'rank-{rank}';directory.mkdir()
        c.write(directory/'complete.json',dict(status='complete',rank=rank,weights_unchanged_identity='fixed',
            optimizer_steps=0,HF_forwards=0,operations=[dict(identity='fixed',coordinate_output_norm={'mode':m}) for m in ['off','off','median','median']]))
    with patch.object(c,'validate_contract',return_value=inputs):c.score(path,output)
    assert c.load(output/'readback.json')['requests']==36
    bad=output/'median/rank-0'/f'{groups[0][0]}.json';row=c.load(bad);row['generation_rank']=7
    row=c.online.seal(row,row['producer']);bad.write_text(json.dumps(row))
    receipt=bad.parent/'complete.json';saved=c.load(receipt);saved['artifacts'][bad.name]=c.online.p.digest(bad);receipt.write_text(json.dumps(saved))
    # Re-sign the inventory too: reject semantic wrong dispatch, not just a checksum.
    (output/'median/frozen.json').unlink()
    with patch.object(c,'validate_contract',return_value=inputs),pytest.raises(AssertionError):c.score(path,output)
