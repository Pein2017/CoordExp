import os
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock
from uuid import UUID

import pytest

from probes import online_row_credit as online
from src.qwen.native import NativeRequest
from src.qwen.vllm_rollout import VllmDoraRollout, _generate


# Captured by the fresh spawn interpreter, before its target executes.
IMPORT_VISIBILITY = os.environ.get('CUDA_VISIBLE_DEVICES')


def cpu_spawn_worker(connection, request, *args):
    startup = {'identity': args[2], 'import_visibility': IMPORT_VISIBILITY}
    if isinstance(request, dict):
        startup['device'] = dict(requested=request, child=dict(pid=os.getpid(),ppid=os.getppid(),
            nspid=next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('NSpid:'))),
            inherited_visibility=IMPORT_VISIBILITY,effective_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
            logical_device=0,physical=request['parent']['physical'])
    connection.send({'ok': True, 'value': startup})
    connection.recv()
    connection.close()


class DeviceRoutingTest(unittest.TestCase):
    def test_fresh_spawn_import_inherits_selected_mask_and_restores_parent(self):
        import tempfile
        from src.qwen import vllm_rollout as v
        props=SimpleNamespace(uuid='11111111-1111-1111-1111-111111111111',
                              pci_domain_id=0,pci_bus_id=103,pci_device_id=0)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'7,5,3','RANK':'2'}), \
             patch('torch.cuda.get_device_properties',return_value=props),patch('torch.cuda.current_device',return_value=2), \
             patch.object(v,'_worker',cpu_spawn_worker):
            with VllmDoraRollout(base_model='/base',checkpoint='/anchor',identity='snapshot',
                                 log_path=Path(tmp)/'worker.log',device=2) as engine:
                self.assertEqual(engine.startup['import_visibility'],'3')
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'7,5,3')

    def test_start_failure_and_success_restore_absence_and_uuid_mask(self):
        import tempfile
        from src.qwen import vllm_rollout as v
        props=SimpleNamespace(uuid=str(UUID(int=1)),pci_domain_id=0,pci_bus_id=103,pci_device_id=0)
        for visibility,device in ((None,0),('7,5,3',1),('GPU-'+str(UUID(int=2))+',GPU-'+str(UUID(int=1)),1)):
            for fail in (False,True):
                context=Mock(); parent,child=Mock(),Mock();context.Pipe.return_value=(parent,child)
                env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
                if visibility is not None:env['CUDA_VISIBLE_DEVICES']=visibility
                responses=[]
                def process(*,target,args,name):
                    request=args[1];proc=Mock();proc.is_alive.return_value=False
                    def start():
                        self.assertEqual(request['device'],device)
                        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],request['physical_token'])
                        if fail:raise OSError('start failed')
                        row=device_row(0);row['request']=request
                        row['startup']['device'].update(requested=request,child=dict(pid=2,ppid=request['parent']['pid'],nspid='NSpid:\t2'),
                            inherited_visibility=request['physical_token'],effective_visibility=request['physical_token'],physical=request['parent']['physical'])
                        responses.append({'value':row['startup']})
                    proc.start.side_effect=start
                    return proc
                context.Process.side_effect=process
                with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,env,clear=True), \
                     patch('torch.cuda.get_device_properties',return_value=props), \
                     patch('torch.cuda.current_device',side_effect=AssertionError('drifted current ordinal must not be used')), \
                     patch.object(v.mp,'get_context',return_value=context),patch.object(VllmDoraRollout,'_receive',side_effect=lambda:responses.pop()):
                    if fail:
                        with self.assertRaisesRegex(OSError,'start failed'):
                            VllmDoraRollout(base_model='/base',checkpoint='/anchor',identity='snapshot',log_path=Path(tmp)/'log',device=device,trainer_rank=0)
                        parent.close.assert_called_once();child.close.assert_called_once()
                    else:
                        with VllmDoraRollout(base_model='/base',checkpoint='/anchor',identity='snapshot',log_path=Path(tmp)/'log',device=device,trainer_rank=0):pass
                    self.assertEqual(dict(os.environ),env)

    def test_child_wrong_or_missing_identity_fails_before_engine_allocation(self):
        import tempfile
        from src.qwen import vllm_rollout as v
        row=device_row(0);request=row['request'];props=SimpleNamespace(uuid=str(UUID(int=1)),pci_domain_id=0,pci_bus_id=103,pci_device_id=0)
        for visibility,count,current,properties in [('0',1,1,props),('0',8,0,props),('wrong',1,0,props),
                 ('0',1,0,SimpleNamespace()),('0',1,0,SimpleNamespace(uuid=str(UUID(int=9)),pci_domain_id=0,pci_bus_id=103,pci_device_id=0))]:
            engine=Mock();connection=Mock()
            with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':visibility}), \
                 patch('torch.cuda.device_count',return_value=count),patch('torch.cuda.current_device',return_value=current), \
                 patch('torch.cuda.get_device_properties',return_value=properties),patch.object(v,'_process_identity',return_value=row['startup']['device']['child']), \
                 patch.object(v.os,'dup2'),patch.dict('sys.modules',{'vllm':SimpleNamespace(LLM=engine,ModelRegistry=Mock())}):
                v._worker(connection,request,'/base','/anchor','snapshot',{},str(Path(tmp)/'log'))
            engine.assert_not_called()
            self.assertFalse(connection.send.call_args.args[0]['ok'])

    def test_permuted_numeric_uuid_and_explicit_ordinal_ignore_current_drift(self):
        from src.qwen import vllm_rollout as v
        self.assertEqual(v._physical_token('7,5,3',1),'5')
        self.assertEqual(v._physical_token('GPU-'+str(UUID(int=2))+',GPU-'+str(UUID(int=1)),1),'GPU-'+str(UUID(int=1)))
        self.assertEqual(v._physical_token(None,7),'7')
        for visibility,device in [('',0),('7,5',2),('7, 5',1),('7',True)]:
            with self.assertRaises(ValueError):v._physical_token(visibility,device)
        with self.assertRaisesRegex(ValueError,'selected parent'):
            v._check_physical(device_row(0)['request']['parent']['physical'],'GPU-'+str(UUID(int=2)))

    def test_all_rank_missing_duplicate_and_reordered_evidence(self):
        import copy
        from src.qwen.vllm_rollout import validate_device_assignments
        rows=[device_row(i) for i in range(8)];validate_device_assignments(rows,list(range(8)))
        for change in (lambda x:x.pop(),lambda x:x.reverse(),
                       lambda x:x[1]['startup']['device'].update(logical_device=1),
                       lambda x:x[1]['startup']['device'].pop('physical'),
                       lambda x:x[1]['request']['parent'].update(physical=x[0]['request']['parent']['physical'])):
            wrong=copy.deepcopy(rows);change(wrong)
            with self.assertRaises(ValueError):validate_device_assignments(wrong,list(range(8)))

    def test_infrastructure_entry_explicit_device_and_saved_admission(self):
        import json,runpy,tempfile,torch
        from contextlib import ExitStack
        from probes import iterative_positive as p
        module=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/probes/coordexp_infras/vllm_dora_rollout.py'))
        entry=module['main'];source=p.load(online.INPUTS)
        q=SimpleNamespace(model=torch.nn.Linear(1,1),base_model_path='/base',
            tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda x:99,pad_token_id=0),
            processor=SimpleNamespace(apply_chat_template=lambda *a,**k:'chat'))
        for duplicate in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint=Path(tmp)/'anchor';checkpoint.mkdir();output=Path(tmp)/'run';engine=Mock()
                row=device_row(1);engine.device_request=row['request'];engine.startup=row['startup']
                engine.__enter__=Mock(return_value=engine);engine.__exit__=Mock(return_value=False)
                constructor=Mock(return_value=engine)
                def gather(rows,value):
                    rows[:]=[device_row(0),value]
                    if duplicate:
                        rows[0]['request']['parent']['physical']=dict(value['request']['parent']['physical'])
                        rows[0]['startup']['device']['physical']=dict(value['request']['parent']['physical'])
                sync=Mock(side_effect=RuntimeError('CPU stop after device admission'))
                contexts=[patch.dict(os.environ,{'WORLD_SIZE':'2','RANK':'1','LOCAL_RANK':'1'}),
                    patch('sys.argv',['smoke','--checkpoint',str(checkpoint),'--output',str(output),'--learning-step']),
                    patch.object(p,'load',side_effect=lambda path:source if path==online.INPUTS else {'prompt':{'system':'s','user':'u'}}),
                    patch.object(p,'compose',return_value=(q,None,{})),patch.object(online,'native_batch',return_value=None),
                    patch.dict(entry.__globals__,{'snapshot':lambda model:('snapshot',590)}),
                    patch('src.artifacts.git_identity.capture_source_identity',return_value={}),
                    patch('torch.cuda.set_device'),patch('torch.cuda.synchronize',sync),
                    patch('torch.nn.parallel.DistributedDataParallel',side_effect=lambda model,**kw:model),
                    patch('torch.distributed.init_process_group'),patch('torch.distributed.destroy_process_group'),
                    patch('torch.distributed.all_gather_object',side_effect=gather),
                    patch('src.qwen.vllm_rollout.VllmDoraRollout',constructor)]
                with ExitStack() as stack:
                    for context in contexts:stack.enter_context(context)
                    with self.assertRaisesRegex(ValueError if duplicate else RuntimeError,'duplicate physical|CPU stop'):entry()
                self.assertEqual(constructor.call_args.kwargs['device'],1)
                self.assertEqual(constructor.call_args.kwargs['trainer_rank'],1)
                engine.generate.assert_not_called()
                receipt=json.loads((output/'rank-1/receipt.json').read_text())
                if duplicate:sync.assert_not_called()
                else:self.assertEqual([r['request']['device'] for r in receipt['vllm_devices']],[0,1])


def device_row(rank):
    physical=dict(uuid='GPU-'+str(UUID(int=rank+1)),pci_domain_id=0,pci_bus_id=103+rank,pci_device_id=0)
    parent=dict(pid=10000+rank,ppid=9999,nspid=f'NSpid:\t{10000+rank}',visibility=None,physical=physical)
    request=dict(schema='coordexp-vllm-device-1',rank=rank,device=rank,physical_token=str(rank),parent=parent)
    child=dict(pid=20000+rank,ppid=parent['pid'],nspid=f'NSpid:\t{20000+rank}')
    device=dict(requested=request,child=child,inherited_visibility=str(rank),effective_visibility=str(rank),logical_device=0,physical=dict(physical))
    return dict(rank=rank,request=request,startup=dict(identity='snapshot',device=device))


def load_tests(loader, tests, pattern):
    # unittest discovery also executes the three original function tests.
    tests.addTests(unittest.FunctionTestCase(test) for test in (
        test_rollout_backend_requires_explicit_fresh_qualification,
        test_rollout_never_accepts_changed_prompt_or_false_stop,
        test_refresh_590_tensors_uses_bytes_not_per_tensor_file_descriptors))
    return tests


def test_rollout_backend_requires_explicit_fresh_qualification():
    with patch.object(online.p, "load", return_value={}):
        online.rollout_binding("unused", "hf")
        with pytest.raises(ValueError, match="qualification"):
            online.rollout_binding("unused", "vllm")
    with patch.object(online.p, "load", return_value={"rollout_backend": "vllm"}):
        online.rollout_binding("unused", "vllm")
        with pytest.raises(ValueError, match="qualification"):
            online.rollout_binding("unused", "hf")


def test_rollout_never_accepts_changed_prompt_or_false_stop():
    from PIL import Image
    request = NativeRequest("image", "prompt", Image.new("RGB", (32, 32)),
                            expected_token_ids=(10, 11))
    completion = SimpleNamespace(token_ids=[5, 99], finish_reason="stop",
                                 logprobs=[{5: SimpleNamespace(logprob=-0.2)},
                                           {99: SimpleNamespace(logprob=-0.1)}])
    output = SimpleNamespace(prompt_token_ids=[10, 11], outputs=[completion])
    engine = SimpleNamespace(generate=lambda *a, **kw: [output])
    result = _generate(engine, [request], [8], 99, 0, True)[0]
    assert result.token_ids == (5, 99) and result.raw_logprobs == (-0.2, -0.1)
    output.prompt_token_ids = [10, 12]
    with pytest.raises(RuntimeError, match="prompt tokens"):
        _generate(engine, [request], [8], 99, 0, False)
    output.prompt_token_ids = [10, 11]
    completion.finish_reason = "length"
    with pytest.raises(RuntimeError, match="stop reason"):
        _generate(engine, [request], [8], 99, 0, False)


def test_refresh_590_tensors_uses_bytes_not_per_tensor_file_descriptors():
    import torch
    from safetensors.torch import load
    adapter = {str(i): torch.tensor([float(i)]) for i in range(588)}
    embeddings = {k: torch.ones(2, 3) for k in ('input_embed_delta', 'output_embed_delta')}
    runtime = object.__new__(VllmDoraRollout)
    calls = []
    runtime._call = lambda operation, payload: calls.append((operation, payload))
    with patch('peft.get_peft_model_state_dict', return_value=adapter):
        runtime.refresh(None, SimpleNamespace(delta_tensors=lambda: embeddings), identity='new')
    operation, (a, e, identity) = calls[0]
    assert operation == 'refresh' and isinstance(a, bytes) and isinstance(e, bytes)
    assert len(load(a)) == 588 and load(a)['587'].item() == 587
    assert set(load(e)) == set(embeddings) and identity == runtime.identity == 'new'


def test_coordinate_norm_rpc_is_explicit_and_snapshot_bound():
    from src.qwen import vllm_rollout as v
    model=SimpleNamespace(configure_coordinate_output_norm=Mock(return_value={'mode':'median','identity':'fixed'}),
                          coordinate_output_norm_receipt=Mock(return_value={'calls':2}))
    assert v._configure_coordinate_output_norm(model,'median',list(range(1000)),'fixed')['mode']=='median'
    model.configure_coordinate_output_norm.assert_called_once_with('median',list(range(1000)),identity='fixed')
    assert v._coordinate_output_norm_receipt(model)=={'calls':2}
    engine=object.__new__(VllmDoraRollout);engine.identity='fixed';engine._call=Mock(return_value='ack')
    assert engine.configure_coordinate_output_norm('off',range(1000),identity='fixed')=='ack'
    engine._call.assert_called_once_with('coordinate_output_norm',('off',list(range(1000)),'fixed'))
    for mode,identity in [('bad','fixed'),('median','stale')]:
        with pytest.raises(ValueError):engine.configure_coordinate_output_norm(mode,range(1000),identity=identity)
    assert engine._call.call_count==1


def exact_cpu_engine(request, *, vocab_size=12):
    """Actual installed token placeholder replacement; no model allocation."""
    from vllm.multimodal.processing.processor import PromptReplacement, _apply_token_matches_with_placeholders
    import math
    calls = []
    def generate(prompts, params, **kwargs):
        calls.append((prompts,params))
        prompt = prompts[0]
        assert 'prompt' not in prompt and prompt['mm_processor_kwargs'] == {'do_resize':False}
        ids, matched, placeholders = _apply_token_matches_with_placeholders(
            prompt['prompt_token_ids'], {'image':[[PromptReplacement(
                modality='image',target=[7],replacement=[7,7,7]).resolve(0)]]})
        assert matched == {'image':[0]} and placeholders['image'][0].length == 3
        logprobs = None
        if params[0].logprobs == -1:
            logprobs = [{t:SimpleNamespace(logprob=-math.log(vocab_size-1) if t else -math.inf)
                         for t in range(vocab_size)}]
        completion = SimpleNamespace(token_ids=[5]*params[0].max_tokens,finish_reason='length',logprobs=logprobs)
        return [SimpleNamespace(prompt_token_ids=ids,outputs=[completion])]
    return SimpleNamespace(generate=generate, calls=calls,
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(get_vocab_size=lambda:vocab_size)))


def test_exact_tokens_expand_original_media_and_preserve_literal_prefix():
    from PIL import Image
    from src.qwen.vllm_rollout import _generate_exact
    request = NativeRequest('exact','must not tokenize this',Image.new('RGB',(32,32)),
                            expected_token_ids=(2,7,7,7,3))
    engine = exact_cpu_engine(request)
    kwargs = dict(chat_token_ids=[[2,7,3]],extensions=[[8,9]],budgets=[1],
                  eos_token_id=11,pad_token_id=0,full_scores=True,vocab_size=12)
    result = _generate_exact(engine,[request],**kwargs)[0]
    assert result['processed_prompt_token_ids'] == [2,7,7,7,3,8,9]
    assert result['full_scores'][0] == '-inf' and len(result['full_scores']) == 12
    assert engine.calls[0][0][0]['prompt_token_ids'] == [2,7,3,8,9]
    kwargs.update(full_scores=False,budgets=[10])
    result = _generate_exact(engine,[request],**kwargs)[0]
    assert result['full_scores'] is None and engine.calls[-1][1][0].logprobs is None
    with pytest.raises(ValueError,match='one-token'):
        _generate_exact(engine,[request],**dict(kwargs,full_scores=True))
    request = NativeRequest('bad','irrelevant',request.image,expected_token_ids=(2,7,7,3))
    with pytest.raises(RuntimeError,match='exact prompt tokens'):
        _generate_exact(engine,[request],**kwargs)


def test_exact_score_support_and_snapshot_fail_closed():
    from PIL import Image
    from src.qwen.vllm_rollout import _generate_exact
    request = NativeRequest('exact','irrelevant',Image.new('RGB',(32,32)),expected_token_ids=(2,7,7,7,3))
    engine = exact_cpu_engine(request)
    valid = engine.generate
    kwargs = dict(chat_token_ids=[[2,7,3]],extensions=[[8,9]],budgets=[1],
                  eos_token_id=11,pad_token_id=0,full_scores=True,vocab_size=12)
    for mutate, message in [(lambda raw:raw.pop(2),'incomplete'),
                           (lambda raw:setattr(raw[2],'logprob',float('nan')),'NaN'),
                           (lambda raw:setattr(raw[2],'logprob',float('inf')),'NaN'),
                           (lambda raw:setattr(raw[2],'logprob',0),'argmax'),
                           (lambda raw:[setattr(v,'logprob',v.logprob+1) for v in raw.values()],'normalization')]:
        def corrupt(*args,**kw):
            outputs = valid(*args,**kw); mutate(outputs[0].outputs[0].logprobs[0]); return outputs
        engine.generate = corrupt
        with pytest.raises(RuntimeError,match=message):
            _generate_exact(engine,[request],**kwargs)
    engine.generate = valid
    with pytest.raises(ValueError,match='vocabulary'):
        _generate_exact(engine,[request],**dict(kwargs,vocab_size=13))
    for ids in [[-1],[True],[12]]:
        with pytest.raises(ValueError,match='token ID'):
            _generate_exact(engine,[request],**dict(kwargs,extensions=[ids]))
    wrapper = object.__new__(VllmDoraRollout); wrapper.identity='bound'; wrapper._call=Mock()
    with pytest.raises(ValueError,match='snapshot'):
        wrapper.generate_exact([request],chat_token_ids=[[2,7,3]],extensions=[[]],budgets=[1],
            eos_token_id=11,pad_token_id=0,identity='wrong',vocab_size=12)
    wrapper._call.assert_not_called()
