"""CPU falsifiers: causal binding, gradients, reductions and persistent identities."""
import copy
import contextlib
import os
import sys
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from probes.pre_row_detection_aux import objective as a, bank as b, experiment as e


class TinyCausal(torch.nn.Module):
    def __init__(self,vocabulary=7):
        super().__init__()
        self.embedding=torch.nn.Embedding(17,4)
        self.layers=torch.nn.ModuleList([torch.nn.Identity()])
        self.norm=torch.nn.LayerNorm(4)
        self.head=torch.nn.Linear(4,vocabulary)

    def get_output_embeddings(self):return self.head

    def forward(self,input_ids,logits_to_keep,**kwargs):
        hidden=self.norm(self.embedding(input_ids%17).cumsum(1))
        return SimpleNamespace(logits=self.head(hidden[:,logits_to_keep]))


def target(position=1,kind='CHAIN_B',weight=1,class_name='person',box=None):
    return dict(row_id=0,eligible=True,position=position,kind=kind,weight=weight,
        **{'class':class_name},box=box or [10,20,500,700],image_id=1,image_identity='image',prefix=[3,4])


class AuxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_causal_opener_future_perturbation_and_nonzero_shared_gradient(self):
        model=TinyCausal();head=a.make_head(4,2)
        states=[]
        for tokens in ([1,2,3,4],[1,2,15,16]):
            with a.TrainableRows(model.norm,[1]) as capture:
                model(torch.tensor([tokens]),torch.tensor([1]))
            states.append(capture.hidden)
        torch.testing.assert_close(states[0],states[1],rtol=0,atol=0)
        loss,parts=a.auxiliary(head,states[0],[target()],['person','truck'],1)
        loss.backward()
        self.assertGreater(float(model.embedding.weight.grad.norm()),0)
        self.assertGreater(float(head.weight.grad.norm()),0)
        self.assertEqual(float(model.embedding.weight.grad[3:].abs().sum()),0)
        with self.assertRaisesRegex(ValueError,'detached'):
            a.auxiliary(head,states[0].detach(),[target()],['person','truck'],1)
        self.assertEqual(set(parts['components']),{'CE','L1','GIoU'})

    def test_opener_absolute_position_rejects_description_or_coordinate_leakage(self):
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _:99)
        atom=lambda p,t:SimpleNamespace(target_position=p,token_type=t)
        sequence=SimpleNamespace(input_ids=[1,2,99,7,8],atoms=[atom(2,'schema'),atom(3,'desc_text'),atom(4,'coordinate')])
        self.assertEqual(b.opener_position(sequence,dict(positions=[0,1,2]),2,tokenizer),2)
        with self.assertRaises(ValueError):b.opener_position(sequence,dict(positions=[1,2]),2,tokenizer)
        sequence.atoms[1].target_position=2
        with self.assertRaisesRegex(ValueError,'leakage'):b.opener_position(sequence,dict(positions=[0,1,2]),2,tokenizer)

    def test_whole_box_conflict_excludes_entire_group_only(self):
        first=target();alias=copy.deepcopy(first);conflict=target(box=[10,20,600,700]);other=target();other['prefix']=[3,5]
        rows=[first,alias,conflict,other];snapshot=copy.deepcopy(rows)
        b.exclude_conflicts(rows)
        self.assertEqual([r['eligible'] for r in rows],[False,False,False,True])
        self.assertEqual([r['box'] for r in rows],[r['box'] for r in snapshot])
        self.assertEqual([r['weight'] for r in rows],[r['weight'] for r in snapshot])
        same=[target(),target()];b.exclude_conflicts(same)
        self.assertTrue(all(r['eligible'] for r in same))

    def test_branch_and_eighteen_image_normalization(self):
        head=a.make_head(4,2);h=torch.randn(2,4,requires_grad=True)
        rows=[target(weight=.25),target(kind='redirect',weight=.75)]
        combined,_=a.auxiliary(head,h,rows,['person','truck'],1)
        left,_=a.auxiliary(head,h[:1],rows[:1],['person','truck'],1)
        right,_=a.auxiliary(head,h[1:],rows[1:],['person','truck'],1)
        torch.testing.assert_close(combined,left+right)
        # Existing local 8/18 weights and DDP mean give the all18 mean, including empty images.
        self.assertAlmostEqual(float((left+right)*(8/18)/8),float(combined/18),places=6)

    def test_saturated_box_giou_loss_and_gradient_are_finite(self):
        values=torch.tensor([[1000.,1000.,-1000.,-1000.],[-1000.,-1000.,1000.,1000.]],requires_grad=True)
        pred=a.boxes(values);gt=torch.tensor([[.1,.2,.5,.7],[.1,.2,.5,.7]])
        loss=(pred-gt).abs().mean()+.5*(1-a.giou(pred,gt)).mean();loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue((pred[:,2:]>=pred[:,:2]).all())
        self.assertTrue(((pred[:,2:]-pred[:,:2])==0).any())

    def test_separate_clipping_head_does_not_rescale_backbone(self):
        backbone=torch.nn.Parameter(torch.ones(1));head=torch.nn.Parameter(torch.ones(1))
        backbone.grad=torch.tensor([2.]);head.grad=torch.tensor([100.])
        norms=a.clip_separately([backbone],[head])
        self.assertEqual(norms,dict(backbone=2.,head=100.))
        self.assertAlmostEqual(float(backbone.grad),1.,places=5)
        self.assertAlmostEqual(float(head.grad),1.,places=5)

    def test_isolated_initializer_and_separate_serialization(self):
        model=TinyCausal();before=copy.deepcopy(model.state_dict());rng=torch.random.get_rng_state()
        with patch('torch.cuda.manual_seed_all',side_effect=AssertionError('head changed CUDA RNG')):
            head=a.make_head(4,2)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        other=a.make_head(4,2);torch.testing.assert_close(head.weight,other.weight,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'training-head.pt';a.save_head(head,path,'bank',['person','truck'])
            a.load_head(other,path,'bank',['person','truck'])
            with self.assertRaisesRegex(ValueError,'identity'):a.load_head(other,path,'wrong',['person','truck'])
            with self.assertRaisesRegex(ValueError,'identity'):a.load_head(other,path,'bank',['truck','person'])
            with self.assertRaises(ValueError):a.save_head(head,path,'bank',['person','truck'])
            torch.save(model.state_dict(),Path(directory)/'deployed.pt')
            saved=torch.load(Path(directory)/'deployed.pt',weights_only=True)
            self.assertEqual(saved.keys(),before.keys())
            for k in before:torch.testing.assert_close(saved[k],before[k],rtol=0,atol=0)

    def test_persisted_bank_and_upstream_identity_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.json';b.write(source,dict(input=1))
            bank=Path(directory)/'bank.json';b.write(bank,dict(schema='pre-row-fixed-bank-v1',input_sha256={str(source):b.fit.sha(source)}))
            digest=b.fit.sha(bank);b.verify_bank(bank,digest)
            source.write_text('{"input":2}')
            with self.assertRaisesRegex(ValueError,'persisted input identity'):b.verify_bank(bank,digest)
            with self.assertRaisesRegex(ValueError,'input identity'):b.verify_bank(bank,'0'*64)

    def test_missing_release_rejects_before_model_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'release.json';b.write(path,dict(native_released=False))
            with patch.object(b.o.p,'compose',side_effect=AssertionError('forbidden model access')):
                with self.assertRaisesRegex(ValueError,'native not released'):
                    e.runtime_release(SimpleNamespace(release=path,release_sha256=b.fit.sha(path)))

    def test_current_row_only_later_good_row_cannot_rescue(self):
        from probes import rollout_row_credit as r
        image=dict(image_id=1,width=100,height=100,image_path='tiny',objects=[])
        row=target();row.update(owner=3,prefix_sha256='prefix',history_kind='actual')
        bank=dict(rows=[row],images=[image])
        good=r.render_row(image,dict(desc='person',bbox_2d=row['box'])).assistant_content_text
        wrong=r.render_row(image,dict(desc='truck',bbox_2d=row['box'])).assistant_content_text
        base=dict(row_id=0,prefix_sha256='prefix',history_kind='actual',generated_tokens=20,stop_reason='im_end')
        for text,correct in ((good.split('<|object_ref_start|>',1)[1],True),
                             ((wrong+good).split('<|object_ref_start|>',1)[1],False),
                             ('malformed'+good,False)):
            result=e.conditional_metrics(bank,[dict(base,text=text)])
            self.assertEqual(result['rows'][0]['first_row_correct'],correct)

    def test_frozen_selection_binding_and_target_drift(self):
        root=b.OUT;bank_path=root/'cpu-01/bank.json';selector=root/'conditional-selection-01.json'
        bank=b.verify_bank(bank_path,'41ffaf9956187ac33f57ec43a71a29300cbf190f7cf5d05a2439665b5bf5c54f')
        cases=b.selection_cases(selector,'3cedbb1dd9ca976c214a998476061d48890004d20b88a3523c78f6c443a3fca6',bank,b.fit.sha(bank_path))
        self.assertEqual([r['row_id'] for r in cases],[0,101,342,571,1,102,296,523])
        self.assertEqual([r['kind'] for r in cases[:4]],['CHAIN_B','CHAIN_B','redirect','redirect'])
        bank['rows'][0]['box']=[1,2,3,4]
        with self.assertRaisesRegex(ValueError,'target or prefix drift'):
            b.selection_cases(selector,b.fit.sha(selector),bank,b.fit.sha(bank_path))


class RealEntryTinyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=b.o.r.frontend().tokenizer
        from src.losses.vocab import build_token_vocabulary_groups
        frontend=b.o.r.frontend()
        cls.vocab=build_token_vocabulary_groups(frontend.token_identity,tokenizer=cls.tokenizer)

    def fixture(self):
        image=dict(image_id=1,image_path='tiny',width=100,height=100,objects=[
            dict(coco_ann_id=1,desc='person',bbox_2d=[10,20,100,200]),
            dict(coco_ann_id=2,desc='truck',bbox_2d=[300,400,500,600])])
        text=b.o.r.render_row(image,image['objects'][0]).assistant_content_text+'<|im_end|>'
        producer=dict(kind='historical_CPU_fixture',update=0,parameter_sha256='tiny',source='tiny',
            owner_region=dict(b.o.OWNER_REGION),training_sha256='tiny',completion_weighting=b.o.RESTORED_M_WEIGHTING,
            completion_arm='treatment',redirect_selection=b.o.IDENTITY_SELECTION)
        raw=b.o.seal(dict(image,request_id='tiny',arm='greedy',crop=[0,0,100,100],prompt_token_ids=[42],media_sha256='tiny',
            image_grid_thw=[1,2,2],text=text,token_ids=self.tokenizer.encode(text,add_special_tokens=False),stop_reason='im_end'),producer)
        raw['generated_tokens']=len(raw['token_ids'])
        plan=b.o.completion_credit(image,raw,self.tokenizer,producer,'treatment',True,b.o.RESTORED_M_WEIGHTING)
        return image,raw,plan

    def test_shared_forward_base_unchanged_aux_off_and_aux_reaches_backbone(self):
        torch.set_num_threads(2)
        image,raw,plan=self.fixture();job=dict(branch='bridge',image_id=1)
        model=TinyCausal(len(self.tokenizer));q=SimpleNamespace(model=model,tokenizer=self.tokenizer)
        rows=[]
        for k,(row,seq,weight,kind) in enumerate(b.positive_rows(image,raw,plan,self.tokenizer,job)):
            r=target(position=b.opener_position(seq,row,1,self.tokenizer),kind=kind,weight=weight,class_name=row['description'],box=b.o.owner_box(image,row));r['row_id']=k;rows.append(r)
        def exact(model,inputs,full,**kwargs):return dict(input_ids=torch.tensor(full))
        with patch('src.qwen.native.exact_history_inputs',side_effect=exact):
            base,_=b.o.forward(q,model,SimpleNamespace(inputs={}),image,raw,plan,None,self.vocab,'bridge',geometry_weight=.1,correction_arm='treatment',duplicate_weight=1,replay_device_type='cpu')
            base_grad=torch.autograd.grad(base,tuple(model.parameters()))
            off,_=e.replay_training_loss(q,model,SimpleNamespace(inputs={}),image,raw,plan,self.vocab,job,rows,['person','truck'],sum(r['weight'] for r in rows))
            off_grad=torch.autograd.grad(off,tuple(model.parameters()),retain_graph=True)
            torch.testing.assert_close(base,off,rtol=0,atol=0)
            for left,right in zip(base_grad,off_grad):torch.testing.assert_close(left,right,rtol=0,atol=0)
            head=a.make_head(4,2)
            enabled,evidence=e.replay_training_loss(q,model,SimpleNamespace(inputs={}),image,raw,plan,self.vocab,job,rows,['person','truck'],sum(r['weight'] for r in rows),head)
            self.assertEqual(evidence['base_loss'],float(base))
            self.assertGreater(float(torch.autograd.grad(enabled-off,model.embedding.weight)[0].norm()),0)
            self.assertEqual(evidence['auxiliary']['rows'],2)
            excluded=copy.deepcopy(rows)
            for row in excluded:row['eligible']=False
            omitted,_=e.replay_training_loss(q,model,SimpleNamespace(inputs={}),image,raw,plan,self.vocab,job,excluded,['person','truck'],0,head)
            omitted_grad=torch.autograd.grad(omitted,tuple(model.parameters()))
            torch.testing.assert_close(base,omitted,rtol=0,atol=0)
            for left,right in zip(base_grad,omitted_grad):torch.testing.assert_close(left,right,rtol=0,atol=0)


class ConsumerBoundaryTest(unittest.TestCase):
    def test_real_conditional_entry_expands_media_once_and_preserves_guard(self):
        from src.qwen.vllm_rollout import _generate_exact
        from vllm.multimodal.processing.processor import PromptReplacement, _apply_token_matches_with_placeholders
        bank_path=b.OUT/'cpu-01/bank.json';bank_sha='41ffaf9956187ac33f57ec43a71a29300cbf190f7cf5d05a2439665b5bf5c54f'
        selection=b.OUT/'conditional-selection-01.json';selection_sha='3cedbb1dd9ca976c214a998476061d48890004d20b88a3523c78f6c443a3fca6'
        bank=b.verify_bank(bank_path,bank_sha);cases=b.selection_cases(selection,selection_sha,bank,bank_sha)
        q=b.o.r.frontend();image_pad=q.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        eos=q.tokenizer.convert_tokens_to_ids('<|im_end|>');observed=[];corrupt=False

        class CpuRollout:
            def __init__(self,**kwargs):self.receipts=[]
            def close(self):pass
            def generate(self,requests,**kwargs):
                return [SimpleNamespace(token_ids=[eos],stop_reason='im_end') for request in requests]
            def generate_exact(self,requests,**kwargs):
                def generate(prompts,params,**options):
                    outputs=[]
                    for request,prompt,param in zip(requests,prompts,params,strict=True):
                        count=math.prod(request.expected_image_grid)//4
                        ids,matched,placeholders=_apply_token_matches_with_placeholders(prompt['prompt_token_ids'],
                            {'image':[[PromptReplacement(modality='image',target=[image_pad],replacement=[image_pad]*count).resolve(0)]]})
                        self_test.assertEqual(matched,{'image':[0]})
                        self_test.assertEqual(placeholders['image'][0].length,count)
                        if corrupt:ids[-1]=(ids[-1]+1)%len(q.tokenizer)
                        outputs.append(SimpleNamespace(prompt_token_ids=ids,outputs=[SimpleNamespace(token_ids=[1]*param.max_tokens,finish_reason='length')]))
                        observed.append(dict(request_id=request.request_id,chat=kwargs['chat_token_ids'][0],processed=ids,extension=kwargs['extensions'][0]))
                    return outputs
                engine=SimpleNamespace(generate=generate,llm_engine=SimpleNamespace(model_config=SimpleNamespace(get_vocab_size=lambda:len(q.tokenizer))))
                kwargs.pop('identity')
                return _generate_exact(engine,requests,full_scores=False,**kwargs)

        self_test=self
        with tempfile.TemporaryDirectory(dir=b.OUT) as directory,contextlib.ExitStack() as stack:
            args=SimpleNamespace(bank=bank_path,bank_sha256=bank_sha,selection=selection,selection_sha256=selection_sha,
                checkpoint=Path(bank['checkpoint']),output=Path(directory)/'valid')
            stack.enter_context(patch.object(e,'runtime_release',return_value=dict(selection_sha256=selection_sha)))
            stack.enter_context(patch.object(e,'verify_bank',return_value=bank))
            stack.enter_context(patch.object(e,'selection_cases',return_value=cases))
            stack.enter_context(patch('src.artifacts.git_identity.capture_source_identity',return_value={}))
            stack.enter_context(patch('src.artifacts.git_identity.verify_source_identity'))
            stack.enter_context(patch.object(b.o,'verify_anchor_payload'))
            stack.enter_context(patch.object(b.o.r,'frontend',return_value=q))
            stack.enter_context(patch('src.qwen.vllm_rollout.VllmDoraRollout',CpuRollout))
            for rank in range(8):
                with patch.dict(os.environ,WORLD_SIZE='8',RANK=str(rank),LOCAL_RANK=str(rank)):e.evaluate(args)
            records,conditional=e.evaluation_readback(args.output,bank,bank_sha,cases,selection_sha)
            self.assertEqual(len(records),18);self.assertEqual(len(conditional),8)
            for row,result in zip(cases,observed,strict=True):
                self.assertEqual(result['processed'],row['prefix'])
                self.assertEqual(result['chat'].count(image_pad),1)
                self.assertEqual(result['extension'],row['prefix'][len(next(r for r in bank['records'] if r['image_id']==row['image_id'])['prompt_token_ids']):])
            corrupt=True;args.output=Path(directory)/'corrupt'
            with patch.dict(os.environ,WORLD_SIZE='8',RANK='0',LOCAL_RANK='0'):
                with self.assertRaisesRegex(RuntimeError,'changed exact prompt tokens'):e.evaluate(args)
            self.assertFalse((args.output/'rank-0/complete.json').exists())

    def evaluation_fixture(self, path):
        bank=dict(images=[dict(image_id=i) for i in range(18)],regime='CPU fixture')
        cases=[dict(row_id=i) for i in range(8)]
        for rank in range(8):
            out=path/f'rank-{rank}';out.mkdir(parents=True)
            ids=list(range(18))[rank::8]
            for i in ids:b.write(out/f'{i}.json',dict(image_id=i))
            b.write(out/'conditional.json',[dict(row_id=rank)])
            b.write(out/'vllm-operations.json',[])
            b.write(out/'complete.json',dict(status='complete',bank_sha256='bank',selection_sha256='selection',
                artifacts={f'{i}.json':b.fit.sha(out/f'{i}.json') for i in ids},
                conditional_sha256=b.fit.sha(out/'conditional.json'),operations_sha256=b.fit.sha(out/'vllm-operations.json')))
        return bank,cases

    def finalize(self, path, bank, cases):
        argv=['probe','eval-readback','--bank','fixture','--bank-sha256','bank',
            '--selection','fixture','--selection-sha256','selection','--output',str(path)]
        with patch.object(sys,'argv',argv),patch.object(e,'verify_bank',return_value=bank),patch.object(e,'selection_cases',return_value=cases):
            e.main()

    def test_rank_interleaving_and_duplicate_through_run_and_evaluate(self):
        for command in ('run','evaluate'):
            with self.subTest(command=command),tempfile.TemporaryDirectory(dir=b.OUT) as directory:
                root=Path(directory);output=root/'stage';release=root/'release.json'
                argv=['probe',command,'--release-sha256','digest']
                b.write(release,dict(native_released=True,unit_id=e.UNIT.name,bank_sha256='bank',selection_sha256='selection',
                    source_commit='commit',runtime={},updates=16,exact_invocations=[[sys.executable,'-m','probes.pre_row_detection_aux',command,
                        '--release-sha256','LEAD_RELEASE_SHA256']]))
                args=SimpleNamespace(command=command,release=release,release_sha256=b.fit.sha(release),output=output,
                    bank=Path('fixture'),bank_sha256='bank',selection=Path('fixture'),selection_sha256='selection',checkpoint=root/'anchor',updates=16,arm='A')
                bank=dict(checkpoint=str(args.checkpoint),checkpoint_manifest_sha256='manifest')
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(sys,'argv',argv))
                    stack.enter_context(patch('subprocess.check_output',return_value='commit'))
                    stack.enter_context(patch.object(e,'verify_bank',return_value=bank))
                    stack.enter_context(patch.object(e,'selection_cases',return_value=[]))
                    stack.enter_context(patch('src.artifacts.git_identity.capture_source_identity',return_value={}))
                    stack.enter_context(patch.object(b.o,'verify_anchor_payload'))
                    stack.enter_context(patch.object(b.o.r,'frontend',side_effect=RuntimeError('CPU stop before model')))
                    stack.enter_context(patch.object(torch.cuda,'set_device',side_effect=RuntimeError('CPU stop before model')))
                    consumer=e.run_arm if command=='run' else e.evaluate
                    for rank in range(8):
                        with patch.dict(os.environ,WORLD_SIZE='8',RANK=str(rank),LOCAL_RANK=str(rank)):
                            with self.assertRaisesRegex(RuntimeError,'CPU stop before model'):consumer(args)
                            self.assertTrue((output/f'rank-{rank}').is_dir())
                            with self.assertRaises(FileExistsError):consumer(args)

    def test_eval_finalizer_then_offline_preserves_payload_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=b.OUT) as directory:
            root=Path(directory);paths=[root/name for name in ('zero','A','B')];hashes=[]
            for path in paths:
                bank,cases=self.evaluation_fixture(path);self.finalize(path,bank,cases)
                hashes.append(b.fit.sha(path/'frozen.json'))
            args=SimpleNamespace(bank=Path('fixture'),bank_sha256='bank',selection=Path('fixture'),selection_sha256='selection',
                zero=paths[0],arm_a=paths[1],arm_b=paths[2],output=root/'offline.json')
            with patch.object(e,'verify_bank',return_value=bank),patch.object(e,'selection_cases',return_value=cases),\
                    patch.object(e,'conditional_metrics',return_value={}),patch.object(b.fit,'evaluate_versions',return_value={}):
                self.assertEqual(e.offline(args)['status'],'offline_complete')
                with self.assertRaises(FileExistsError):e.offline(args)
            self.assertEqual(hashes,[b.fit.sha(p/'frozen.json') for p in paths])
            self.assertNotIn('readback.json',b.read(paths[0]/'frozen.json'))
            with self.assertRaises(FileExistsError):self.finalize(paths[0],bank,cases)

    def test_evaluation_payload_and_derived_receipt_drift_reject(self):
        for mutation in ('missing','changed','extra','nested-readback','receipt','manifest'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(dir=b.OUT) as directory:
                path=Path(directory)/'evaluation';bank,cases=self.evaluation_fixture(path);self.finalize(path,bank,cases)
                if mutation=='missing':(path/'rank-0/0.json').unlink()
                if mutation=='changed':(path/'rank-0/conditional.json').write_text('[]')
                if mutation=='extra':b.write(path/'extra.json',{})
                if mutation=='nested-readback':b.write(path/'rank-0/readback.json',{})
                if mutation=='receipt':(path/'readback.json').write_text('{}')
                if mutation=='manifest':(path/'frozen.json').unlink()
                with self.assertRaises((AssertionError,ValueError,FileNotFoundError)):
                    e.evaluation_readback(path,bank,'bank',cases,'selection')


if __name__=='__main__':unittest.main()
