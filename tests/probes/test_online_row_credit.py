import copy
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from probes import online_row_credit as o
from src.losses.vocab import build_token_vocabulary_groups


class OnlineCreditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.q=o.r.frontend();cls.t=cls.q.tokenizer
        cls.vocab=build_token_vocabulary_groups(cls.q.token_identity,tokenizer=cls.t)
        cls.images=o.p.load(o.RETAINED)
        cls.producer=dict(kind='historical_CPU_fixture',update=0,parameter_sha256='fixture-not-live',source='accepted-zero')
        cls.records=[o.seal(x,cls.producer) for x in o.frozen_records(o.r.ZERO,[i['image_id'] for i in cls.images])]
        cls.plans=[o.credit(i,r,cls.t,cls.producer) for i,r in zip(cls.images,cls.records)]

    def fixture(self, boxes, objects=None):
        text=''.join('<|object_ref_start|>person<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{v}|>' for v in b)+'<|box_end|>' for b in boxes)+'<|im_end|>'
        ids=self.t.encode(text,add_special_tokens=False)
        record=o.seal(dict(self.records[0],text=text,token_ids=ids,generated_tokens=len(ids),stop_reason='im_end'),self.producer)
        image=copy.deepcopy(self.images[0])
        image['objects']=objects if objects is not None else [dict(coco_ann_id=j,desc='person',bbox_2d=b) for j,b in enumerate(dict.fromkeys(tuple(b) for b in boxes)) if b[0]<b[2] and b[1]<b[3]]
        return image,record,o.credit(image,record,self.t,self.producer)

    def test_real18_denied_hidden_and_stale_mixed_producer(self):
        self.assertEqual(sum(len(x['objects']) for x in self.images),513)
        # Renderer reads the frozen prompt policy; every other file access is denied.
        policy=o.p.POLICY; load=o.p.load; frozen_policy=load(policy)
        def allowed(path):
            if path==policy:return frozen_policy
            raise AssertionError('hidden or unwhitelisted read')
        with patch.object(o.p,'load',side_effect=allowed):
            again=[o.credit(i,r,self.t,self.producer) for i,r in zip(self.images,self.records)]
        self.assertEqual(again,self.plans)
        # Extra evaluator-owned state is ignored, never unpacked by the actual caller.
        changed=[dict(x,evaluator_hidden=dict(box=[0,0,999,999],desc='mutation')) for x in self.records]
        self.assertEqual([o.credit(i,r,self.t,self.producer) for i,r in zip(self.images,changed)],self.plans)
        o.verify_producer(self.records,self.producer,[x['image_id'] for x in self.images])
        bad=copy.deepcopy(self.records);bad[1]['producer']['update']=1
        with self.assertRaises(AssertionError):o.verify_producer(bad,self.producer,[x['image_id'] for x in self.images])
        bad=copy.deepcopy(self.records);bad[0]['token_ids'][0]=0
        with self.assertRaises(AssertionError):o.verify_producer(bad,self.producer,[x['image_id'] for x in self.images])

    def test_first_occurrence_redirect_and_literal_prefix(self):
        a=[10,20,100,200];b=[300,400,500,600]
        image,record,plan=self.fixture([a,a,b])
        self.assertEqual([x['order'] for x in plan['M']],[0,2])
        target=plan['redirect'];self.assertEqual(target['source'],'later_matched_first');self.assertEqual(target['duplicate_order'],1)
        sequence=o.redirect_sequence(image,record,target,self.t)
        cut=target['prefix_cut'];self.assertEqual(list(sequence.input_ids[:len(record['prompt_token_ids'])+cut]),record['prompt_token_ids']+record['token_ids'][:cut])
        self.assertEqual([a.token_id for a in sequence.atoms],target['token_ids'])
        self.assertEqual(sequence.atoms[-1].field,'box_end')
        self.assertFalse(any(a.token_type=='eos' for a in sequence.atoms))
        # Overlapping known support vetoes even a differently worded/extent proposal.
        _,_,veto=self.fixture([a,a],[dict(coco_ann_id=0,desc='person',bbox_2d=[11,21,101,201])])
        self.assertIsNone(veto['redirect']);self.assertGreater(veto['redirect_events'][0]['veto']['overlap'],0)
        _,_,repair=self.fixture([a,a],[dict(coco_ann_id=0,desc='person',bbox_2d=a),dict(coco_ann_id=1,desc='person',bbox_2d=b)])
        self.assertEqual(repair['redirect']['source'],'retained_repair')
        wrong=copy.deepcopy(target);wrong['positions']=[x+1 for x in wrong['positions']]
        with self.assertRaises((ValueError,IndexError)):o.redirect_sequence(image,record,wrong,self.t)
        with self.assertRaises(ValueError):o.semantic_site([self.t.convert_tokens_to_ids('<|box_start|>')],[self.t.convert_tokens_to_ids('<|box_end|>')],self.t)

    def test_actual_margin_high_confidence_causal_and_closure_gradients(self):
        image,record,plan=self.fixture([[10,20,100,200],[10,20,100,200],[300,400,500,600]])
        target=plan['redirect'];sequence=o.redirect_sequence(image,record,target,self.t)
        positions=tuple(a.causal_logits_position for a in sequence.atoms)
        d=target['site']['offset'];good=target['site']['good'];bad=target['site']['bad']
        logits=torch.zeros(1,len(positions),self.vocab.vocab_size)
        logits[0,d,bad]=1000;logits.requires_grad_()
        loss,terms=o.redirect_objective(logits,positions,sequence,target,len(record['prompt_token_ids']),self.vocab)
        margin_grad,=torch.autograd.grad(terms['redirect_margin'],logits,retain_graph=True)
        self.assertEqual(float(terms['redirect_margin']),1001.)
        self.assertEqual(float(margin_grad[0,d,bad]),1.);self.assertEqual(float(margin_grad[0,d,good]),-1.)
        self.assertEqual(int(torch.count_nonzero(margin_grad)),2)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[0,-1].abs().sum()),0)
        with self.assertRaises(AssertionError):o.redirect_objective(logits,tuple(x+1 for x in positions),sequence,target,len(record['prompt_token_ids']),self.vocab)

    def test_legal_all_vocabulary_own_start_999_and_rowmean(self):
        image,record,plan=self.fixture([[999,20,999,200],[10,20,10,200]])
        self.assertEqual([len(o.legal_slots(x)) for x in plan['observations']],[3,4])
        positions=o.trace_positions(plan,record)
        logits=torch.zeros(1,len(positions),self.vocab.vocab_size,requires_grad=True)
        value,terms=o.trace_objective(logits,positions,plan,record,image,self.t,self.vocab)
        self.assertEqual(float(terms['M']),0)
        expected=[]
        import math
        for row in plan['observations']:expected.append(sum(math.log(self.vocab.vocab_size/(hi-lo)) for _,lo,hi in o.legal_slots(row))/len(o.legal_slots(row)))
        self.assertAlmostEqual(float(value),sum(expected)/2,places=5)
        self.assertNotAlmostEqual(float(value),(expected[0]*3+expected[1]*4)/7,places=5)
        value.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        # A non-coordinate escape incurs more loss, with a positive gradient on escape logit.
        escape=torch.zeros_like(logits);escape[:,:,0]=1000;escape.requires_grad_()
        large,_=o.trace_objective(escape,positions,plan,record,image,self.t,self.vocab);large.backward()
        self.assertGreater(float(large),900);self.assertTrue(torch.all(escape.grad[:,:,0]>0))
        with self.assertRaises(AssertionError):o.trace_objective(logits,tuple(x+1 for x in positions),plan,record,image,self.t,self.vocab)

    def test_actual_M_empty_and_diagnostic_invariance(self):
        image,record,plan=self.fixture([[10,20,100,200]])
        positions=o.trace_positions(plan,record)
        logits=torch.zeros(1,len(positions),self.vocab.vocab_size,requires_grad=True)
        loss,terms=o.trace_objective(logits,positions,plan,record,image,self.t,self.vocab)
        self.assertGreater(float(terms['M']),0);self.assertGreater(float(terms['legal']),0)
        before=torch.autograd.grad(loss,logits,retain_graph=True)[0]
        o.diagnostics(terms,logits)
        after=torch.autograd.grad(loss,logits)[0];self.assertTrue(torch.equal(before,after))
        image,record,plan=self.fixture([]);positions=o.trace_positions(plan,record)
        logits=torch.zeros(1,1,self.vocab.vocab_size,requires_grad=True)
        loss,_=o.trace_objective(logits,positions,plan,record,image,self.t,self.vocab);loss.backward()
        self.assertEqual(float(loss),0);self.assertEqual(float(logits.grad.abs().sum()),0)

    def test_actual_quarter_R_and_EOS_mask(self):
        image,record,plan=self.fixture([[10,20,100,200]])
        seq=o.r.positive_sequence(image,record,plan['M'][0],self.t)
        positions=tuple(a.causal_logits_position for a in seq.atoms)+(len(seq.input_ids)-1,)
        logits=torch.zeros(1,len(positions),self.vocab.vocab_size,requires_grad=True)
        loss,terms,rows=o.retained_credit(logits,seq,('row',),self.vocab,positions)
        self.assertTrue(torch.equal(loss,.25*terms['R_unweighted']))
        self.assertGreater(float(loss.detach()),0)
        loss.backward()
        self.assertEqual(float(logits.grad[0,-1].abs().sum()),0)
        self.assertGreater(float(logits.grad[0,-2].abs().sum()),0)
        self.assertEqual(len(rows),1)

    def test_offline_freezes_all_outputs_before_truth_and_bad_freeze_rejects(self):
        output=o.ROOT/'test-offline-no-files';events=[]
        receipt=[dict(update=k,frozen_sha256='bound') for k in (0,1)]
        def load(path):
            if path==o.ROOT/'qualification.json':return {'sha256':{},'evaluator_sha256':{}}
            if path==output/'readback.json':return receipt
            if path==o.RETAINED:return self.images
            if path==o.r.ROOT/'cpu-03/evaluator-partitions.json':
                self.assertIn('offline-inputs-frozen.json',events)
                return dict(truth_sha256='bound',hidden10=[])
            if path==o.r.TRUTH:
                self.assertIn('offline-inputs-frozen.json',events);events.append('truth');return []
            raise AssertionError(path)
        with patch.object(o.p,'load',side_effect=load), patch.object(o,'frozen_records',return_value=[]), \
             patch.object(o.p,'write',side_effect=lambda path,value:events.append(path.name)), \
             patch.object(o.p,'digest',return_value='bound'), patch.object(o.r,'assess_outputs',return_value=[]), \
             patch.object(o.r,'family_outcomes',return_value={}):
            o.offline(output,o.ROOT,1)
            self.assertLess(events.index('offline-inputs-frozen.json'),events.index('truth'))
            events.clear()
            with patch.object(o.p,'digest',return_value='changed'):
                with self.assertRaises(AssertionError):o.offline(output,o.ROOT,1)
            self.assertNotIn('truth',events)

    def test_runtime_qualifier_has_no_evaluator_dependency(self):
        # Execute the actual start caller until the first CUDA operation. Model-free.
        root=o.ROOT/'test-qualification-no-files'
        qualifier={'sha256':{'input-only':'good'},'evaluator_sha256':{'hidden-truth':'mutated'}}
        class ReachedDevice(Exception):pass
        with patch.object(o.p,'load',return_value=qualifier), \
             patch.object(o.p,'digest',side_effect=lambda path: 'good' if path=='input-only' else self.fail('hidden hash accessed')), \
             patch.dict('os.environ',{'WORLD_SIZE':'8','RANK':'0','LOCAL_RANK':'0'}), \
             patch.object(torch.cuda,'set_device',side_effect=ReachedDevice):
            with self.assertRaises(ReachedDevice):o.start(root,root)
            # Deliberate wrong binding fails at the actual caller before device setup.
            qualifier['sha256']['hidden-truth']='mutated'
            with self.assertRaises(AssertionError):o.start(root,root)

    def test_readback_requires_exact_scheduled_exports_and_valid_hashes(self):
        import tempfile
        from pathlib import Path
        parent=o.ROOT/'cpu-02';parent.mkdir(exist_ok=True)
        original_load=o.p.load
        for updates,schedule in ((1,(0,1)),(64,(0,1,2,4,8,16,32,64))):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory(dir=parent) as tmp:
                output=Path(tmp);written=[]
                def producer(version):
                    params={'parameter':str(version)}
                    return dict(kind='live_online',update=version,parameter_sha256=o.identity(params)),params
                def load(path):
                    if path==output/'qualification.json':return {'sha256':{}}
                    if path==o.INPUTS:return [{'image_id':i} for i in range(18)]
                    if path.name=='complete.json':return dict(status='complete',updates=updates,artifacts={},source={'files':[]})
                    if path.name.startswith('update-'):
                        step=int(path.stem.split('-')[1])
                        return dict(optimizer_steps=[step],optimizer_state_count=590,lrs=[1e-5,5e-6],
                            synchronized_norms=['same']*8,forwards=[dict(sync=True,image_weight=8/18)])
                    if path.name.startswith('producer-'):
                        version=int(path.stem.split('-')[1]);p,params=producer(version)
                        return dict(producer=p,parameters=params)
                    return original_load(path)
                def frozen(path,ids,freeze):
                    version=int(path.name.split('-')[1]);p,_=producer(version)
                    return [o.seal(dict(image_id=i,request_id=str(i),token_ids=[],text='',prompt_token_ids=[1],
                        media_sha256='media',image_grid_thw=[1,2,2],stop_reason='im_end',generated_tokens=0),p) for i in range(18)]
                (output/'frozen.json').write_text('frozen')
                digest=o.p.digest
                with patch.object(o.p,'load',side_effect=load), patch.object(o,'frozen_records',side_effect=frozen), \
                     patch.object(o.p,'write',side_effect=lambda path,value:written.append(path.name)), \
                     patch.object(o.p,'digest',side_effect=lambda path: 'frozen' if Path(path).name=='frozen.json' else digest(path)):
                    with self.assertRaises((AssertionError,FileNotFoundError)):o.readback(output,output,updates)
                    self.assertNotIn('readback.json',written)
                    for step in schedule[:-1]:
                        checkpoint=output/f'checkpoint-{step}';checkpoint.mkdir()
                        (checkpoint/'payload').write_text(str(step))
                        (checkpoint/'identity.json').write_text(__import__('json').dumps({'payload':digest(checkpoint/'payload')}))
                    with self.assertRaises((AssertionError,FileNotFoundError)):o.readback(output,output,updates)
                    checkpoint=output/f'checkpoint-{schedule[-1]}';checkpoint.mkdir()
                    with self.assertRaises((AssertionError,FileNotFoundError)):o.readback(output,output,updates)
                    (checkpoint/'payload').write_text('final')
                    (checkpoint/'identity.json').write_text(__import__('json').dumps({'payload':digest(checkpoint/'payload')}))
                    o.readback(output,output,updates);self.assertEqual(written,['readback.json'])
                    written.clear();(checkpoint/'payload').write_text('corrupt')
                    with self.assertRaises(AssertionError):o.readback(output,output,updates)
                    self.assertFalse(written)

    def test_actual_unequal_rank_schedule_M_plus_quarter_R_no_skip(self):
        ids=list(range(18));plans={i:dict(redirect=None if i%2 else {}) for i in ids}
        # Use truthy redirects to exercise 2/3 forwards, unequal eligible counts.
        for i in ids:
            if i%2==0:plans[i]['redirect']={'exists':True}
        class Model:
            def no_sync(self):return nullcontext()
        rank_grads=[]
        for rank in range(8):
            weight=torch.tensor(1.,requires_grad=True)
            jobs=o.jobs(ids,rank,plans)
            self.assertEqual(sum(j['sync'] for j in jobs),1);self.assertTrue(jobs[-1]['sync'])
            self.assertEqual(sum(j['branch']=='R' for j in jobs),len(ids[rank::8]))
            def forward(job):
                i=job['image_id'];branch=job['branch']
                value=(i+1) if branch=='trace' else (.25*(2*i+1) if branch=='R' else 3)
                return weight*value,{}
            o.r.accumulate_family_step(Model(),jobs,forward);rank_grads.append(float(weight.grad))
        expected=sum(i+1+.25*(2*i+1)+(3 if i%2==0 else 0) for i in ids)/18
        self.assertAlmostEqual(sum(rank_grads)/8,expected,places=5)
        wrong=sum(rank_grads[r]/len(ids[r::8]) for r in range(8))/8
        self.assertNotAlmostEqual(wrong,expected)


if __name__=='__main__':unittest.main()
