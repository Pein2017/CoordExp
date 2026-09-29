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
        parent=o.ROOT/'greedy-geometry-01';parent.mkdir(exist_ok=True)
        original_load=o.p.load
        for updates,schedule in ((1,(0,1)),(16,(0,1,2,4,8,16)),(64,(0,1,2,4,8,16,32,64))):
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

    def test_preservation_eligibility_prefix_and_denied_hidden(self):
        a=[10,20,100,200];b=[300,400,500,600]
        image,record,_=self.fixture([a,a,b],[])
        producer=dict(self.producer,kind='live_online')
        record=o.seal(record,producer)
        entry=o.preservation_entry(image,record,self.t)
        self.assertEqual([x['reason'] for x in entry['dispositions']],['eligible','literal_repeat','eligible'])
        seqs,positions=o.preservation_sequences(image,entry,self.t)
        self.assertEqual(len(seqs),2)
        for seq,row in zip(seqs,entry['rows']):
            self.assertEqual(list(seq.input_ids),record['prompt_token_ids']+record['token_ids'])
            self.assertEqual([a.causal_logits_position for a in seq.atoms],[len(record['prompt_token_ids'])+j-1 for j in row['positions']])
            targets=[a.coordinate_target for a in seq.atoms if a.token_type=='coordinate']
            self.assertEqual([a.bbox for a in targets],[tuple(row['bbox'])]*4)
            self.assertEqual([a.slot_index for a in targets],list(range(4)))
            self.assertEqual(seq.atoms[0].field,'object_ref_start');self.assertEqual(seq.atoms[-1].field,'box_end')
        for mutation in ('repeat','eos','shift'):
            bad=copy.deepcopy(entry)
            if mutation=='repeat':bad['rows'].append(bad['rows'][0])
            elif mutation=='eos':bad['rows'][0]['positions'].append(len(record['token_ids'])-1)
            else:bad['rows'][0]['positions']=[j+1 for j in bad['rows'][0]['positions']]
            with self.assertRaises(AssertionError):o.preservation_sequences(image,bad,self.t)
        image['objects']=[dict(coco_ann_id=1,desc='person',bbox_2d=a),dict(coco_ann_id=2,desc='car',bbox_2d=a)]
        withheld=o.preservation_entry(image,record,self.t)
        self.assertEqual(withheld['dispositions'][0]['reason'],'cross_category_conflict')
        image['objects'].pop();self.assertEqual(o.preservation_entry(image,record,self.t)['dispositions'][0]['reason'],'same_category_withheld')
        with patch.object(o.p,'load',side_effect=AssertionError('hidden read')):
            self.assertEqual(o.preservation_entry(dict(image,hidden={'box':[999]*4}),record,self.t),o.preservation_entry(image,record,self.t))
        stale=o.seal(record,dict(producer,update=1))
        with self.assertRaises(AssertionError):o.preservation_entry(image,stale,self.t)
        with self.assertRaises(AssertionError):o.credit(image,record,self.t,dict(producer,update=1))

    def test_preservation_actual_rowmean_empty_and_control_zero(self):
        image,record,_=self.fixture([[10,20,100,200],[10,20,100,200],[300,400,500,600]],[])
        record=o.seal(record,dict(self.producer,kind='live_online'))
        entry=o.preservation_entry(image,record,self.t);seqs,positions=o.preservation_sequences(image,entry,self.t)
        logits=torch.zeros(1,len(positions),self.vocab.vocab_size,requires_grad=True)
        loss,terms=o.preservation_objective(logits,positions,seqs,self.vocab,.25)
        expected=sum(o.p.image_loss(logits,s,self.vocab,positions)[0] for s in seqs)/2
        self.assertTrue(torch.equal(loss,.25*expected))
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        for seq in seqs:
            for atom in (seq.atoms[0],seq.atoms[-1]):self.assertGreater(float(logits.grad[0,positions.index(atom.causal_logits_position)].abs().sum()),0)
        unselected={len(record['prompt_token_ids'])+j-1 for j in range(len(record['token_ids']))}-set(positions)
        self.assertIn(len(record['prompt_token_ids'])+len(record['token_ids'])-2,unselected)
        with self.assertRaises(AssertionError):o.preservation_objective(logits,tuple(j+1 for j in positions),seqs,self.vocab,.25)
        empty=torch.zeros(1,1,self.vocab.vocab_size,requires_grad=True)
        zero,_=o.preservation_objective(empty,(0,),[],self.vocab,.25);zero.backward()
        self.assertEqual(float(zero),0);self.assertEqual(float(empty.grad.abs().sum()),0)
        ids=list(range(18));plans={i:dict(redirect=None) for i in ids}
        class Model:
            def no_sync(self):return nullcontext()
        grads=[]
        for rank in range(8):
            self.assertEqual(o.jobs(ids,rank,plans),o.jobs(ids,rank,plans,0))
            jobs=o.jobs(ids,rank,plans,.25)
            self.assertEqual([j['branch'] for j in jobs],['trace','P0','R']*len(ids[rank::8]))
            w=torch.tensor(1.,requires_grad=True)
            o.r.accumulate_family_step(Model(),jobs,lambda j:(w*(j['image_id']+1)*(.25 if j['branch']=='P0' else 1),{}))
            grads.append(float(w.grad))
        self.assertAlmostEqual(sum(grads)/8,2.25*sum(range(1,19))/18,places=5)
        # Actual zero-weight consumer adds neither loss nor gradient.
        fresh=logits.detach().requires_grad_();base=fresh.square().sum()+fresh.sum()
        before=torch.autograd.grad(base,fresh,retain_graph=True)[0]
        z,_=o.preservation_objective(fresh,positions,seqs,self.vocab,0)
        after=torch.autograd.grad(base+z,fresh)[0]
        self.assertTrue(torch.equal(before,after));self.assertEqual(float(z),0)

    def test_preservation_binding_wrong_bank_and_arm_rejected(self):
        bank={'kind':'fixed_incoming_prediction_preservation','images':[{'image_id':1}]}
        qualifier={'preservation':{'path':'bank','sha256':'expected'}}
        with patch.object(o.p,'load',side_effect=lambda path:bank if path=='bank' else qualifier),patch.object(o.p,'digest',return_value='expected'):
            binding,entries=o.preservation_binding(o.ROOT,.25,'expected')
            self.assertEqual(binding['weight'],.25);self.assertEqual(set(entries),{1})
            with self.assertRaises(AssertionError):o.preservation_binding(o.ROOT,.25,'wrong')
            with self.assertRaises(AssertionError):o.preservation_binding(o.ROOT,.5,'expected')
            with patch.object(o.p,'digest',return_value='corrupt'):
                with self.assertRaises(AssertionError):o.preservation_binding(o.ROOT,.25,'expected')

    def test_preservation_forward_consumer_and_unequal_rows(self):
        from types import SimpleNamespace
        image,record,_=self.fixture([[10,20,100,200],[300,400,500,600]],[])
        text=record['text'].replace('person','traffic light',1)
        ids=self.t.encode(text,add_special_tokens=False)
        record=o.seal(dict(record,text=text,token_ids=ids,generated_tokens=len(ids)),dict(self.producer,kind='live_online'))
        entry=o.preservation_entry(image,record,self.t);seqs,positions=o.preservation_sequences(image,entry,self.t)
        self.assertNotEqual(len(seqs[0].atoms),len(seqs[1].atoms))
        logits=torch.zeros(1,len(positions),self.vocab.vocab_size)
        logits[:,len(seqs[0].atoms):,0]=8;logits.requires_grad_()
        expected=[o.p.image_loss(logits,s,self.vocab,positions)[0] for s in seqs]
        loss,_=o.preservation_objective(logits,positions,seqs,self.vocab,.25)
        self.assertTrue(torch.equal(loss,.25*sum(expected)/2))
        pooled=.25*sum(v*len(s.atoms) for v,s in zip(expected,seqs))/sum(len(s.atoms) for s in seqs)
        self.assertNotAlmostEqual(float(loss),float(pooled),places=5)
        # Exercise the actual forward dispatcher using CPU compact logits, no model call.
        tensor=torch.tensor
        q=SimpleNamespace(model=object(),tokenizer=self.t)
        def compact(**kw):
            self.assertEqual(kw['logits_to_keep'].tolist(),list(positions))
            return SimpleNamespace(logits=logits)
        with patch('src.qwen.native.exact_history_inputs',return_value={}) as history, \
             patch.object(torch,'autocast',side_effect=lambda *a,**k:nullcontext()), \
             patch.object(torch,'tensor',side_effect=lambda data,**kw:tensor(data)):
            actual,evidence=o.forward(q,compact,SimpleNamespace(inputs={}),image,record,{'producer':{'update':7}},None,self.vocab,'P0',entry,.25)
        self.assertTrue(torch.equal(actual,loss))
        self.assertEqual(history.call_args.args[2],[record['prompt_token_ids']+record['token_ids']])
        self.assertEqual(evidence['preservation']['source_producer']['update'],0)
        self.assertEqual(evidence['producer']['update'],7)
        self.assertEqual(evidence['input_sha256'],o.identity(list(seqs[0].input_ids)))
        self.assertEqual(evidence['row_losses'],[dict(order=x['order'],atoms=len(s.atoms)) for x,s in zip(entry['rows'],seqs)])

    def test_preservation_readback_rejects_wrong_expected_weight(self):
        root=o.ROOT/'preservation-01/cpu';output=root/'not-a-runtime'
        qualifier={'sha256':{},'preservation':{'path':'bank','sha256':'expected'}}
        bank={'kind':'fixed_incoming_prediction_preservation','images':[{'image_id':i} for i in range(18)]}
        def load(path):
            if path==root/'qualification.json':return qualifier
            if path=='bank':return bank
            if path==o.INPUTS:return [{'image_id':i} for i in range(18)]
            if path.name=='complete.json':return dict(status='complete',updates=1)
            if path.name=='preservation.json':return dict(weight=.25,bank_sha256='expected',bank_path='bank')
            self.fail(str(path))
        with patch.object(o.p,'load',side_effect=load),patch.object(o.p,'digest',return_value='expected'):
            with self.assertRaisesRegex(AssertionError,'wrong preservation arm'):o.readback(output,root,1,0,'expected')
            with self.assertRaisesRegex(AssertionError,'wrong preservation bank identity'):o.readback(output,root,1,.25,'swapped')

    def test_sparse_geometry_actual_consumer_high_mass_illegal_greedy_ties(self):
        image,record,plan=self.fixture([[742,30,742,86]],[])
        positions=o.trace_positions(plan,record);row=plan['observations'][0]
        pos=row['coordinate_positions'][2];index=positions.index(len(record['prompt_token_ids'])+pos-1)
        self.assertEqual(o.erroneous_slots(plan['observations']),((pos,743,1000),))
        z=torch.full((1,len(positions),self.vocab.vocab_size),-100.)
        legal=list(self.vocab.coordinate[743:]);bad=self.vocab.coordinate[742]
        z[0,index,legal]=0;z[0,index,bad]=1;z.requires_grad_()
        old,oldterms=o.trace_objective(z,positions,plan,record,image,self.t,self.vocab)
        same,_=o.trace_objective(z,positions,plan,record,image,self.t,self.vocab,0)
        self.assertTrue(torch.equal(old,same))
        go=torch.autograd.grad(old,z,retain_graph=True)[0]
        self.assertTrue(torch.equal(go,torch.autograd.grad(same,z,retain_graph=True)[0]))
        new,terms=o.trace_objective(z,positions,plan,record,image,self.t,self.vocab,.1)
        gn=torch.autograd.grad(terms['Gmax_weighted'],z,retain_graph=True)[0]
        self.assertAlmostEqual(float(terms['Gmax_unweighted']),float(torch.nn.functional.softplus(torch.tensor(2.))),places=6)
        self.assertGreater(float(go[0,index,bad]),0);self.assertGreater(float(gn[0,index,bad]),0)
        self.assertTrue(torch.all(gn[0,index,legal]<0));self.assertTrue(torch.all(gn[0,index,legal]==gn[0,index,legal[0]]))
        self.assertEqual(int(torch.count_nonzero(gn[:,[j for j in range(len(positions)) if j!=index]])),0)
        mass=257/(257+__import__('math').exp(1));self.assertAlmostEqual(mass,.989534,places=6)
        self.assertEqual(int(z[0,index].argmax()),bad)
        self.assertGreater(float(gn[0,index,bad]/go[0,index,bad]),30)
        escaped=z.detach().clone();escaped[0,index,0]=1000;escaped.requires_grad_()
        escape=o.greedy_geometry_objective(escaped,positions,plan['observations'],len(record['prompt_token_ids']),self.vocab.coordinate)
        escape.backward();self.assertGreater(float(escaped.grad[0,index,0]),0)
        with self.assertRaises((AssertionError,ValueError)):o.trace_objective(z,tuple(j+1 for j in positions),plan,record,image,self.t,self.vocab,.1)
        # Detached diagnostic uses the same margin; it cannot alter training gradients.
        before=torch.autograd.grad(new,z,retain_graph=True)[0]
        detail=o.geometry_diagnostic_rows(z,positions,plan['observations'],record,self.vocab.coordinate)
        after=torch.autograd.grad(new,z)[0];self.assertTrue(torch.equal(before,after))
        d=detail[2];self.assertTrue(d['eligible_error']);self.assertTrue(d['emitted_equals_replay_argmax'])
        self.assertAlmostEqual(d['legal_mass'],mass,places=6);self.assertEqual(d['legal_max_ties'],257)
        self.assertEqual(detail[0]['new_weighted_image']['l2'],0)
        # Ties in the full illegal complement share gradient too.
        tied=torch.tensor([1.,1.,0.,0.],requires_grad=True)
        o.max_geometry_margin(tied,[2,3]).backward()
        self.assertEqual(float(tied.grad[0]),float(tied.grad[1]));self.assertEqual(float(tied.grad[2]),float(tied.grad[3]))

    def test_sparse_geometry_999_repeated_contexts_empty_and_means(self):
        image,record,plan=self.fixture([[999,20,999,200],[10,20,10,200],[10,20,10,200]],[])
        errors=o.erroneous_slots(plan['observations']);self.assertEqual(len(errors),3)
        first=plan['observations'][0];self.assertIn(first['coordinate_positions'][0],[e[0] for e in errors])
        self.assertNotIn(first['coordinate_positions'][2],[e[0] for e in errors])
        positions=o.trace_positions(plan,record);z=torch.zeros(1,len(positions),self.vocab.vocab_size,requires_grad=True)
        value=o.greedy_geometry_objective(z,positions,plan['observations'],len(record['prompt_token_ids']),self.vocab.coordinate)
        self.assertAlmostEqual(float(value),float(torch.nn.functional.softplus(torch.tensor(1.))),places=6)
        value.backward();self.assertTrue(torch.isfinite(z.grad).all())
        _,legal_record,legal_plan=self.fixture([[10,20,100,200]],[])
        lp=o.trace_positions(legal_plan,legal_record);empty=torch.zeros(1,len(lp),self.vocab.vocab_size,requires_grad=True)
        v=o.greedy_geometry_objective(empty,lp,legal_plan['observations'],len(legal_record['prompt_token_ids']),self.vocab.coordinate)
        v.backward();self.assertEqual(float(v),0);self.assertEqual(float(empty.grad.abs().sum()),0)
        # Different error counts produce image means, not a pooled-position mean.
        w=torch.tensor(1.,requires_grad=True);ids=list(range(18));plans={i:{'redirect':None} for i in ids};grads=[]
        class Model:
            def no_sync(self):return nullcontext()
        for rank in range(8):
            w=torch.tensor(1.,requires_grad=True)
            def f(job):
                i=job['image_id'];n=1+i%3
                scalar=torch.nn.functional.softplus(w*(i+1)).repeat(n).mean()*.1 if job['branch']=='trace' else w*0
                return scalar,{}
            o.r.accumulate_family_step(Model(),o.jobs(ids,rank,plans),f);grads.append(float(w.grad))
        w=torch.tensor(1.,requires_grad=True);expected=sum(torch.nn.functional.softplus(w*(i+1)) for i in ids)*.1/18;expected.backward()
        self.assertAlmostEqual(sum(grads)/8,float(w.grad),places=6)

    def test_geometry_explicit_start_actual_run_binding_and_exports(self):
        from pathlib import Path
        root=o.ROOT/'greedy-geometry-01/cpu';checkpoint=Path('/frozen/control64')
        spec=dict(checkpoint=str(checkpoint),checkpoint_identity_sha256='bound',recipe='sparse actual illegal .1; no P0')
        qualifier={'sha256':{},'geometry':spec};recipe=o.identity(spec)
        original=o.p.load
        def load(path):
            if path==root/'qualification.json':return qualifier
            if path==checkpoint/'identity.json':return {'adapter':'bound'}
            return original(path)
        with patch.object(o.p,'load',side_effect=load),patch.object(o.p,'digest',return_value='bound'):
            self.assertEqual(o.geometry_binding(root,checkpoint,.1,recipe)['checkpoint'],str(checkpoint))
            for cp,weight,sha in [(None,.1,recipe),(Path('/wrong'),.1,recipe),(checkpoint,.25,recipe),(checkpoint,.1,'changed')]:
                with self.assertRaises(AssertionError):o.geometry_binding(root,cp,weight,sha)
            with patch.object(o.p,'digest',return_value='changed'):
                with self.assertRaises(AssertionError):o.geometry_binding(root,checkpoint,.1,recipe)
            class AtCompose(Exception):pass
            with patch.object(o,'start',return_value=(0,root,[],{})),patch.object(o.p,'write'), \
                 patch('torch.distributed.init_process_group'),patch.object(o.p,'compose',side_effect=AtCompose) as compose:
                with self.assertRaises(AtCompose):o.run(root,root,16,geometry_weight=.1,start_checkpoint=checkpoint,recipe_sha256=recipe)
                compose.assert_called_once_with(checkpoint,evaluation=False)
            with patch.object(o.p,'load',return_value={'sha256':{},'geometry':spec}):
                with self.assertRaises(AssertionError):o.readback(root,root,16,geometry_weight=.1,start_checkpoint=Path('/wrong'),recipe_sha256=recipe)
        self.assertEqual(o.export_steps(1),(0,1));self.assertEqual(o.export_steps(16),(0,1,2,4,8,16))
        self.assertEqual(o.export_steps(64),(0,1,2,4,8,16,32,64))

    def test_geometry_start_export_tensor_rejection(self):
        from pathlib import Path
        start=Path('/start');export=Path('/export');changed=False
        def payload(path):
            is_adapter='/adapter/' in path
            key='layer.lora_A.weight' if is_adapter else 'input_embed_delta'
            return {key:torch.tensor([2. if changed and path.startswith('/export') else 1.])}
        with patch('safetensors.torch.load_file',side_effect=payload):
            o.verify_start_export(start,export)
            changed=True
            with self.assertRaisesRegex(AssertionError,'zero differs'):o.verify_start_export(start,export)

    def witness_fixture(self):
        image,record,_=self.fixture([[742,30,742,86]],[])
        with patch.object(o,'frozen_records',return_value=[record]):
            bank=o.witness_bank(o.ROOT,[image['image_id']],self.t)
        return image,record,bank['images'][0]

    def test_witness_actual_prefix_loss_and_detached_diagnostic(self):
        from types import SimpleNamespace
        image,record,entry=self.witness_fixture()
        full,pos=o.witness_inputs(entry,self.t)
        self.assertEqual(full,record['prompt_token_ids']+record['token_ids'][:6])
        self.assertEqual(pos,[len(full)-1])
        for key,value in [('input_ids',full+[record['token_ids'][6]]),
                          ('causal_position',len(full)),('legal_range',[742,1000]),
                          ('source_version',1)]:
            bad=copy.deepcopy(entry);bad[key]=value
            with self.assertRaises(AssertionError):o.witness_inputs(bad,self.t)
        z=torch.zeros(1,1,self.vocab.vocab_size,requires_grad=True)
        tensor=torch.tensor;q=SimpleNamespace(model=object(),tokenizer=self.t)
        def model(**kwargs):
            self.assertEqual(kwargs['logits_to_keep'].tolist(),pos)
            return SimpleNamespace(logits=z*1)
        with patch('src.qwen.native.exact_history_inputs',return_value={}) as history, \
             patch.object(torch,'autocast',side_effect=lambda *a,**k:nullcontext()), \
             patch.object(torch,'tensor',side_effect=lambda data,**kw:tensor(data)):
            loss,e=o.witness_forward(q,model,SimpleNamespace(inputs={}),entry,self.vocab,{'update':9},.1)
            before=torch.autograd.grad(loss,z,retain_graph=True)[0]
            diagnostic,d=o.witness_forward(q,model,SimpleNamespace(inputs={}),entry,self.vocab,{'update':9},0,True)
            after=torch.autograd.grad(loss,z)[0]
        self.assertTrue(torch.equal(before,after));self.assertFalse(diagnostic.requires_grad)
        self.assertEqual(history.call_args.args[2],[full]);self.assertEqual(e['producer']['update'],9)
        self.assertEqual(e['source_producer']['update'],0);self.assertEqual(d['positions'],pos)
        self.assertNotIn('row_losses',e)
        self.assertGreater(float(before[0,0,self.vocab.coordinate[742]]),0)
        self.assertTrue(torch.all(before[0,0,list(self.vocab.coordinate[743:])]<0))
        self.assertAlmostEqual(e['loss'],.1*e['margin'],places=6)

    def test_witness_four_image_mean_schedule_zero_and_gradient_state(self):
        from types import SimpleNamespace
        ids=list(range(18));plans={i:{'redirect':None} for i in ids};selected={1,4,9,16}
        self.assertEqual(o.jobs(ids,0,plans),o.jobs(ids,0,plans,0,()))
        grads=[];seen=[]
        class Model:
            def no_sync(self):return nullcontext()
        for rank in range(8):
            w=torch.tensor(2.,requires_grad=True);schedule=o.jobs(ids,rank,plans,0,selected)
            self.assertEqual(sum(j['sync'] for j in schedule),1)
            self.assertEqual(schedule[-1]['branch'],'R')
            for n,j in enumerate(schedule):
                if j['branch']=='witness':
                    self.assertEqual(schedule[n+1]['branch'],'R');self.assertFalse(j['sync']);seen.append(j['image_id'])
            o.r.accumulate_family_step(Model(),schedule,lambda j:(.1*w if j['branch']=='witness' else w*0,{}))
            grads.append(float(w.grad))
        self.assertEqual(sorted(seen),sorted(selected))
        self.assertAlmostEqual(sum(grads)/8,.1*4/18,places=7)
        self.assertNotAlmostEqual(sum(grads)/8,.1,places=6)
        model=torch.nn.Linear(1,1);model.weight.grad=torch.ones_like(model.weight)*3
        model.bias.grad=torch.ones_like(model.bias)*4;q=SimpleNamespace(model=model)
        producer={'parameter_sha256':o.identity(o.parameter_identity(model))}
        before=model.weight.grad.clone()
        with patch.object(o,'witness_forward',return_value=(None,{'diagnostic':True})):
            o.witness_diagnostics(q,{1:object()},{1:object()},None,producer)
        self.assertTrue(torch.equal(before,model.weight.grad))
        with self.assertRaisesRegex(AssertionError,'stale'):
            o.witness_diagnostics(q,{}, {},None,{'parameter_sha256':'wrong'})
        def leak(*args):model.weight.grad.add_(1);return None,{}
        with patch.object(o,'witness_forward',side_effect=leak):
            with self.assertRaises(AssertionError):o.witness_diagnostics(q,{1:object()},{1:object()},None,producer)

    def test_witness_readback_wrong_arm_rejects_before_artifacts(self):
        root=o.ROOT/'test-witness';output=root/'run'
        def load(path):
            if path==root/'qualification.json':return {'sha256':{}}
            if path==o.INPUTS:return [{'image_id':i} for i in range(18)]
            if path.name=='complete.json':return dict(status='complete',updates=1)
            if path.name=='geometry.json':return {'weight':.1}
            if path.name=='witness.json':return {'weight':0}
            self.fail(str(path))
        with patch.object(o.p,'load',side_effect=load),patch.object(o,'geometry_binding',return_value={'weight':.1}), \
             patch.object(o,'witness_binding',return_value=({'weight':.1},{})):
            with self.assertRaisesRegex(AssertionError,'wrong witness arm'):
                o.readback(output,root,1,geometry_weight=.1,witness_weight=.1)

    def test_witness_stability_counts_versions_not_repeat_votes(self):
        burdens=dict(geometry_invalid=100,literal_complete_repeats=99,literal_valid_repeats=0,
            malformed=0,near_repeat_occurrence_pairs=0,caps=1,valid_rows=2)
        row=dict(mode='category',cohort='human13',image_id=1,burdens=burdens,
            sets={'retained':{'gained':[7]},'hidden':{'gained':[]}})
        outcomes={('zero' if v==0 else str(v)):{'images':[copy.deepcopy(row)]} for v in range(17)}
        result=o.witness_stability(outcomes,16)['late/category/combined']
        self.assertEqual(result['bad_image_versions'],8);self.assertEqual(result['bad_versions'],8)
        self.assertEqual(result['burdens']['geometry_invalid'],dict(sum=800,mean=100,maximum=100))
        self.assertEqual(result['acquired_id_versions']['retained:1:7'],list(range(9,17)))

    def test_witness_real_four_hidden_denial_and_binding_rejection(self):
        source=o.ROOT/'greedy-geometry-01/treatment-01'
        bank=o.witness_bank(source,[i['image_id'] for i in self.images],self.t)
        self.assertEqual([(e['image_id'],e['source_version']) for e in bank['images']],[(13348,4),(16228,11),(351017,0),(417044,8)])
        actual_load=o.p.load
        def allowed(path):
            path=__import__('pathlib').Path(path)
            if source in path.parents and any(part.startswith('rollout-') for part in path.parts):return actual_load(path)
            raise AssertionError('hidden or nonprediction read')
        with patch.object(o.p,'load',side_effect=allowed):
            self.assertEqual(bank,o.witness_bank(source,[i['image_id'] for i in self.images],self.t))
        for entry in bank['images']:
            changed=copy.deepcopy(entry);changed['record']['evaluator_hidden']={'bbox':[0,0,999,999],'desc':'changed'}
            self.assertEqual(o.witness_inputs(entry,self.t),o.witness_inputs(changed,self.t))
        root=o.ROOT/'test-witness';spec=dict(path=str(root/'bank.json'),sha256='bound',
            image_ids=[e['image_id'] for e in bank['images']],source=str(source))
        def load(path):
            if path==root/'qualification.json':return {'witness':spec}
            if str(path)==spec['path']:return bank
            self.fail(str(path))
        with patch.object(o.p,'load',side_effect=load),patch.object(o.p,'digest',return_value='bound'):
            for weight in (0,.1):self.assertEqual(o.witness_binding(root,weight,'bound')[0]['weight'],weight)
            with self.assertRaisesRegex(AssertionError,'bank identity'):o.witness_binding(root,.1,'wrong')
            with patch.object(o.p,'digest',return_value='drift'):
                with self.assertRaises(AssertionError):o.witness_binding(root,.1,'bound')
            bank['source']='wrong'
            with self.assertRaises(AssertionError):o.witness_binding(root,.1,'bound')

    def multi_fixture(self):
        a=[300,10,400,100];c=[800,10,900,100]
        boxes=[a,[450,400,480,500],a,[600,10,600,100],c]
        known=[a,c,[100,10,130,100],[200,10,230,100],[500,10,530,100],[950,10,980,100]]
        image,record,_=self.fixture(boxes,[dict(coco_ann_id=i,desc='person',bbox_2d=x) for i,x in enumerate(known)])
        return image,record,o.bridge_credit(image,record,self.t,self.producer,'chain')

    def test_bridge_all_sites_unknown_suffix_and_atom_rebasing(self):
        image,record,plan=self.multi_fixture();repair=plan['bridge']
        self.assertEqual((plan['m'],plan['k'],plan['n']),(2,4,6))
        self.assertEqual([x['annotation_id'] for x in repair['B']],[2,3,4,5])
        self.assertEqual(repair['B'][0]['cut'],repair['B'][1]['cut'])
        self.assertEqual([x['boundary'] for x in repair['B']],['successor']*3+['terminal'])
        self.assertEqual([repair['token_ids'][j] for j in repair['original_to_repaired']],record['token_ids'])
        seqs=o.bridge_sequences(image,record,plan,self.t)
        n=len(record['prompt_token_ids']);selected={a.target_position-n for seq in seqs for a in seq.atoms}
        for row in plan['observations']:
            mapped={repair['original_to_repaired'][j] for j in row['positions']}
            if row['order'] in {x['order'] for x in plan['M']}:self.assertTrue(mapped<=selected)
            else:self.assertFalse(mapped & selected)
        self.assertNotIn(len(repair['token_ids'])-1,selected)
        for row,seq in zip(repair['B']+repair['M'],seqs):
            self.assertEqual([a.token_id for a in seq.atoms],[repair['token_ids'][j] for j in row['positions']])
            self.assertEqual([a.causal_logits_position for a in seq.atoms],[n+j-1 for j in row['positions']])
            coords=[a.coordinate_target for a in seq.atoms if a.coordinate_target is not None]
            self.assertEqual([list(x.bbox) for x in coords],[row['bbox']]*4)
            self.assertEqual([x.slot_index for x in coords],list(range(4)))
            self.assertEqual(seq.atoms[-1].field,'box_end')
        for mutation in ('token_ids','M','B'):
            bad=copy.deepcopy(plan)
            if mutation=='token_ids':bad['bridge']['token_ids'][0]=0
            else:bad['bridge'][mutation][0]['positions'][0]+=1
            with self.assertRaises(AssertionError):o.bridge_sequences(image,record,bad,self.t)
        bad=copy.deepcopy(plan);bad['producer']=dict(plan['producer'],update=99)
        with self.assertRaises(AssertionError):o.bridge_sequences(image,record,bad,self.t)

    def test_bridge_terminal_cap_invalid_and_empty_global(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        b=dict(coco_ann_id=1,desc='person',bbox_2d=[900,0,950,100])
        image,record,_=self.fixture([[20,10,20,100]],[b])
        plan=o.bridge_credit(image,record,self.t,self.producer,'chain')
        self.assertEqual((plan['m'],plan['k']),(0,1));self.assertTrue(plan['terminal_certified'])
        # Complete invalid predecessor does not spoil a certified EOS boundary.
        empty,er,_=self.fixture([],[b]);ep=o.bridge_credit(empty,er,self.t,self.producer,'chain')
        self.assertEqual((ep['m'],ep['k']),(0,1))
        eos=self.t.convert_tokens_to_ids('<|im_end|>')
        for ids,stop in [([eos],'max_new_tokens'),([eos,eos],'im_end'),([eos]+record['token_ids'],'im_end')]:
            bad=o.seal(dict(er,token_ids=ids,text=self.t.decode(ids,skip_special_tokens=False),generated_tokens=len(ids),stop_reason=stop),self.producer)
            rejected=o.bridge_credit(empty,bad,self.t,self.producer,'chain')
            self.assertFalse(rejected['terminal_certified']);self.assertEqual(rejected['k'],0)
        with TemporaryDirectory() as d:
            o.require_bridge_supervision({0:ep},Path(d),0)
            cap=o.seal(dict(er,text='',token_ids=[],generated_tokens=0,stop_reason='max_new_tokens'),self.producer)
            no=o.bridge_credit(empty,cap,self.t,self.producer,'chain');self.assertEqual(no['n'],0)
            optimizer=unittest.mock.Mock()
            with self.assertRaisesRegex(RuntimeError,'no-supervision'):
                o.require_bridge_supervision({0:no},Path(d),0);optimizer.step()
            optimizer.step.assert_not_called();self.assertTrue((Path(d)/'no-supervision-0.json').exists())
        image,record,plan=self.multi_fixture()
        # Capped/malformed suffix cannot supply terminal B, earlier matched cuts survive.
        for suffix,stop in [('', 'max_new_tokens'),('<|object_ref_start|>person<|im_end|>','im_end')]:
            ids=record['token_ids'][:-1]+self.t.encode(suffix,add_special_tokens=False)
            changed=o.seal(dict(record,token_ids=ids,text=self.t.decode(ids,skip_special_tokens=False),generated_tokens=len(ids),stop_reason=stop),self.producer)
            result=o.bridge_credit(image,changed,self.t,self.producer,'chain')
            self.assertFalse(result['terminal_certified']);self.assertEqual(result['k'],3)
            self.assertEqual(result['bridge_dispositions'][-1]['reason'],'unsupported_boundary')
        image,record,_=self.fixture([[10,10,100,100]])
        plan=o.bridge_credit(image,record,self.t,self.producer,'chain');self.assertEqual((plan['m'],plan['k'],plan['n']),(1,0,1))
        self.assertEqual(o.bridge_trace_plan(plan,'chain')['M'],plan['M'])

    def test_bridge_coherent_rowmean_and_same_subset_identity(self):
        image,record,plan=self.multi_fixture()
        image=copy.deepcopy(image);image['objects'][2]['desc']='traffic light'
        plan=o.bridge_credit(image,record,self.t,self.producer,'chain');seqs=o.bridge_sequences(image,record,plan,self.t)
        pos=tuple(sorted({a.causal_logits_position for seq in seqs for a in seq.atoms}))
        z=torch.zeros(1,len(pos),self.vocab.vocab_size,requires_grad=True)
        with torch.no_grad():
            for j,seq in enumerate(seqs):
                for a in seq.atoms:z[0,pos.index(a.causal_logits_position),a.token_id]=j/3
        total,terms=o.bridge_objective(z,pos,seqs,plan,'chain',self.vocab)
        values=[o.p.image_loss(z,seq,self.vocab,pos)[0] for seq in seqs]
        torch.testing.assert_close(total,sum(values)/plan['n'])
        pooled=sum(v*len(seq.atoms) for v,seq in zip(values,seqs))/sum(len(seq.atoms) for seq in seqs)
        self.assertFalse(torch.isclose(total,pooled));self.assertFalse(torch.isclose(total,total*plan['n']/plan['m']))
        moved=o.bridge_trace_plan(plan,'chain');self.assertEqual(moved['M'],[])
        grad,=torch.autograd.grad(total,z,retain_graph=True)
        self.assertTrue(torch.isfinite(grad).all())
        for seq in seqs:self.assertGreater(float(grad[0,pos.index(seq.atoms[-1].causal_logits_position)].abs().sum()),0)
        with self.assertRaises(AssertionError):o.bridge_objective(z,tuple(j+1 for j in pos),seqs,plan,'chain',self.vocab)
    def test_bridge_local_independent_branches_and_credit_ledger(self):
        image,record,_=self.multi_fixture();plan=o.bridge_credit(image,record,self.t,self.producer,'local')
        self.assertEqual((plan['m'],plan['k']),(2,4))
        self.assertEqual(o.bridge_trace_plan(plan,'local')['M'],[])
        ledger={};n=len(record['prompt_token_ids']);losses=[]
        for j,b in enumerate(plan['eligible']):
            seqs=o.bridge_sequences(image,record,plan,self.t,j)
            tokens,rows,weights,nb=o.bridge_rows(record,plan,j)
            self.assertEqual(tokens[:b['cut']],record['token_ids'][:b['cut']])
            self.assertEqual(tokens[b['cut']:b['cut']+len(b['token_ids'])],b['token_ids'])
            self.assertEqual(len(seqs),1 if b['boundary']=='terminal' else 2)
            self.assertEqual(len(tokens),b['cut']+sum(len(x['positions']) for x in rows))
            for row,seq,w in zip(rows,seqs,weights):
                ledger[row['annotation_id']]=ledger.get(row['annotation_id'],0)+w
                self.assertEqual([a.causal_logits_position for a in seq.atoms],[n+x-1 for x in row['positions']])
                self.assertEqual([list(a.coordinate_target.bbox) for a in seq.atoms if a.coordinate_target],[row['bbox']]*4)
                self.assertEqual(seq.atoms[-1].field,'box_end')
            pos=tuple(sorted({a.causal_logits_position for seq in seqs for a in seq.atoms}))
            z=torch.zeros(1,len(pos),self.vocab.vocab_size,requires_grad=True)
            value,terms=o.bridge_objective(z,pos,seqs,plan,'local',self.vocab,j)
            raw=[o.p.image_loss(z,seq,self.vocab,pos)[0] for seq in seqs]
            torch.testing.assert_close(value,sum(v*w for v,w in zip(raw,weights)))
            if j<2:
                self.assertEqual(weights,[1/6,1/12])
                self.assertFalse(torch.isclose(value,sum(raw)/6)) # missing 1/q must fail
            value.backward();self.assertTrue(torch.isfinite(z.grad).all())
            self.assertTrue(all(a.token_id!=self.t.convert_tokens_to_ids('<|im_end|>') for seq in seqs for a in seq.atoms))
            losses.append(value.detach())
        self.assertEqual(set(ledger),set(range(6)))
        for value in ledger.values():self.assertAlmostEqual(value,1/6)
        # Adding original C would double its vote; all trace semantic rows are removed here.
        self.assertAlmostEqual(sum(ledger.values()),1)
        remaining=copy.deepcopy(image);remaining['objects']=[x for x in image['objects'] if x['coco_ann_id'] not in (2,3)]
        rp=o.bridge_credit(remaining,record,self.t,self.producer,'local')
        self.assertEqual([x['annotation_id'] for x in o.bridge_trace_plan(rp,'local')['M']],[0])
        self.assertEqual(o.bridge_trace_plan(dict(rp,policy='chain'),'chain')['M'],[])
        for field in ('cut','successor_order','annotation_id'):
            bad=copy.deepcopy(plan);bad['eligible'][0][field]=999
            with self.assertRaises(AssertionError):o.bridge_sequences(image,record,bad,self.t,0)
        with self.assertRaises(AssertionError):o.bridge_sequences(image,record,plan,self.t,plan['k'])

    def test_bridge_real18_hidden_denial_all_admissible(self):
        source=o.ROOT/'greedy-geometry-01/treatment-01/rollout-16'
        records=o.frozen_records(source,[x['image_id'] for x in self.images]);by={x['image_id']:x for x in records}
        load=o.p.load;policy=load(o.p.POLICY)
        def allowed(path):
            if path==o.p.POLICY:return policy
            raise AssertionError('non-whitelisted/hidden read')
        counts=[]
        with patch.object(o.p,'load',side_effect=allowed):
            for image in self.images:
                record=by[image['image_id']];plan=o.bridge_credit(image,record,self.t,record['producer'],'chain')
                admitted={x['annotation_id'] for x in plan['bridge']['B']} if plan['bridge'] else set()
                self.assertEqual(admitted,{x['annotation_id'] for x in plan['bridge_dispositions'] if x['reason']=='inserted'})
                self.assertEqual(plan['n'],plan['m']+len(admitted))
                self.assertFalse(any(x['reason']=='eligible' for x in plan['bridge_dispositions']))
                changed=dict(record,evaluator_hidden={'box':[999]*4,'description':'changed'})
                self.assertEqual(o.bridge_credit(image,changed,self.t,record['producer'],'chain'),plan)
                o.bridge_sequences(image,record,plan,self.t)
                local=o.bridge_credit(image,record,self.t,record['producer'],'local')
                self.assertEqual(local['eligible'],plan['eligible']);self.assertEqual(local['bridge_dispositions'],plan['bridge_dispositions'])
                for j in range(local['k']):o.bridge_sequences(image,record,local,self.t,j)
                counts.append((plan['m'],plan['k']))
        self.assertEqual(sum(x[0] for x in counts),240)
        self.assertEqual(sum(x[1] for x in counts),269)

    def test_bridge_actual_dispatch_readback_and_equal_image_sync(self):
        from types import SimpleNamespace
        image,record,_=self.multi_fixture();tensor=torch.tensor
        q=SimpleNamespace(model=object(),tokenizer=self.t);histories={}
        def compact(**kw):return SimpleNamespace(logits=torch.zeros(1,len(kw['logits_to_keep']),self.vocab.vocab_size,requires_grad=True))
        for arm in ('local','chain'):
            plan=o.bridge_credit(image,record,self.t,self.producer,arm)
            for branch,index in [('trace',None)]+[('bridge',j) for j in (range(plan['k']) if arm=='local' else [None])]:
                with patch('src.qwen.native.exact_history_inputs',return_value={}) as history, \
                     patch.object(torch,'autocast',side_effect=lambda *a,**k:nullcontext()), \
                     patch.object(torch,'tensor',side_effect=lambda data,**kw:tensor(data)):
                    loss,evidence=o.forward(q,compact,SimpleNamespace(inputs={}),image,record,plan,None,self.vocab,branch,geometry_weight=.1,insertion_policy=arm,branch_index=index)
                histories[arm,branch,index]=history.call_args.args[2];o.verify_bridge_forward(evidence,image,record,plan,self.t,arm)
                for key in ('positions','input_sha256','bridge','row_losses'):
                    bad=copy.deepcopy(evidence)
                    if key=='positions':bad[key][0]+=1
                    elif key=='bridge':bad[key]['n']-=1
                    elif key=='row_losses':bad[key]=[{}]
                    else:bad[key]='wrong'
                    with self.assertRaises(AssertionError):o.verify_bridge_forward(bad,image,record,plan,self.t,arm)
                with self.assertRaises(AssertionError):o.verify_bridge_forward(evidence,image,record,plan,self.t,'chain' if arm=='local' else 'local')
                loss.backward()
        self.assertEqual(histories['local','trace',None],histories['chain','trace',None])
        ids=list(range(18));plans={i:dict(k=i%4) for i in ids}
        class Model:
            def __init__(self):self.calls=0
            def no_sync(self):self.calls+=1;return nullcontext()
        for arm in ('local','chain'):
            gradients=[]
            for rank in range(8):
                schedule=o.jobs(ids,rank,plans,insertion_policy=arm);model=Model();x=torch.tensor(1.,requires_grad=True)
                optimizer=unittest.mock.Mock()
                def consume(j):
                    optimizer.step.assert_not_called();optimizer.zero_grad.assert_not_called()
                    return x*(j['image_id']+1 if j['branch']=='trace' else 0),dict(image_id=j['image_id'],branch=j['branch'],bridge={'branch_index':j['branch_index']})
                evidence=o.r.accumulate_family_step(model,schedule,consume)
                optimizer.step.assert_not_called();optimizer.zero_grad.assert_not_called()
                gradients.append(x.grad);o.verify_bridge_schedule(evidence,ids,rank,plans,arm)
                self.assertEqual(model.calls,len(schedule)-1)
                for field,value in [('sync',False),('branch','R'),('image_weight',1/3)]:
                    bad=copy.deepcopy(evidence);bad[-1][field]=value
                    with self.assertRaises(AssertionError):o.verify_bridge_schedule(bad,ids,rank,plans,arm)
                bad=copy.deepcopy(evidence);bad.pop()
                with self.assertRaises(AssertionError):o.verify_bridge_schedule(bad,ids,rank,plans,arm)
            torch.testing.assert_close(sum(gradients)/8,torch.tensor(9.5))

    def test_bridge_external_anchor_actual_start_and_missing_arm(self):
        from pathlib import Path
        root=Path('/cpu');checkpoint=Path('/external')
        spec=dict(checkpoint=str(checkpoint),manifest_sha256='manifest')
        qualifier=dict(sha256={},bridge=spec)
        payload=dict(schema='coordexp-infras-inference-checkpoint-payload-manifest',schema_version=1,
            adapter=dict(status='present',relative_root='adapter',files=[]),
            special_token_embedding_delta=dict(status='present',relative_root='special_token_embeddings',files=[]))
        original=o.p.load
        def load(path):
            if path==root/'qualification.json':return qualifier
            if path==checkpoint/'inference_payload_manifest.json':return payload
            if path==o.ENCODINGS:raise AssertionError('forbidden old retained encodings')
            return original(path)
        class AtCompose(Exception):pass
        with patch.object(o.p,'load',side_effect=load),patch.object(o.p,'digest',return_value='manifest'):
            for arm in ('local','chain'):
                with patch.object(o,'start',return_value=(0,root,[],{})),patch.object(o.p,'write'), \
                     patch('torch.distributed.init_process_group'),patch.object(o.p,'compose',side_effect=AtCompose) as compose:
                    with self.assertRaises(AtCompose):o.run(root,root,1,geometry_weight=.1,start_checkpoint=checkpoint,recipe_sha256=o.identity(spec),insertion_policy=arm)
                    compose.assert_called_once_with(checkpoint,evaluation=False)
            for arm,cp,sha in [('wrong',checkpoint,o.identity(spec)),('local',Path('/wrong'),o.identity(spec)),('local',checkpoint,'wrong')]:
                with self.assertRaises(AssertionError):o.bridge_binding(root,cp,.1,sha,arm)
            with self.assertRaisesRegex(AssertionError,'bridge arm required'):o.geometry_binding(root,None,0,None)
            with patch.object(o.p,'digest',return_value='changed'):
                with self.assertRaises(AssertionError):o.bridge_binding(root,checkpoint,.1,o.identity(spec),'local')

    def test_microbatch_native_causal_sum_gradients_and_diagnostics(self):
        from types import SimpleNamespace
        image,record,_=self.multi_fixture();image['objects'][2]['desc']='traffic light'
        plan=o.bridge_credit(image,record,self.t,self.producer,'local')
        class Toy(torch.nn.Module):
            def __init__(self,vocab):
                super().__init__();self.w=torch.nn.Parameter(torch.linspace(-.03,.03,vocab));self.seen=[]
            def get_rope_index(self,ids,grid,video,*,attention_mask):
                assert grid.tolist()==[[1,2,2]]*ids.shape[0]
                return (attention_mask.cumsum(-1)-1).clamp_min(0).unsqueeze(0).expand(3,-1,-1),None
            def forward(self,input_ids,attention_mask,position_ids,pixel_values,image_grid_thw,logits_to_keep,**kw):
                assert pixel_values.tolist()==[[2.,3.]]*input_ids.shape[0]
                assert torch.equal(position_ids[0],(attention_mask.cumsum(-1)-1).clamp_min(0))
                self.seen.append((input_ids.detach().clone(),attention_mask.detach().clone()))
                h=((input_ids%19).float()*attention_mask).cumsum(-1)/1000+position_ids[0]/10000
                h=h.index_select(1,logits_to_keep) if isinstance(logits_to_keep,torch.Tensor) else h[:,-logits_to_keep:]
                return SimpleNamespace(logits=h[:,:,None]*self.w)
        batch=SimpleNamespace(inputs=dict(input_ids=torch.tensor([record['prompt_token_ids']]),attention_mask=torch.ones(1,len(record['prompt_token_ids'])),image_grid_thw=torch.tensor([[1,2,2]]),pixel_values=torch.tensor([[2.,3.]])))
        tensor=torch.tensor;results=[];model=Toy(self.vocab.vocab_size);q=SimpleNamespace(model=model,tokenizer=self.t)
        with patch.object(torch,'autocast',side_effect=lambda *a,**k:nullcontext()),patch.object(torch,'tensor',side_effect=lambda data,**kw:tensor(data,**{k:v for k,v in kw.items() if k!='device'})):
            for mode in ('singleton','group','no_diagnostics'):
                model.zero_grad(set_to_none=True)
                if mode=='singleton':
                    ls=[o.forward(q,model,batch,image,record,plan,None,self.vocab,'bridge',geometry_weight=.1,insertion_policy='local',branch_index=j)[0] for j in range(4)]
                    value=sum(ls)
                else:
                    value,evidence=o.group_forward(q,model,batch,image,record,plan,self.vocab,[0,1,2,3],diagnostic=mode=='group')
                    for row in evidence['logical']:o.verify_bridge_forward(row,image,record,plan,self.t,'local')
                    physical=[dict(evidence,sync=True,image_weight=8/18)]
                    logical=[dict(image_id=image['image_id'],branch='bridge',branch_index=j,weight=8/18,sync=j==3) for j in range(4)]
                    with patch.object(o,'jobs',return_value=logical):
                        o.verify_physical_forwards(physical,[image['image_id']],0,{image['image_id']:plan},{image['image_id']:record},{image['image_id']:image},self.t,'local',4)
                        for field in ('members','shape','sync'):
                            bad=copy.deepcopy(physical)
                            if field=='members':bad[0][field]=[0,1,3,2]
                            elif field=='shape':bad[0][field]['padded_tokens']+=1
                            else:bad[0][field]=False
                            with self.assertRaises(AssertionError):o.verify_physical_forwards(bad,[image['image_id']],0,{image['image_id']:plan},{image['image_id']:record},{image['image_id']:image},self.t,'local',4)
                    self.assertEqual(evidence['shape']['batch'],4)
                    self.assertGreater(max(evidence['shape']['left_padding']),0)
                    self.assertEqual(evidence['shape']['unpadded_tokens'],sum(x['tokens'] for x in evidence['logical']))
                    self.assertAlmostEqual(float(value.detach()),sum(x['loss'] for x in evidence['logical']),places=5)
                value.backward();results.append((value.detach(),model.w.grad.detach().clone()))
            for value,grad in results[1:]:
                torch.testing.assert_close(value,results[0][0]);torch.testing.assert_close(grad,results[0][1],atol=1e-7,rtol=1e-5)
            self.assertFalse(torch.allclose(results[1][1]/4,results[0][1]))
            with self.assertRaises(AssertionError):o.group_forward(q,model,batch,image,record,plan,self.vocab,[0,2])
            # A changed row stays in that independent batch row; no cross-row conditioning.
            from src.qwen.native import select_compact_replay_logits
            raw=torch.randn(2,5,7,requires_grad=True);selected=select_compact_replay_logits(raw,[2,4])
            selected[0].sum().backward();self.assertEqual(float(raw.grad[1].abs().sum()),0)
            self.assertEqual(float(raw.grad[0,-1].abs().sum()),0)
            self.assertEqual(float(raw.grad[0,:2].abs().sum()),0)
        self.assertTrue(torch.isfinite(results[0][1]).all());self.assertGreater(float(results[0][1].abs().sum()),0)

    def test_microbatch_schedule_settings_and_no_intermediate_update(self):
        ids=list(range(18));plans={i:dict(k=i%6) for i in ids}
        class Model:
            def __init__(self):self.n=0
            def no_sync(self):self.n+=1;return nullcontext()
        for rank in range(8):
            logical=o.jobs(ids,rank,plans,insertion_policy='local')
            self.assertIs(o.physical_jobs(logical,1),logical)
            grouped=o.physical_jobs(logical,4);actual=[]
            for job in grouped:
                if job['branch']=='bridge_group':actual.extend((job['image_id'],'bridge',j) for j in job['members'])
                else:actual.append((job['image_id'],job['branch'],job['branch_index']))
            self.assertEqual(actual,[(j['image_id'],j['branch'],j['branch_index']) for j in logical])
            self.assertEqual(sum(j['sync'] for j in grouped),1);self.assertTrue(grouped[-1]['sync'])
            model=Model();x=torch.tensor(1.,requires_grad=True);optimizer=unittest.mock.Mock()
            def consume(job):
                optimizer.step.assert_not_called();optimizer.zero_grad.assert_not_called()
                return x*(len(job.get('members',[])) or 1),dict(job)
            o.r.accumulate_family_step(model,grouped,consume)
            torch.testing.assert_close(x.grad,torch.tensor(len(logical)*8/18))
            self.assertEqual(model.n,len(grouped)-1);optimizer.step.assert_not_called()
        from pathlib import Path
        profiles=[dict(arm='local',microbatch=4,activation_checkpointing=False)]
        with patch.object(o.p,'load',return_value={'execution':{'profiles':profiles}}):
            self.assertEqual(o.execution_binding(Path('/cpu'),'local',4,False),profiles[0])
            for a,m,c in [('chain',4,False),('local',1,False),('local',4,True)]:
                with self.assertRaises(AssertionError):o.execution_binding(Path('/cpu'),a,m,c)
        with patch.object(o.p,'load',return_value={}):
            self.assertIsNone(o.execution_binding(Path('/cpu'),'local',1,True))
            with self.assertRaises(AssertionError):o.execution_binding(Path('/cpu'),'local',4,True)
        model=unittest.mock.Mock(is_gradient_checkpointing=False);o.set_checkpointing(model,False)
        model.gradient_checkpointing_disable.assert_called_once();model.enable_input_require_grads.assert_called_once()
        model.enable_input_require_grads.reset_mock();o.set_checkpointing(model,False,enable_inputs=False)
        model.enable_input_require_grads.assert_not_called()

    def test_microbatch_actual_readback_rejects_execution_drift(self):
        from pathlib import Path
        root=Path('/cpu');output=Path('/run');expected=dict(arm='local',microbatch=4,activation_checkpointing=False)
        def load(path):
            if path==root/'qualification.json':return {'sha256':{}}
            if path==o.INPUTS:return [{'image_id':x['image_id']} for x in self.images]
            if path==o.RETAINED:return self.images
            if path==output/'rank-0/complete.json':return dict(status='complete',updates=1)
            if path==output/'rank-0/execution.json':return dict(expected,microbatch=1)
            raise AssertionError('unexpected read')
        with patch.object(o.p,'load',side_effect=load),patch.object(o,'execution_binding',return_value=expected),patch.object(o,'preservation_binding',return_value=(None,None)),patch.object(o,'witness_binding',return_value=(None,None)),patch.object(o,'bridge_binding',return_value={}),patch.object(o.r,'frontend',return_value=self.q):
            with self.assertRaises(AssertionError):o.readback(output,root,1,geometry_weight=.1,insertion_policy='local',microbatch=4,activation_checkpointing=False)

    def test_microbatch_readback_group_membership_mutation(self):
        image,record,plan=self.multi_fixture();plan=o.bridge_credit(image,record,self.t,self.producer,'local')
        # Actual verifier is also exercised with the real grouped forward in the native test;
        # here wrong membership/sync must fail before an input can be silently reassigned.
        ids=[image['image_id']]+[100000+i for i in range(17)];plans={i:dict(k=0) for i in ids};plans[image['image_id']]=plan
        expected=o.physical_jobs(o.jobs(ids,0,plans,insertion_policy='local'),4)
        wrong=[dict(image_id=j['image_id'],branch=j['branch'],sync=False,image_weight=j['weight']) for j in expected]
        with self.assertRaises((AssertionError,KeyError)):o.verify_physical_forwards(wrong,ids,0,plans,{image['image_id']:record},{image['image_id']:image},self.t,'local',4)
        a={'a':torch.tensor([1.,2.]),'b':torch.tensor([0.])};b={'a':torch.tensor([1.,1.]),'b':torch.tensor([0.])}
        measured=o.gradient_comparison(a,b)
        self.assertAlmostEqual(measured['relative_l2'],2**-.5);self.assertEqual(measured['max_abs_difference'],1)
        self.assertEqual(measured['parameters']['b']['difference_l2'],0)
        import ast,inspect
        tree=ast.parse(inspect.getsource(o.replay_benchmark));calls=[ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)]
        self.assertFalse(any('optimizer' in x or 'generate_continuations' in x or 'clip_grad' in x or 'save_checkpoint' in x for x in calls))
        self.assertEqual(calls.count('r.accumulate_family_step'),1)
        with patch('sys.argv',['online_row_credit','replay-benchmark','--output','/run','--start-checkpoint','/wrong']),patch.object(o,'replay_benchmark') as call:
            with self.assertRaises(AssertionError):o.main()
            call.assert_not_called()



if __name__=='__main__':unittest.main()
