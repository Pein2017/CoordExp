import copy
import math
import unittest
from dataclasses import replace

import torch

from probes import rollout_row_credit as r
from src.losses.vocab import build_token_vocabulary_groups


class RowCreditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.q=r.frontend();cls.t=cls.q.tokenizer
        cls.vocab=build_token_vocabulary_groups(cls.q.token_identity,tokenizer=cls.t)
        cls.truth=r.p.load(r.TRUTH);cls.visible,cls.hidden=r.split_reference(cls.truth,10)
        cls.records=r.p.read_evaluation(r.ZERO)
        cls.plans=r.credit_plan(cls.visible,cls.records,cls.t)

    def fixture(self,text,objects=None,stop='im_end'):
        record=dict(self.records[0],text=text,token_ids=self.t.encode(text,add_special_tokens=False),stop_reason=stop)
        image=copy.deepcopy(self.visible[0])
        if objects is not None: image['objects']=objects
        return image,record,r.credit_plan([image],[record],self.t)[0]

    def test_real18_split_hidden_mutation_and_membership(self):
        v20,h20=r.split_reference(self.truth,20)
        self.assertEqual(len(self.hidden),57);self.assertEqual(len(h20),114)
        self.assertEqual(sum(len(i['objects']) for i in self.visible),513)
        self.assertLessEqual(set(map(tuple,self.hidden)),set(map(tuple,h20)))
        self.assertTrue(any(i>=0 for _,i in self.hidden));self.assertTrue(any(o['coco_ann_id']<0 for i in self.visible for o in i['objects']))
        changed=copy.deepcopy(self.truth)
        for image in changed:
            for obj in image['objects']:
                if [image['image_id'],obj['coco_ann_id']] in self.hidden:
                    obj.update(desc='changed',bbox_2d=[0,0,999,999])
        retained,_=r.split_reference(changed,10)
        self.assertEqual(retained,self.visible)
        self.assertEqual(r.credit_plan(retained,self.records,self.t),self.plans)
        # Deliberate wrong boundary: passing full truth changes the actual plans.
        self.assertNotEqual(r.credit_plan(self.truth,self.records,self.t),self.plans)
        evaluated=r.evaluate_partition(self.truth,self.hidden,self.records)
        self.assertEqual(sum(x['hidden_denominator'] for x in evaluated),57)
        self.assertEqual(sum(x['retained_denominator'] for x in evaluated),513)
        self.assertNotEqual(sum(o['coco_ann_id']<0 for i in self.truth for o in i['objects']),57)

    def test_actual_positive_loss_shift_mask_and_fn(self):
        image=self.visible[0];record=self.records[0];plan=self.plans[0]
        # One real row, with all earlier real output retained, plus unequal trailing logits.
        row=plan['M'][-1]; seq=r.positive_sequence(image,record,row,self.t)
        positions=tuple(a.causal_logits_position for a in seq.atoms)+(len(seq.input_ids)-1,)
        logits=torch.randn(1,len(positions),self.vocab.vocab_size,requires_grad=True)
        small=dict(plan,M=[row],F=None,D=[],G=[])
        value,terms=r.image_objective('A',small,record,image,self.t,self.vocab,logits,positions)
        value.backward()
        self.assertTrue(all(logits.grad[0,i].abs().sum()>0 for i in range(len(positions)-1)))
        self.assertEqual(float(logits.grad[0,-1].abs().sum()),0)
        self.assertEqual(seq.atoms[0].field,'object_ref_start');self.assertEqual(seq.atoms[-1].field,'box_end')
        self.assertTrue(all(a.token_type!='eos' for a in seq.atoms))
        for a in seq.atoms: self.assertEqual(seq.input_ids[a.target_position],a.token_id)
        wrong=dict(row,positions=[i+1 for i in row['positions']])
        with self.assertRaises(ValueError): r.positive_sequence(image,record,wrong,self.t)
        fn=r.positive_sequence(image,record,plan['F'],self.t,True)
        self.assertEqual(fn.input_ids[:fn.atoms[0].target_position],tuple(record['prompt_token_ids']+record['token_ids'][:-1]))
        self.assertIn(plan['F']['annotation_id'],{o['coco_ann_id'] for o in image['objects']})
        self.assertNotIn([image['image_id'],plan['F']['annotation_id']],self.hidden)
        capped=dict(record,token_ids=record['token_ids'][:-1],text=self.t.decode(record['token_ids'][:-1],skip_special_tokens=False),stop_reason='max_new_tokens')
        cap=r.credit_plan([image],[capped],self.t)[0]
        self.assertIsNone(cap['F']);self.assertTrue(cap['known_fn_ids'])
        corrupted=dict(record,token_ids=record['token_ids'][:-1]+[1])
        with self.assertRaises(ValueError): r.credit_plan([image],[corrupted],self.t)

    def test_complete_row_unlikelihood_analytic_and_wrong_substitutes(self):
        # targets at2/3, unequal context and trailing probabilities expose shifts.
        probs=torch.tensor([[.1,.9],[.8,.2],[.5,.5],[.3,.7]],dtype=torch.float64)
        logits=probs.log().unsqueeze(0).requires_grad_()
        loss,prob=r.row_unlikelihood(logits,(0,1,2,3),[1,1,0,0,1],[2,3])
        loss.backward()
        self.assertAlmostEqual(float(loss),-math.log(.6),6);self.assertAlmostEqual(float(prob),.4,6)
        self.assertAlmostEqual(float(logits.grad[0,1,0]),.1333333333,6)
        self.assertAlmostEqual(float(logits.grad[0,2,0]),.3333333333,6)
        self.assertEqual(float(logits.grad[0,0].abs().sum()+logits.grad[0,3].abs().sum()),0)
        self.assertNotAlmostEqual(float(loss),(-math.log(.2)-math.log(.5))/2,4)
        self.assertNotAlmostEqual(float(loss),-math.log(1-math.sqrt(.4)),4)
        wrong,_=r.row_unlikelihood(logits,(0,1,2,3),[1,1,0,0,1],[1,2])
        self.assertNotAlmostEqual(float(loss),float(wrong),4)

    def test_C_complete_row_extreme_confidence_and_uniform_control(self):
        # Actual C consumer, nine saved targets at their causal rows; M/F/G absent.
        vocabulary=151936
        record=dict(prompt_token_ids=[0],token_ids=list(range(1,10)))
        plan=dict(M=[],F=None,D=[dict(positions=list(range(9)))],G=[])
        positions=tuple(range(9))
        for target_logit in (40.,1000.,0.):
            with self.subTest(target_logit=target_logit):
                logits=torch.zeros(1,9,vocabulary)
                logits[0,torch.arange(9),torch.arange(1,10)]=target_logit
                logits.requires_grad_()
                loss,terms=r.image_objective('C',plan,record,None,None,None,logits,positions)
                d=terms['branches']['D']
                self.assertTrue(torch.isfinite(d));self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                target_grad=logits.grad[0,torch.arange(9),torch.arange(1,10)]
                if target_logit:
                    expected=target_logit-math.log(9*(vocabulary-1))
                    self.assertAlmostEqual(float(d.detach()),expected,places=4)
                    self.assertTrue(torch.all(target_grad>0))
                    self.assertTrue(torch.all(logits.grad[0,:,0]<0))
                    self.assertTrue(torch.allclose(target_grad,torch.full((9,),.1/9),atol=1e-6))
                else:
                    # Small-P FP32 underflow must stay zero, not acquire complement noise.
                    self.assertEqual(float(d.detach()),0.)
                    self.assertEqual(float(logits.grad.abs().sum()),0.)

    def test_UL_ordinary_regimes_match_direct_formula(self):
        for probs in ((.05,.2,.1),(.8,.5),(.95,.99),(.999,.9999)):
            logits=torch.tensor([[math.log(p),math.log1p(-p)] for p in probs]).unsqueeze(0).requires_grad_()
            ids=[1]+[0]*len(probs);positions=tuple(range(len(probs)))
            value,_=r.row_unlikelihood(logits,positions,ids,list(range(1,len(ids))))
            direct=-torch.log1p(-logits.log_softmax(-1)[0,:,0].sum().exp())
            self.assertTrue(torch.allclose(value,direct,rtol=2e-4,atol=1e-6))
            actual=torch.autograd.grad(value,logits,retain_graph=True)[0]
            expected=torch.autograd.grad(direct,logits)[0]
            self.assertTrue(torch.allclose(actual,expected,rtol=2e-4,atol=1e-6))

    def test_literal_votes_neighbors_invalids_and_cap(self):
        def row(box): return '<|object_ref_start|>person<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{i}|>' for i in box)+'<|box_end|>'
        a=row([10,10,100,100]);neighbor=row([11,10,100,100]);bad=row([999,10,999,100])
        image,record,plan=self.fixture(a+a+a+neighbor+bad+bad+'<|im_end|>',[])
        self.assertEqual(len(plan['observations']),6)
        self.assertEqual(len(plan['D']),2);self.assertEqual([d['order'] for d in plan['D']],[1,5])
        self.assertNotIn(3,[d['order'] for d in plan['D']])
        self.assertEqual(len(plan['G']),1);self.assertEqual(len(plan['negative_evidence']['complete_geometry_invalid']),2)
        self.assertFalse(plan['M']);self.assertIsNone(plan['F'])
        # Wrong geometry coordinate token or wrong parser row identity rejected.
        from probes.hidden_human_recovery import aligned_invalid
        evidence=plan['G'][0];drop=evidence['parser_drop'];encoded=r.aligned_tokens(record,self.t)
        with self.assertRaises(ValueError): aligned_invalid(record,dict(drop,generated_order=0),encoded,self.t)
        altered=copy.deepcopy(encoded);altered['input_ids'][evidence['coordinate_token_positions'][0]]=0
        with self.assertRaises(ValueError): aligned_invalid(record,drop,altered,self.t)

    def test_C_actual_consumer_unknowns_and_repeat_vote_cap(self):
        def row(box): return '<|object_ref_start|>person<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{i}|>' for i in box)+'<|box_end|>'
        a=row([10,10,100,100]);near=row([11,10,100,100]);bad=row([999,10,999,100])
        image,record,plan=self.fixture(a+a+a+near+bad+bad+'<|im_end|>',[])
        n=len(record['prompt_token_ids'])
        untouched=[n+plan['observations'][i]['positions'][0]-1 for i in (0,2,3)]
        positions=tuple(sorted(set(r.objective_positions(plan,record,'C'))|set(untouched)))
        full=record['prompt_token_ids']+record['token_ids']
        logits=torch.zeros(1,len(positions),self.vocab.vocab_size)
        for i,pos in enumerate(positions): logits[0,i,full[pos+1]]=math.log(4*(self.vocab.vocab_size-1))
        logits.requires_grad_()
        loss,terms=r.image_objective('C',plan,record,image,self.t,self.vocab,logits,positions)
        self.assertTrue(torch.allclose(loss,.1*terms['branches']['D']+.01*terms['branches']['G']))
        self.assertEqual(len(terms['row_probabilities']),2)
        self.assertTrue(all(0<v<1 for v in terms['row_probabilities']))
        loss.backward()
        for pos in untouched: self.assertEqual(float(logits.grad[0,positions.index(pos)].abs().sum()),0)
        opener=n+plan['D'][0]['positions'][0]-1
        self.assertGreater(float(logits.grad[0,positions.index(opener),full[opener+1]]),0)

    def test_unequal_positive_rows_are_row_means(self):
        objects=[dict(coco_ann_id=1,desc='person',bbox_2d=[10,10,100,100]),
                 dict(coco_ann_id=2,desc='traffic light',bbox_2d=[500,500,900,900])]
        text=''.join(r.render_row(self.visible[0],o).assistant_content_text for o in objects)+'<|im_end|>'
        image,record,plan=self.fixture(text,objects)
        seqs=[r.positive_sequence(image,record,x,self.t) for x in plan['M']]
        self.assertNotEqual(len(seqs[0].atoms),len(seqs[1].atoms))
        positions=r.objective_positions(plan,record,'A')
        logits=torch.zeros(1,len(positions),self.vocab.vocab_size,requires_grad=True)
        loss,_=r.image_objective('A',plan,record,image,self.t,self.vocab,logits,positions)
        direct=[r.p.image_loss(logits,seq,self.vocab,positions)[0] for seq in seqs]
        self.assertTrue(torch.allclose(loss,torch.stack(direct).mean()))
        weighted=sum(v*len(seq.atoms) for v,seq in zip(direct,seqs))/sum(len(seq.atoms) for seq in seqs)
        self.assertFalse(torch.allclose(loss,weighted,rtol=1e-7,atol=1e-7))

    def test_actual_geometry_dead_end_and_empty_arm_means(self):
        ids=tuple(range(1000));n=5
        e=dict(coordinate_bins=[999,20,999,10],coordinate_token_positions=[3,4,5,6],illegal_slots=[2,3])
        positions=(7,10) # x1 target8->7; y2 target11->10
        logits=torch.zeros(1,2,1002,requires_grad=True)
        loss=r.geometry_loss(logits,positions,e,n,ids);loss.backward()
        self.assertGreater(float(logits.grad[0,0,999]),0)
        self.assertLess(float(logits.grad[0,0,998]),0)
        self.assertGreater(float(logits.grad[0,1,20]),0)
        self.assertLess(float(logits.grad[0,1,21]),0)
        self.assertGreater(float(logits.grad[0,0,1001]),0) # coordinate gate
        changed=dict(e,coordinate_bins=[999,200,999,10])
        self.assertNotEqual(float(loss),float(r.geometry_loss(logits,positions,changed,n,ids)))
        empty=dict(M=[],F=None,D=[],G=[])
        x=torch.ones(1,1,self.vocab.vocab_size,requires_grad=True)
        for arm in 'ABC':
            value,_=r.image_objective(arm,empty,self.records[0],self.visible[0],self.t,self.vocab,x,(0,))
            self.assertEqual(float(value),0);self.assertTrue(value.requires_grad)
        # Actual M/F consumer uses independent row means, not pooled token means.
        plan=self.plans[0];image=self.visible[0];record=self.records[0]
        small=dict(plan,M=plan['M'][:2],D=[],G=[])
        positions=r.objective_positions(small,record,'B')
        x=torch.zeros(1,len(positions),self.vocab.vocab_size,requires_grad=True)
        fn=r.positive_sequence(image,record,small['F'],self.t,True)
        fp=tuple(a.causal_logits_position for a in fn.atoms)
        y=torch.ones(1,len(fp),self.vocab.vocab_size,requires_grad=True)
        a,_=r.image_objective('A',small,record,image,self.t,self.vocab,x,positions)
        b,terms=r.image_objective('B',small,record,image,self.t,self.vocab,x,positions,y,fp)
        self.assertTrue(torch.allclose(b,a+terms['branches']['F']))
        self.assertFalse(torch.allclose(b,(a+terms['branches']['F'])/2))
        b.backward();self.assertGreater(float(x.grad.abs().sum()),0);self.assertGreater(float(y.grad.abs().sum()),0)

    def test_family_scheduled_consumer_unequal_ranks_and_empty_F(self):
        from contextlib import contextmanager
        ids=[x['image_id'] for x in self.visible];plans={x['image_id']:x for x in self.plans}
        class Rank:
            def __init__(self): self.sync=True;self.trace=[]
            @contextmanager
            def no_sync(self):
                self.sync=False
                try:yield
                finally:self.sync=True
        for arm in 'ABCI':
            gradients=[];visited=[];wrong_rank_means=[];wrong_micro_means=[]
            for rank in range(8):
                model=Rank();parameter=torch.tensor(2.,requires_grad=True)
                jobs=r.family_jobs(ids,arm,rank);values=[]
                def forward(job):
                    i=job['image_id'];value=(ids.index(i)+1)**2
                    if job['fn']:value=(value+3) if plans[i]['F'] else 0
                    values.append(value);visited.append((i,job['fn']));model.trace.append(model.sync)
                    return parameter*value,dict(value=value)
                evidence=r.accumulate_family_step(model,jobs,forward)
                gradients.append(float(parameter.grad))
                self.assertEqual(model.trace,[False]*(len(jobs)-1)+[True])
                self.assertEqual(sum(e['sync'] for e in evidence),1)
                wrong_rank_means.append(sum(values)/len(ids[rank::8]))
                wrong_micro_means.append(sum(values)/len(values))
            expected=sum((ids.index(i)+1)**2+(((ids.index(i)+1)**2+3) if arm!='A' and plans[i]['F'] else 0) for i in ids)/18
            self.assertAlmostEqual(sum(gradients)/8,expected,5)
            self.assertNotAlmostEqual(sum(wrong_rank_means)/8,expected,4)
            if arm!='A':
                self.assertNotAlmostEqual(sum(wrong_micro_means)/8,expected,4)
                eligible=[i for i in ids if plans[i]['F']]
                wrong_F=sum((ids.index(i)+1)**2 for i in ids)/18+sum((ids.index(i)+1)**2+3 for i in eligible)/len(eligible)
                self.assertNotAlmostEqual(expected,wrong_F,4)
                self.assertEqual(len(visited),36)
                self.assertEqual(sum(plans[i]['F'] is None for i,fn in visited if fn),2)
            else:self.assertEqual(len(visited),18)
            self.assertEqual(sorted(i for i,fn in visited if not fn),sorted(ids))
            self.assertEqual([len(r.family_jobs(ids,arm,k)) for k in range(8)],([3,3,2,2,2,2,2,2] if arm=='A' else [6,6,4,4,4,4,4,4]))
        changed=copy.deepcopy(self.visible)
        for image in changed:image['hidden_truth']={'objects':[{'bbox':[0,0,999,999]}]}
        self.assertEqual(r.family_jobs(ids,'C',0),r.family_jobs([x['image_id'] for x in changed],'C',0))

    def test_logit_diagnostics_actual_consumer_preserves_training_gradient(self):
        obj=dict(coco_ann_id=-2,desc='person',bbox_2d=[10,10,100,100])
        row=r.render_row(self.visible[0],obj).assistant_content_text
        bad='<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_999|><|coord_10|><|coord_999|><|coord_100|><|box_end|>'
        image,record,plan=self.fixture(row+row+bad+'<|im_end|>',[obj])
        positions=r.objective_positions(plan,record,'C');base=torch.zeros(1,len(positions),self.vocab.vocab_size)
        full=record['prompt_token_ids']+record['token_ids']
        for j,pos in enumerate(positions):base[0,j,full[pos+1]]=math.log(4*(self.vocab.vocab_size-1))
        signal=torch.zeros_like(base);signal[0,:,0]=1
        for dtype in (torch.float32,torch.bfloat16):
            gradients=[];losses=[]
            for diagnostic in (False,True):
                parameter=torch.tensor(.2,requires_grad=True);events=[]
                parameter.register_hook(lambda grad:events.append(grad.detach().clone()))
                logits=(base+parameter*signal).to(dtype)
                loss,detail=r.image_objective('C',plan,record,image,self.t,self.vocab,logits,positions)
                if diagnostic:
                    counts=dict(M=len(plan['M']),F=0,D=len(plan['D']),G=len(plan['G']))
                    values=r.logit_diagnostics(detail['branches'],counts,logits)
                    self.assertIsNone(parameter.grad) # stop at logits, never parameter diagnostic backward
                    self.assertEqual(events,[])
                    self.assertEqual(values['F']['support_rows'],0)
                    for name in ('M','D','G'):
                        self.assertGreater(values[name]['support_rows'],0)
                        expected=values[name]['coefficient']*values[name]['unweighted']['l2']
                        self.assertAlmostEqual(values[name]['weighted']['l2'],expected,delta=.02*expected)
                loss.backward();self.assertEqual(len(events),1)
                gradients.append(parameter.grad.clone());losses.append(loss.detach())
            self.assertTrue(torch.equal(gradients[0],gradients[1]));self.assertTrue(torch.equal(losses[0],losses[1]))
        changed=dict(plan,hidden_truth={'all':'changed'},omitted_annotations=[{'class':'changed'}])
        old,old_terms=r.image_objective('C',plan,record,image,self.t,self.vocab,base,positions)
        new,new_terms=r.image_objective('C',changed,record,image,self.t,self.vocab,base,positions)
        self.assertTrue(torch.equal(old,new));self.assertEqual(r.objective_positions(plan,record,'C'),r.objective_positions(changed,record,'C'))
        self.assertTrue(all(torch.equal(old_terms['branches'][k],new_terms['branches'][k]) for k in old_terms['branches']))

    def test_family_offline_full_matcher_and_overlap_burdens(self):
        retained=dict(coco_ann_id=-2,desc='person',bbox_2d=[10,10,100,100])
        hidden=dict(coco_ann_id=1,desc='person',bbox_2d=[500,500,900,900])
        row=r.render_row(self.visible[0],retained).assistant_content_text
        near=r.render_row(self.visible[0],dict(retained,bbox_2d=[11,10,100,100])).assistant_content_text
        wrong=r.render_row(self.visible[0],dict(hidden,desc='car')).assistant_content_text
        image,zero,_=self.fixture(row+'<|im_end|>',[retained,hidden])
        _,later,_=self.fixture(row+row+near+wrong+'<|im_end|>',[retained,hidden])
        hkeys=[[image['image_id'],1]]
        scored={'zero':r.assess_outputs([image],hkeys,[zero]),'B-4':r.assess_outputs([image],hkeys,[later])}
        burdens=scored['B-4'][0]['burdens']
        self.assertEqual(burdens['literal_valid_repeats'],1);self.assertEqual(burdens['near_repeat_occurrence_pairs'],2)
        self.assertEqual(burdens['category_disagreements'],1)
        result=r.family_outcomes(scored,[[image['image_id'],-2]])
        raw=result['B-4']['summary']['raw']['combined']['metrics'];cat=result['B-4']['summary']['category']['combined']['metrics']
        self.assertEqual(raw['hidden_FN_acquired'],1);self.assertEqual(cat['hidden_FN_acquired'],0)
        self.assertEqual(cat['retained_incumbent_preserved'],1);self.assertEqual(cat['selected_FN_coverage'],1)
        self.assertEqual(cat['selected_FN_acquired'],0) # a selected-FN may already be full-matcher-covered
        self.assertEqual(cat['hidden_denominator'],1);self.assertEqual(cat['retained_denominator'],1)
        _,empty,_=self.fixture('<|im_end|>',[retained,hidden])
        scored['C-4']=r.assess_outputs([image],hkeys,[empty]);lost=r.family_outcomes(scored,[])['C-4']['summary']['category']['combined']['metrics']
        self.assertEqual(lost['retained_utility'],-1);self.assertEqual(lost['retained_incumbent_lost'],1)

    def test_offline_truth_open_waits_for_all_six_frozen_outputs(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        class ReachedTruth(Exception): pass
        opened=[];original_load=r.p.load
        with tempfile.TemporaryDirectory(dir=r.FAMILY) as tmp:
            root=Path(tmp);zero=root/'zero'
            def readback(path):
                opened.append(path);path.mkdir(parents=True,exist_ok=True)
                r.p.write(path/'frozen.json',{'fixture':'already-qualified readback seam'})
                return self.records
            def load(path):
                if Path(path)==r.TRUTH:
                    self.assertEqual(len(opened),7)
                    self.assertEqual(len(original_load(root/'all-evaluations-frozen.json')),7)
                    raise ReachedTruth()
                return original_load(path)
            with patch.object(r,'ZERO',zero),patch.object(r.p,'read_evaluation',side_effect=readback),patch.object(r.p,'load',side_effect=load):
                with self.assertRaises(ReachedTruth):r.family_offline(root)

    def test_insert_real18_targets_cuts_masks_and_terminal_substitution(self):
        frozen=r.p.load(r.ROOT/'cpu-04/credit-plan.json')
        self.assertEqual(self.plans,frozen)
        moved=r.insertion_plan(frozen,self.records)
        records={x['image_id']:x for x in self.records};images={x['image_id']:x for x in self.visible}
        selected=conflicts=rejected_terminal=0
        for old,new in zip(frozen,moved,strict=True):
            i=old['image_id'];record=records[i];image=images[i]
            self.assertEqual(new['M'],old['M'])
            self.assertEqual(r.objective_positions(new,record,'B'),r.objective_positions(old,record,'B'))
            for row in old['M']:
                self.assertEqual(r.positive_sequence(image,record,row,self.t),
                                 r.positive_sequence(image,record,new['M'][old['M'].index(row)],self.t))
            if old['F'] is None:
                self.assertEqual(new,old);continue
            selected+=1;fn=new['F'];target=tuple(fn['bbox'][:2])
            greater=[x for x in old['observations'] if tuple(x['bbox'][:2])>target]
            successor=min(greater,key=lambda x:x['order']);cut=successor['positions'][0]
            self.assertEqual(fn['prefix_cut'],cut)
            self.assertTrue(all(tuple(x['bbox'][:2])<=target for x in old['observations'] if x['order']<successor['order']))
            self.assertEqual({k:v for k,v in fn.items() if k not in ('positions','prefix_cut')},
                             {k:v for k,v in old['F'].items() if k!='positions'})
            expected=tuple(record['prompt_token_ids']+record['token_ids'][:cut]+old['F']['token_ids'])
            seq=r.positive_sequence(image,record,fn,self.t,True)
            self.assertEqual(seq.input_ids,expected)
            self.assertEqual([a.token_id for a in seq.atoms],old['F']['token_ids'])
            self.assertEqual([a.target_position for a in seq.atoms],list(range(len(record['prompt_token_ids'])+cut,len(expected))))
            self.assertTrue(all(a.causal_logits_position==a.target_position-1 for a in seq.atoms))
            self.assertTrue(all(a.token_type!='eos' for a in seq.atoms))
            self.assertEqual(seq.atoms[0].field,'object_ref_start');self.assertEqual(seq.atoms[-1].field,'box_end')
            coords=[a.to_artifact_dict()['coordinate_target'] for a in seq.atoms if a.token_type=='coordinate']
            self.assertEqual([x['bbox'] for x in coords],[fn['bbox']]*4)
            conflicts+=any(x['positions'][0]==cut for x in old['M'])
            # Actual consumer with the old terminal row is a deliberate wrong placement.
            terminal=r.positive_sequence(image,record,old['F'],self.t,True)
            with self.assertRaises(AssertionError):self.assertEqual(terminal.input_ids,expected)
            rejected_terminal+=1
        self.assertEqual((selected,conflicts,rejected_terminal),(16,9,16))
        for rank in range(8):self.assertEqual(r.family_jobs(list(images),'I',rank),r.family_jobs(list(images),'B',rank))
        changed=copy.deepcopy(self.truth)
        for image in changed:
            for obj in image['objects']:
                if [image['image_id'],obj['coco_ann_id']] in self.hidden:obj.update(desc='hidden mutation',bbox_2d=[1,2,3,4])
        visible,_=r.split_reference(changed,10)
        self.assertEqual(r.insertion_plan(r.credit_plan(visible,self.records,self.t),self.records),moved)

    def test_insert_selected_loss_own_prefix_and_empty_F(self):
        plan=r.insertion_plan([self.plans[0]],[self.records[0]])[0]
        image=self.visible[0];record=self.records[0];seq=r.positive_sequence(image,record,plan['F'],self.t,True)
        positions=tuple(a.causal_logits_position for a in seq.atoms)+(len(seq.input_ids)-1,)
        logits=torch.zeros(1,len(positions),self.vocab.vocab_size,requires_grad=True)
        loss,detail=r.p.image_loss(logits,seq,self.vocab,positions);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(logits.grad[0,j].abs().sum()>0 for j in range(len(positions)-1)))
        self.assertEqual(float(logits.grad[0,-1].abs().sum()),0)
        small=dict(plan,M=[],F=None,D=[],G=[])
        value,terms=r.image_objective('B',small,record,image,self.t,self.vocab,logits,positions)
        self.assertEqual(float(value),0);self.assertEqual(float(terms['branches']['F']),0)

    def test_insert_offline_binds_controls_and_freezes_before_truth(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        class ReachedTruth(Exception):pass
        opened=[];original_load=r.p.load
        with tempfile.TemporaryDirectory(dir=r.ROOT/'family-insert-01') as tmp:
            root=Path(tmp);reference=root/'reference';zero=root/'zero'
            r.p.write(root/'control-reuse.json',dict(reference_root=str(reference),sha256={}))
            expected={zero,*[reference/f'evaluation-{arm}-{step}' for arm in 'AB' for step in (4,8)],
                      *[root/f'evaluation-I-{step}' for step in (4,8)]}
            def readback(path):
                opened.append(path);path.mkdir(parents=True,exist_ok=True);r.p.write(path/'frozen.json',{})
                return self.records
            def load(path):
                if Path(path)==r.TRUTH:
                    self.assertEqual(set(opened),expected);self.assertEqual(len(opened),7)
                    self.assertEqual(len(original_load(root/'all-evaluations-frozen.json')),7)
                    raise ReachedTruth()
                return original_load(path)
            with patch.object(r,'ZERO',zero),patch.object(r.p,'read_evaluation',side_effect=readback),patch.object(r.p,'load',side_effect=load):
                with self.assertRaises(ReachedTruth):r.family_offline(root,reference)

    def test_retained_real18_identity_order_and_hidden_independence(self):
        enc=r.p.load(r.ROOT/'retained-sft-01/encodings.json')
        images={x['image_id']:x for x in self.visible};keys=[]
        for e in enc:
            image=images[e['image_id']]
            ordered=sorted(image['objects'],key=lambda o:(o['bbox_2d'][0],o['bbox_2d'][1],o['coco_ann_id']))
            self.assertEqual(e['annotation_ids'],[o['coco_ann_id'] for o in ordered])
            self.assertEqual(e['row_ids'],['retained:'+str(o['coco_ann_id']) for o in ordered])
            self.assertEqual(e['prefix_source'],'rendered_retained_labels')
            for oid,obj in zip(e['row_ids'],ordered,strict=True):
                atoms=[a for a in e['atoms'] if a['object_id']==oid]
                self.assertEqual(self.t.decode([a['token_id'] for a in atoms if a['token_type']=='desc_text']),obj['desc'])
                coords=[a['coordinate_target'] for a in atoms if a['token_type']=='coordinate']
                self.assertEqual([x['bbox'] for x in coords],[obj['bbox_2d']]*4)
                self.assertEqual([x['slot_index'] for x in coords],[0,1,2,3])
                self.assertEqual(atoms[0]['field'],'object_ref_start');self.assertEqual(atoms[-1]['field'],'box_end')
                keys.append([e['image_id'],obj['coco_ann_id']])
            for atom in e['atoms']:
                self.assertEqual(e['input_ids'][atom['target_position']],atom['token_id'])
                self.assertEqual(atom['target_position']-1,atom['causal_logits_position'])
                self.assertNotEqual(atom['token_type'],'eos')
        self.assertEqual(len(enc),18);self.assertEqual(len(keys),513);self.assertEqual(len(set(map(tuple,keys))),513)
        self.assertFalse(set(map(tuple,keys))&set(map(tuple,self.hidden)))
        changed=copy.deepcopy(self.truth)
        for image in changed:
            for obj in image['objects']:
                if [image['image_id'],obj['coco_ann_id']] in self.hidden:obj.update(desc='mutated hidden',bbox_2d=[1,2,3,4])
        retained,_=r.split_reference(changed,10);self.assertEqual(retained,self.visible)
        _,before,ids=r.retained_sequence(self.visible[0],self.q)
        _,after,newids=r.retained_sequence(retained[0],self.q)
        self.assertEqual(before,after);self.assertEqual(ids,newids)
        for rank in range(8):self.assertEqual(r.family_jobs(list(images),'S',rank),r.family_jobs(list(images),'A',rank))

    def test_retained_actual_consumer_closure_EOS_and_omission(self):
        from src.packing.planner import PackedSequence
        from src.supervision.tokens import build_token_sequence_from_packed_supervision
        image=next(x for x in self.visible if x['image_id']==7116)
        _,seq,ids=r.retained_sequence(image,self.q)
        terminal=range(max(a.target_position for a in seq.atoms)+1,len(seq.input_ids))
        positions=tuple(a.causal_logits_position for a in seq.atoms)+tuple(i-1 for i in terminal)
        logits=torch.zeros(1,len(positions),self.vocab.vocab_size,requires_grad=True)
        loss,rows=r.retained_objective(logits,seq,ids,self.vocab,positions);loss.backward()
        self.assertEqual(len(rows),len(image['objects']));self.assertTrue(torch.isfinite(loss))
        for atom in seq.atoms:
            if atom.field in ('object_ref_start','box_end'):
                self.assertGreater(float(logits.grad[0,positions.index(atom.causal_logits_position)].abs().sum()),0)
        for pos in terminal:self.assertEqual(float(logits.grad[0,positions.index(pos-1)].abs().sum()),0)
        pack=PackedSequence(seq.pack_index,seq.input_ids,seq.segments,r.p.MAX_LENGTH)
        omitted=build_token_sequence_from_packed_supervision(pack,[a for a in seq.atoms if a.object_id!=ids[-1]])
        with self.assertRaises(AssertionError):r.retained_objective(logits,omitted,ids,self.vocab,positions)
        end=next(iter(terminal))
        leaked=replace(seq.atoms[-1],target_position=end,logical_target_position=end,logical_target_end=end+1,
                       token_id=seq.input_ids[end],token_type='eos',text='<|im_end|>',field='eos',coordinate_target=None)
        leak=build_token_sequence_from_packed_supervision(pack,seq.atoms+(leaked,))
        with self.assertRaises(AssertionError):r.retained_objective(logits,leak,ids,self.vocab,positions)

    def test_retained_unequal_real_row_and_image_means(self):
        means=[];all_rows=[];pooled_tokens=[]
        for index,i in enumerate((7116,13348)):
            image=next(x for x in self.visible if x['image_id']==i)
            _,seq,ids=r.retained_sequence(image,self.q)
            positions=tuple(a.causal_logits_position for a in seq.atoms)
            logits=torch.zeros(1,len(positions),self.vocab.vocab_size)
            for j,atom in enumerate(seq.atoms):logits[0,j,atom.token_id]=1+index+ids.index(atom.object_id)*.2
            loss,rows=r.retained_objective(logits,seq,ids,self.vocab,positions)
            self.assertAlmostEqual(float(loss),sum(x['loss'] for x in rows)/len(rows),5)
            means.append(float(loss));all_rows.extend(rows)
            pooled_tokens.append(sum(x['loss']*x['atoms'] for x in rows)/sum(x['atoms'] for x in rows))
        expected=sum(means)/2
        self.assertNotAlmostEqual(expected,sum(x['loss'] for x in all_rows)/len(all_rows),4)
        self.assertNotAlmostEqual(expected,sum(pooled_tokens)/2,4)

    def test_retained_offline_freeze_and_zero_A_I_contrasts(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        original_load=r.p.load;opened=[]
        with tempfile.TemporaryDirectory(dir=r.ROOT/'retained-sft-01') as tmp:
            root=Path(tmp);reference=root/'A';insertion=root/'I';zero=root/'zero'
            r.p.write(root/'control-reuse.json',dict(arm='S',reference_root=str(reference),insertion_root=str(insertion),sha256={}))
            expected={zero,*[reference/f'evaluation-A-{s}' for s in (4,8)],*[insertion/f'evaluation-I-{s}' for s in (4,8)],*[root/f'evaluation-S-{s}' for s in (4,8)]}
            def readback(path):
                opened.append(path);path.mkdir(parents=True,exist_ok=True);r.p.write(path/'frozen.json',{})
                return self.records
            def load(path):
                if Path(path)==r.TRUTH:
                    self.assertEqual(set(opened),expected);self.assertEqual(len(opened),7)
                    self.assertEqual(len(original_load(root/'all-evaluations-frozen.json')),7)
                return original_load(path)
            with patch.object(r,'ZERO',zero),patch.object(r.p,'read_evaluation',side_effect=readback),patch.object(r.p,'load',side_effect=load):r.family_offline(root,reference)
            result=original_load(root/'offline-results.json')
            self.assertEqual(set(result['contrasts']),{f'S-{a}-{s}' for a in ('zero','A','I') for s in (4,8)})
            for contrast in result['contrasts'].values():
                for mode in contrast.values():
                    for cohort in mode.values():
                        self.assertTrue(all(v==0 for part in cohort.values() for v in part.values()))

    def test_dose_freezes_all_outputs_before_truth(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ref=root/'reference';ref.mkdir();opened=[]
            r.p.write(root/'control-reuse.json',dict(reference_root=str(ref),dose_updates=16,retained_root=str(ref),sha256={}))
            original=r.p.load
            def readback(path):
                opened.append(path);path.mkdir(exist_ok=True)
                r.p.write(path/'frozen.json',{});return self.records
            def load(path):
                if Path(path)==r.TRUTH:
                    self.assertEqual(set(opened),{root/'zero',ref/'evaluation-S-8',root/'evaluation-S-16'})
                    self.assertEqual(len(original(root/'all-evaluations-frozen.json')),3)
                    raise RuntimeError('truth reached after freeze')
                return original(path)
            with patch.object(r,'ZERO',root/'zero'),patch.object(r.p,'read_evaluation',side_effect=readback),patch.object(r.p,'load',side_effect=load):
                with self.assertRaisesRegex(RuntimeError,'truth reached after freeze'):r.family_offline(root,ref)

    def test_dose_schedule_and_checkpoint_gate(self):
        import tempfile
        from pathlib import Path
        from safetensors.torch import save_file
        self.assertEqual(r.family_schedule(),r.family_schedule(16)[:8])
        self.assertEqual([x['step'] for x in r.family_schedule(16) if x['save']],[4,8,16])
        self.assertEqual([x['step'] for x in r.family_schedule(16) if x['diagnostics']],[1,8])
        for step in r.family_schedule(16):
            jobs=[j for rank in range(8) for j in r.family_jobs(list(range(18)),'S',rank)]
            self.assertEqual(len(jobs),18);self.assertTrue(all(j['weight']==8/18 and not j['fn'] for j in jobs))
        with self.assertRaises(ValueError):r.family_schedule(32)
        with tempfile.TemporaryDirectory() as d:
            roots=[Path(d)/x for x in ('new','old')]
            for root in roots:
                root.mkdir();save_file({'delta':torch.ones(2)},str(root/'weights.safetensors'))
                r.p.write(root/'identity.json',{'weights.safetensors':r.p.digest(root/'weights.safetensors')})
            binding={str(roots[1]/'identity.json'):r.p.digest(roots[1]/'identity.json')}
            self.assertEqual(r.checkpoint_gate(*roots,binding)['tensors'],1)
            save_file({'delta':torch.zeros(2)},str(roots[0]/'weights.safetensors'))
            (roots[0]/'identity.json').unlink();r.p.write(roots[0]/'identity.json',{'weights.safetensors':r.p.digest(roots[0]/'weights.safetensors')})
            with self.assertRaises(AssertionError):r.checkpoint_gate(*roots,binding)
            (roots[1]/'identity.json').write_text('{}')
            with self.assertRaises(AssertionError):r.checkpoint_gate(*roots,binding)

    def test_tail_event_boundary_and_hidden_independence(self):
        from unittest.mock import patch
        row=r.render_row(self.visible[0],dict(desc='person',bbox_2d=[1,2,3,4])).assistant_content_text
        image,record,_=self.fixture(row+row+'<|im_end|>')
        record['generated_tokens']=len(record['token_ids'])
        with patch.object(r.p,'load',side_effect=AssertionError('truth/file access forbidden')):
            prefix,event=r.tail_prefix(record,self.t,'literal_repeat')
            self.assertEqual(prefix['text'],row);self.assertEqual(event['event']['order'],1)
            unchanged,noevent=r.tail_prefix(record,self.t,'geometry_invalid')
            self.assertEqual(unchanged,record);self.assertIsNone(noevent['event'])
        import re
        spans=list(re.finditer(r'<\|coord_\d+\|>',row))
        self.assertEqual(len(spans),4)
        bad=row[:spans[2].start()]+spans[0].group()+row[spans[2].end():]
        _,invalid,_=self.fixture(row+bad+bad+'<|im_end|>')
        invalid['generated_tokens']=len(invalid['token_ids'])
        first,cut=r.tail_prefix(invalid,self.t,'geometry_invalid')
        self.assertEqual(first['text'],row);self.assertEqual(cut['event']['order'],1)
        repeat,cut=r.tail_prefix(invalid,self.t,'literal_repeat')
        self.assertEqual(repeat['text'],row+bad);self.assertEqual(cut['event']['order'],2)
        self.assertNotEqual(prefix['text'],record['text'])


if __name__=='__main__': unittest.main()
