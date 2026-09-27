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


if __name__=='__main__': unittest.main()
