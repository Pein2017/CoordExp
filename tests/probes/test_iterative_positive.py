import copy
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from probes import iterative_positive as p
from src.losses.vocab import build_token_vocabulary_groups
from src.qwen.runtime_loading import QwenLoadOptions,load_qwen_components_from_options
from src.supervision.tokens import TokenSequence


class IterativePositiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=p.load(p.ROOT/'learning-plan.json')
        cls.q=load_qwen_components_from_options(QwenLoadOptions(p.load(p.POLICY)['base_model'],'fp32','sdpa',load_model=False))
        cls.vocab=build_token_vocabulary_groups(cls.q.token_identity,tokenizer=cls.q.tokenizer)

    def test_real_diversity_and_hidden_independence(self):
        bank=p.load(p.ROOT/'candidates.json')
        selected=p.diverse_selection(bank)
        altered=copy.deepcopy(bank)
        altered['hidden_truth']=[{'box':[0,0,999,999],'category':'changed'}]
        self.assertEqual(selected,p.diverse_selection(altered))
        self.assertEqual(selected,self.plan['selection'])
        for image in self.plan['human_ids']:
            kept=[c for c in selected['selected'] if c['image_id']==image]
            self.assertLessEqual(len(kept),4)
            for j,a in enumerate(kept):
                for b in kept[j+1:]:
                    if a['description']==b['description']:
                        self.assertLess(p.iou_xyxy(a['quantized_full_image_bins'],b['quantized_full_image_bins']),.8)
        # Frozen original top four contain a counterexample in all three images.
        for image in (1584,2299,14439):
            old=[c for c in bank['candidates'] if c['image_id']==image and c['disposition']=='selected_experimental_addition']
            self.assertTrue(any(a['description']==b['description'] and p.iou_xyxy(a['quantized_full_image_bins'],b['quantized_full_image_bins'])>=.8 for j,a in enumerate(old) for b in old[j+1:]))

    def test_actual_consumer_mask_and_three_losses(self):
        raw,selected=p.raw_for(self.plan,'pseudo',1584)
        # Two real rows expose selected opener/local closure and a competing omitted row.
        kept=[next(o for o in raw.objects if o.object_id not in selected),next(o for o in raw.objects if o.object_id in selected)]
        raw=replace(raw,objects=tuple(sorted(kept,key=lambda o:(o.bbox[0],o.bbox[1],o.object_id))))
        selected={o.object_id for o in kept if o.object_id in selected}
        _,full=p.encode(raw,None,self.q)
        encoded,masked=p.encode(raw,selected,self.q)
        own=[a for a in full.atoms if a.object_id in selected]
        self.assertEqual(masked.atoms,tuple(own))
        self.assertEqual(own[0].field,'object_ref_start')
        self.assertEqual(own[-1].field,'box_end')
        self.assertTrue(any(a.token_type=='eos' for a in full.atoms))
        positions=tuple(a.causal_logits_position for a in full.atoms)
        logits=torch.randn(1,len(positions),self.vocab.vocab_size,requires_grad=True)
        loss,terms=p.image_loss(logits,masked,self.vocab,positions)
        loss.backward()
        allowed={a.causal_logits_position for a in own}
        for j,position in enumerate(positions):
            self.assertEqual(bool(logits.grad[0,j].abs().sum()>0),position in allowed)
        self.assertTrue(all(v>0 for v in terms.values()))
        # Deliberate wrong dependency: unmasking completion/nonselected rows changes loss.
        wrong,_=p.image_loss(logits,full,self.vocab,positions)
        self.assertNotEqual(float(wrong),float(loss))
        corrupted=list(masked.input_ids)
        first=next(a for a in own if a.field=='bbox[0]')
        corrupted[first.target_position]+=1
        with self.assertRaises(Exception):
            p.image_loss(logits,replace(masked,input_ids=tuple(corrupted)),self.vocab,positions)
        empty=replace(masked,atoms=(),spans=())
        zero,_=p.image_loss(logits,empty,self.vocab,positions)
        self.assertEqual(float(zero),0)
        self.assertTrue(zero.requires_grad)
        mutated=copy.deepcopy(self.plan);mutated['hidden_truth']={'changed':True}
        raw2,ids2=p.raw_for(mutated,'pseudo',1584)
        a,seq=p.encode(*p.raw_for(self.plan,'pseudo',1584),self.q)
        b,seq2=p.encode(raw2,ids2,self.q)
        self.assertEqual(a.input_ids,b.input_ids);self.assertEqual(seq,seq2)

    def test_schedule_and_separate_gradient_means(self):
        plan=self.plan
        excluded={r['image_id'] for r in plan['visible']}
        self.assertFalse(excluded.intersection(plan['replay_ids']))
        self.assertEqual(len(set(plan['replay_ids'])),128)
        self.assertEqual(plan['schedule'],p.make_schedule(plan['human_ids'],plan['replay_ids']))
        x=torch.tensor(2.,requires_grad=True)
        common=[];pseudo=[];distributed=0
        step=plan['schedule'][0]
        for rank in range(8):
            jobs=p.rank_examples(step,rank,treatment=True)
            self.assertEqual(jobs[0][1],jobs[-1][1])
            for branch,i,weight in jobs:
                value=x*x*(i%17+1)
                distributed=distributed+value*weight/8
                (pseudo if branch=='pseudo' else common).append(value)
        direct=torch.stack(common).mean()+.05*torch.stack(pseudo).mean()
        self.assertTrue(torch.allclose(distributed,direct))
        self.assertTrue(torch.allclose(torch.autograd.grad(distributed,x,retain_graph=True)[0],torch.autograd.grad(direct,x)[0]))
        wrong=torch.stack(common+pseudo).mean()
        self.assertFalse(torch.allclose(wrong,direct))
        control=[p.rank_examples(step,r) for r in range(8)]
        self.assertEqual([j for jobs in control for j in jobs], [j for r in range(8) for j in p.rank_examples(step,r,treatment=True) if j[0]!='pseudo'])


if __name__=='__main__': unittest.main()
