import copy
import json
from pathlib import Path
import tempfile
import unittest
from probes import hidden_human_recovery as probe


def fixture():
    public = dict(image_id=1, image_path='/image.jpg', image_sha256='abc', width=1024, height=1024,
                  cohort='fixture', objects=[dict(coco_ann_id=2,bbox_2d=[1,2,30,40],desc='person',category_id=1,category_name='person'),
                  dict(coco_ann_id=-1,bbox_2d=[100,200,300,400],desc='bird',category_id=16,category_name='bird')])
    policy = dict(seeds=[1,2,3,4],temperature=0.7)
    raw = [dict(request_id='1:full:0',image_id=1,arm='full',width=1024,height=1024,crop=[0,0,1024,1024],
        text='<|object_ref_start|>bird<|object_ref_end|><|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>',
        actual_pixels=1024**2,visual_tokens=1024,generated_tokens=9,model_calls=1,generation_seconds=1.0,prepare_seconds=0.1,stop_reason='im_end')]
    return [public], policy, raw


class RecoveryCPU(unittest.TestCase):
    def test_hidden_sensitivity_and_deliberate_wrong_dependency(self):
        records, policy, raw = fixture()
        changed = copy.deepcopy(records)
        changed[0]['objects'][1] = dict(coco_ann_id=-999,bbox_2d=[400,500,600,700],desc='boat',category_id=9,category_name='boat')
        changed[0]['objects'].append(dict(coco_ann_id=-998,bbox_2d=[1,1,10,10],desc='dog',category_id=18,category_name='dog'))
        def boundary(bank):
            visible, _ = probe.split_views(bank)
            return probe.request_plan(visible,policy), probe.admission_inputs(visible,raw)
        self.assertEqual(boundary(records), boundary(changed))
        # The same equality oracle must reject a deliberately leaked hidden count.
        def wrong_boundary(bank):
            requests, admission = boundary(bank)
            requests[0]['hidden_count'] = len(bank[0]['objects'])
            return requests, admission
        with self.assertRaises(AssertionError):
            self.assertEqual(wrong_boundary(records), wrong_boundary(changed))
        self.assertNotEqual(probe.split_views(records)[1], probe.split_views(changed)[1])

    def test_truth_view_rejected_at_acquisition_boundary(self):
        records, policy, raw = fixture()
        with self.assertRaisesRegex(ValueError, 'visible'):
            probe.request_plan(records, policy)
        with self.assertRaisesRegex(ValueError, 'visible'):
            probe.admission_inputs(records, raw)

    def test_mapping_and_annotation_proxy_readback(self):
        records, policy, raw = fixture()
        _, truth = probe.split_views(records)
        output = probe.evaluate(truth,raw,[])
        report = next(r for r in output['reports'] if r['arm']=='full' and r['stage']=='raw')
        self.assertEqual(report['new_hidden_beyond_greedy'],['-1'])
        self.assertEqual(next(r for r in output['reports'] if r['arm']=='full' and r['stage']=='admitted')['hidden_recovered'],[])
        raw[0].update(arm='region',crop=[512,0,1024,512])
        candidate = probe.candidates(raw)[0][0]
        self.assertEqual(candidate['coord_bins_1000'],[550,100,650,200])

    def test_scheduled_crop_norm1000_hidden_recovery(self):
        records, _, raw = fixture()
        records[0]['objects'][1]['bbox_2d'] = [383,100,394,200]
        _, truth = probe.split_views(records)
        raw[0].update(request_id='1:region:1', arm='region', crop=[384,0,1024,640],
                      text='<|object_ref_start|>bird<|object_ref_end|><|box_start|><|coord_8|><|coord_160|><|coord_24|><|coord_320|><|box_end|>')
        output = probe.evaluate(truth, raw, [])
        report = next(r for r in output['reports'] if r['arm']=='region' and r['stage']=='raw')
        self.assertEqual(report['hidden_denominator'], 1)
        self.assertEqual(report['new_hidden_beyond_greedy'], ['-1'])
        self.assertEqual(report['matches'][0]['iou'], 0.5)
        self.assertEqual(probe.candidates(raw)[0][0]['coord_bins_1000'], [380,100,390,200])
        raw[0]['crop'] = [0,384,640,1024]
        self.assertEqual(probe.candidates(raw)[0][0]['coord_bins_1000'], [5,475,15,575])
        raw[0]['crop'] = [0,0,1024,1024]
        self.assertEqual(probe.candidates(raw)[0][0]['coord_bins_1000'], [8,160,24,320])
        raw[0]['crop'] = [384,0,1024,640]
        raw[0]['text'] = raw[0]['text'].replace('coord_8|', 'coord_9|')
        self.assertEqual(probe.candidates(raw)[0][0]['coord_bins_1000'][0], 380.625)

    def test_truth_changes_score_not_selection(self):
        records, policy, raw = fixture()
        _, truth = probe.split_views(records)
        before = probe.evaluate(truth,raw,[])
        changed = copy.deepcopy(truth)
        changed[0]['objects'][1]['bbox_2d']=[600,600,900,900]
        after = probe.evaluate(changed,raw,[])
        self.assertNotEqual(before,after)

    def test_frozen_readback_rejects_tamper(self):
        _,_,raw=fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            probe.write(p/'1-full-0.json',raw[0])
            probe.write(p/'frozen.json',dict(status='generated',requests={'1:full:0':probe.digest(p/'1-full-0.json')}))
            self.assertEqual(probe.read_frozen(p),raw)
            (p/'1-full-0.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):
                probe.read_frozen(p)

    def test_sharding_and_complete_readback(self):
        from src.inference.data_parallel import plan_data_parallel_shards
        ids = [str(i) for i in range(18)]
        plan = plan_data_parallel_shards(row_ids=ids, per_device_batch_size=1, visible_cuda_tokens=[str(i) for i in range(8)])
        self.assertEqual(plan.active_ranks,8)
        self.assertEqual(sorted(i for rank in plan.ranks for i in rank.row_indices),list(range(18)))
        _,_,raw=fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            shard=root/'rank-000';shard.mkdir()
            probe.write(shard/'1-full-0.json',raw[0])
            manifest=dict(status='generated',requests={'1:full:0':probe.digest(shard/'1-full-0.json')},global_request_ids=['1:full:0','1:full:1'],visible_sha256='v',policy_sha256='p',source_identity={})
            probe.write(shard/'frozen.json',manifest)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                probe.read_frozen(root)
            shard2=root/'rank-001';shard2.mkdir()
            second={**raw[0],'request_id':'1:full:1'}
            probe.write(shard2/'1-full-1.json',second)
            probe.write(shard2/'frozen.json',{**manifest,'requests':{'1:full:1':probe.digest(shard2/'1-full-1.json')}})
            self.assertEqual(probe.read_frozen(root),[raw[0],second])

    def test_unreleased_policy_cannot_load_model(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            probe.write(root/'visible.json',[])
            probe.write(root/'policy.json',{'policy_status':'proposal_pending_lead'})
            with self.assertRaisesRegex(ValueError,'not released'):
                probe.execute(root/'visible.json',root/'policy.json',root/'run',[],generate=True)
            self.assertFalse((root/'run').exists())


class CandidateLocalCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.root=Path('/data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-26')
        cls.raw=probe.read_frozen(cls.root/'pilot-01')
        cls.visible=probe.load(cls.root/'preparation-v3/acquisition/visible.json')
        cls.tokenizer=AutoTokenizer.from_pretrained(probe.BASE,local_files_only=True)

    def test_real18_hidden_invariance_and_wrong_dependency(self):
        truth=probe.load(self.root/'preparation-v3/evaluator/truth.json')
        changed=copy.deepcopy(truth)
        for r in changed:
            r['objects']=[o for o in r['objects'] if o['coco_ann_id']>=0]+[
                dict(coco_ann_id=-987,bbox_2d=[1,2,3,4],desc='changed',category_id=999,category_name='changed')]
            r['hidden_objects']=r['objects'][-1:]
        def boundary(records):
            visible,_=probe.split_views(records)
            return probe.local_plan(visible,self.raw),probe.negative_evidence(self.raw,self.tokenizer)
        before=boundary(truth);after=boundary(changed)
        self.assertEqual(before,after)
        self.assertEqual(len(before[0]['bank']),2111)
        self.assertLessEqual(len(before[0]['selected']),216)
        self.assertEqual(len(before[0]['requests']),2*len(before[0]['selected']))
        self.assertEqual(len(before[1]['complete_geometry_invalid']),342)
        self.assertEqual(len(before[1]['malformed_or_censored']),2)
        with self.assertRaises(AssertionError):
            self.assertEqual((before,len(truth[0]['objects'])),(after,len(changed[0]['objects'])))

    def test_local_caller_offset_edge_and_two_scales(self):
        from unittest.mock import patch
        from PIL import Image
        from types import SimpleNamespace
        records,_,raw=fixture();visible,_=probe.split_views(records)
        raw[0]['text']=raw[0]['text'].replace('coord_100','coord_0').replace('coord_200','coord_500').replace('coord_300','coord_50').replace('coord_400','coord_600')
        plan=probe.local_plan(visible,raw)
        self.assertEqual(len(plan['requests']),2)
        self.assertEqual(plan['requests'][0]['crop'],[0,448,128,672])
        with tempfile.TemporaryDirectory() as d:
            image=Path(d)/'image.png';Image.new('RGB',(1024,1024)).save(image)
            processor=SimpleNamespace(apply_chat_template=lambda *a,**kw:'fixed class-blind prompt')
            seen=[]
            def prepare(processor,requests,**kw):
                # Inspect the actual NativeRequest image, before its owning caller closes it.
                seen.append(requests[0].image.size)
                return requests[0].image.size
            for r in plan['requests']:
                r.update(image_path=str(image),image_sha256=probe.digest(image))
                with patch('src.qwen.native.prepare_native_inputs',side_effect=prepare):
                    probe.native_request(r,{'prompt':{'system':'s','user':'u'}},processor)
                mapped=probe.candidates([{**r,'text':fixture()[2][0]['text']}])[0][0]
                self.assertEqual(mapped['coord_bins_1000'],[12.5,481.25,37.5,525.0])
            self.assertEqual(seen,[(128,224),(256,448)])

    def test_real_token_row_alignment_rejection_and_dead_end(self):
        _,drops=probe.candidates(self.raw)
        d=next(x for x in drops if x['reason']=='geometry_invalid')
        r=next(x for x in self.raw if x['request_id']==d['request_id'])
        encoded=self.tokenizer(r['text'],add_special_tokens=False,return_offsets_mapping=True)
        probe.aligned_invalid(r,d,encoded,self.tokenizer)
        bad=copy.deepcopy(r);bad['token_ids'][0]+=1
        with self.assertRaisesRegex(ValueError,'alignment'):
            probe.aligned_invalid(bad,d,encoded,self.tokenizer)
        wrong=copy.deepcopy(d);wrong['generated_order']+=1
        with self.assertRaisesRegex(ValueError,'alignment'):
            probe.aligned_invalid(r,wrong,encoded,self.tokenizer)
        synthetic=copy.deepcopy(r)
        synthetic['text']='<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_999|><|coord_0|><|coord_999|><|coord_999|><|box_end|>'
        synthetic['token_ids']=self.tokenizer.encode(synthetic['text'],add_special_tokens=False)
        bank=probe.negative_evidence([synthetic],self.tokenizer)['complete_geometry_invalid']
        self.assertEqual(bank[0]['empty_legal_set_predecessor_slots'],[0])
        self.assertLess(bank[0]['earlier_dead_end_token_positions'][0],bank[0]['first_illegal_token_position'])
        synthetic['text']=synthetic['text'].replace('<|coord_999|>','<|coord_0|>',1)
        synthetic['token_ids']=self.tokenizer.encode(synthetic['text'],add_special_tokens=False)
        self.assertEqual(probe.negative_evidence([synthetic],self.tokenizer)['complete_geometry_invalid'],[])


if __name__=='__main__':
    unittest.main()
