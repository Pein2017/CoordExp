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


if __name__=='__main__':
    unittest.main()
