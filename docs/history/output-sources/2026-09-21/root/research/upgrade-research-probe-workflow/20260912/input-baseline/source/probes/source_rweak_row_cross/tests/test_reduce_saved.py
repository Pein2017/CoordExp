"""Native save/reload/parser/matcher fixture plus decision-bearing counterexamples."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import os
import pytest
from probes.source_rweak_row_cross import reduce as r

MANIFEST = Path(os.environ.get('ROW_CROSS_MANIFEST', '/nonexistent/manifest.json'))
ORIGINAL_CODE_ROOT = os.environ.get('ROW_CROSS_ORIGINAL_CODE_ROOT')
pytestmark = pytest.mark.skipif(not MANIFEST.is_file(), reason='requires explicit saved row-cross manifest and local tokenizer')


class ReducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=r.load_manifest(MANIFEST, original_code_root=ORIGINAL_CODE_ROOT)
        from transformers import AutoTokenizer
        cls.tokenizer=AutoTokenizer.from_pretrained(cls.manifest['sources']['source']['config']['model']['base_model'],local_files_only=True)
        cls.case=cls.manifest['cases'][0]
        cls.tempdir=tempfile.TemporaryDirectory(prefix='row-cross-consumer-')
        cls.root=Path(cls.tempdir.name)
        cls.cross=cls.root/'cross'
        cls.cross.mkdir()
        cls.records=[cls.record(cell) for cell in ('01','10')]
        (cls.cross/'rows.jsonl').write_text(''.join(json.dumps(record)+'\n' for record in cls.records))

    @classmethod
    def record(cls,cell):
        case=cls.case
        recipient,donor=r.CELLS[cell]
        prefix=case['common_prefix_token_ids']
        action=case['actions'][donor]['token_ids']
        suffix=[r.EOS]
        ids=prefix+action+suffix
        text=r.decode(cls.tokenizer,ids)
        parsed=r.parse_raw(text,case['diagonals']['source']['raw_record'])
        parsed['decode_stop_reason']='im_end'
        return {'case_id':case['row_id'],'recipient':recipient,'action_source':donor,'cell':cell,'mode':'qualify',
                'manifest_sha256':r.MANIFEST_SHA256,'prefix_token_ids':prefix,'action_token_ids':action,
                'suffix_token_ids':suffix,'generated_token_ids':ids,'raw_decode_text':text,'decode_stop_reason':'im_end',
                'generated_token_count':len(ids),'remaining_budget':r.CAP-len(prefix)-len(action),'parsed':parsed,'synthetic_fixture':True}

    def test_real_saved_reload_native_parser_matcher_consumer(self):
        result=r.reduce_files(MANIFEST,[self.cross],[self.case['row_id']], original_code_root=ORIGINAL_CODE_ROOT)
        self.assertEqual(result['scope'],'qualification_subset')
        self.assertEqual(result['complete_output_count'],4)
        self.assertTrue(result['synthetic_fixture'])
        self.assertEqual(result['aggregate']['00']['thresholds']['iou_0.50']['matched_gt_ids'],self.case['diagonals']['source']['matched_gt_ids']['iou_0.50'])
        output=self.root/'reduction.json'
        output.write_text(json.dumps(result,sort_keys=True))
        self.assertEqual(json.loads(output.read_text()),result)

    def test_subset_not_silently_final(self):
        with self.assertRaisesRegex(ValueError,'incomplete cross'):
            r.reduce_files(MANIFEST,[self.cross], original_code_root=ORIGINAL_CODE_ROOT)

    def test_duplicate_case_cell_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate cross'):
            r.reduce_files(MANIFEST,[self.cross,self.cross],[self.case['row_id']], original_code_root=ORIGINAL_CODE_ROOT)

    def test_corruption_cases(self):
        corruptions=[
            ('token',lambda x:x['generated_token_ids'].__setitem__(0,42),'concatenation'),
            ('budget',lambda x:x.__setitem__('remaining_budget',1),'remaining budget'),
            ('identity',lambda x:x['parsed'].__setitem__('image_path','wrong-image'),'image_path'),
            ('recipient',lambda x:x.__setitem__('recipient','rweak'),'donor/recipient'),
            ('hash',lambda x:x.__setitem__('manifest_sha256','bad'),'manifest hash'),
            ('text',lambda x:x.__setitem__('raw_decode_text','wrong'),'token/text'),
            ('prediction',lambda x:x['parsed']['pred'].__setitem__(0,{}),'pred'),
        ]
        for label,mutate,expected in corruptions:
            with self.subTest(label=label):
                record=copy.deepcopy(self.records[0]);mutate(record)
                with self.assertRaisesRegex(ValueError,expected):
                    r.validate_cross(record,self.case,self.tokenizer,r.MANIFEST_SHA256)

    def test_manifest_corruption_rejected(self):
        path=self.root/'corrupt-manifest.json'
        path.write_bytes(MANIFEST.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'frozen manifest hash'):
            r.load_manifest(path)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()
