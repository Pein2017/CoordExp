"""Native save/reload/parser/matcher fixture plus decision-bearing counterexamples."""
import unittest

from probes.source_rweak_row_cross import reduce as r



def textrow(category,box):
    return '<|object_ref_start|>'+category+'<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{i}|>' for i in box)+'<|box_end|>'


def identity(boxes):
    return {'row_id':'fixture','row_index':0,'example_id':'fixture','image_path':'synthetic-no-image-load',
            'image_width':1000,'image_height':1000,'gt':[{'object_id':str(i+1),'description':'person','bbox':list(box)} for i,box in enumerate(boxes)]}


class ReducerTests(unittest.TestCase):
    def test_terminal_direct_b_vs_c_is_not_successor_failure(self):
        ident=identity([(0,0,100,100),(200,0,300,100)])
        btext=textrow('person',(0,0,100,100))
        ctext=textrow('person',(200,0,300,100))
        b=r.owner_evidence(r.parse_raw(btext+'<|im_end|>',ident),.5)
        c=r.owner_evidence(r.parse_raw(ctext+'<|im_end|>',ident),.5)
        b['prefix_action']=r.owner_evidence(r.parse_raw(btext,ident),.5)
        c['prefix_action']=r.owner_evidence(r.parse_raw(ctext,ident),.5)
        change=r.compare(b,c)
        self.assertEqual(change['lost_gt_ids'],[1])
        self.assertEqual(change['gained_gt_ids'],[2])
        self.assertEqual(change['attribution']['lost_gt_ids']['pure_tail'],[])
        self.assertEqual(change['attribution']['gained_gt_ids']['pure_tail'],[])
        self.assertEqual(change['attribution']['lost_gt_ids']['direct_or_matching_ambiguous'],[1])

    def test_global_reassignment_incident_edges_prevent_false_tail(self):
        ident=identity([(0,0,100,100),(40,0,140,100)])
        prefix=textrow('person',(20,0,120,100))
        suffix=textrow('person',(0,0,100,100))
        direct=r.owner_evidence(r.parse_raw(prefix,ident),.5)
        full=r.owner_evidence(r.parse_raw(prefix+suffix+'<|im_end|>',ident),.5)
        self.assertEqual(len(direct['matched_gt_ids']),1)
        self.assertEqual(direct['incident_gt_ids'],[1,2])
        self.assertEqual(full['matched_gt_ids'],[1,2])
        self.assertEqual({x['gt_id']:x['prediction_index'] for x in full['matches']},{1:1,2:0})
        # Subtracting selected direct matches would falsely label one owner tail.
        self.assertEqual(len(set(full['matched_gt_ids'])-set(direct['matched_gt_ids'])),1)
        self.assertEqual(set(full['matched_gt_ids'])-set(direct['incident_gt_ids']),set())

    def test_category_independent_repeat_strict_threshold(self):
        ident=identity([])
        row=r.parse_raw(textrow('person',(0,0,100,100))+textrow('car',(0,0,100,100)),ident)
        repeats=r.strict_later_repeats(row)
        self.assertEqual(len(repeats),1)
        self.assertEqual(repeats[0]['later_prediction_index'],1)
        # Exact 0.95 must not count; use already-pixel native prediction records.
        row['pred'][0]['bbox']=[0,0,100,100]
        row['pred'][1]['bbox']=[0,0,95,100]
        self.assertEqual(r.strict_later_repeats(row),[])

    def test_eos_and_cap_rejection(self):
        eos={'kind':'eos','token_ids':[r.EOS]}
        self.assertEqual(r.validate_stop([r.EOS],[],eos,'im_end'),'forced_eos')
        with self.assertRaisesRegex(ValueError,'reopened'):
            r.validate_stop([r.EOS,42],[42],eos,'im_end')
        row={'kind':'row','token_ids':[151646,151649]}
        with self.assertRaisesRegex(ValueError,'incorrectly capped'):
            r.validate_stop([42],[],row,'length')
        with self.assertRaisesRegex(ValueError,'cap mismatch'):
            r.validate_stop([42]*(r.CAP+1),[],row,'length')


if __name__=='__main__':
    unittest.main()


def test_preserved_original_code_locator_is_explicit_and_hash_checked(tmp_path, monkeypatch):
    import pytest
    original = r.ORIGINAL_CODE_PREFIX / 'src/inference/parsing.py'
    preserved = tmp_path / 'preserved'
    path = preserved / 'src/inference/parsing.py'
    path.parent.mkdir(parents=True)
    path.write_text('original bytes')
    manifest = {'sources': {'code': [{'path': str(original), 'sha256': r.sha256(path)}]}}
    real_sha = r.sha256
    def without_provider(p):
        if str(p).startswith(str(r.ORIGINAL_CODE_PREFIX)):
            raise FileNotFoundError('original provider unavailable')
        return real_sha(p)
    monkeypatch.setattr(r, 'sha256', without_provider)
    with pytest.raises(FileNotFoundError, match='provider unavailable'):
        r.original_code_bindings(manifest)
    binding = r.original_code_bindings(manifest, preserved)[0]
    assert binding['original_path'] == str(original)
    assert binding['resolved_path'] == str(path.resolve())
    path.write_text('changed bytes')
    with pytest.raises(ValueError, match='original consumer/provider code changed'):
        r.original_code_bindings(manifest, preserved)
