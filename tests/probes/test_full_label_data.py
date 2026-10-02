import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from probes.full_label_fit import experiment as f
from probes.full_label_fit import recipe as fit_recipe
from probes import iterative_positive as p
from probes import rollout_row_credit as r

HISTORICAL_RECIPE_SHA256 = 'a2b982a8526ca53efa2ed68ac04bd1fda8df81937122b3bbdb804313b56d8141'


class FullLabelDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images, cls.manifest = f.verify_inputs()
        cls.retained = p.load(r.ROOT / 'cpu-04/retained-10.json')
        cls.capture = p.load(f.INPUTS)

    def _run_qualification_cli(self, root, *profile_args):
        checkpoint = Path('/data/CoordExp/outputs/shared/checkpoints/start-loss-instance-margin-order17-step256/payload')
        argv = ['full_label_fit.experiment', 'qualify', '--unit', str(f.UNIT), '--root', str(root),
                '--qualification-run', str(root / 'run-2'), '--observation-run', str(root / 'run-16'),
                '--checkpoint', str(checkpoint), *profile_args]
        with patch('sys.argv', argv), patch(
                'src.artifacts.git_identity.capture_source_identity', return_value={'schema': 'test-source-identity'}):
            f.main()
        return checkpoint, json.loads((root / 'qualification.json').read_text())

    def test_manifest_and_role_stripping(self):
        self.assertEqual((18, 570), (len(self.images), sum(len(x['objects']) for x in self.images)))
        self.assertEqual((18, 513), (len(self.retained), sum(len(x['objects']) for x in self.retained)))
        self.assertEqual(57, self.manifest['retained']['excluded_objects'])
        self.assertTrue(all('hidden_objects' not in x for x in self.images))
        self.assertEqual([x['image_id'] for x in self.images], [x['image_id'] for x in self.manifest['images']])

    def test_provenance_mutations_fail_closed(self):
        changed = copy.deepcopy(self.images)
        changed[0]['objects'][0]['coco_ann_id'] += 1
        with self.assertRaises(AssertionError):
            f.validate_inputs(changed, self.retained, self.capture)
        changed = copy.deepcopy(self.images)
        changed[0]['image_sha256'] = '0' * 64
        with self.assertRaises(AssertionError):
            f.validate_inputs(changed, self.retained, self.capture)
        changed = copy.deepcopy(self.images)
        changed[0]['objects'][0]['bbox_2d'][0] = 1.5
        with self.assertRaises(AssertionError):
            f.validate_inputs(changed, self.retained, self.capture)
        changed_retained = copy.deepcopy(self.retained)
        changed_retained[0]['objects'][0]['bbox_2d'][0] += 1
        with self.assertRaises(AssertionError):
            f.validate_inputs(self.images, changed_retained, self.capture)

    def test_hidden_role_is_stripped_even_if_duplicated_or_malformed(self):
        raw = p.load(f.TRUTH)
        hidden = raw[0]['hidden_objects'][0]
        raw[0]['hidden_objects'] = [hidden, copy.deepcopy(hidden), {'coco_ann_id': 'not-an-int'}]
        f.validate_inputs(raw, self.retained, self.capture)
        snapshot = [{key: image[key] for key in f.FULL_FIELDS} for image in raw]
        self.assertEqual(snapshot, self.images)
        self.assertNotIn('hidden_objects', snapshot[0])

    def test_qualification_roundtrip_matches_runtime_consumer(self):
        from probes import online_row_credit as o

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint, qual = self._run_qualification_cli(root)
            manifest = f.UNIT / 'inputs/manifest.json'
            training = f.UNIT / 'inputs/full-labels.json'
            historical = fit_recipe.full_label_recipe(checkpoint, f.sha(checkpoint / 'inference_payload_manifest.json'),
                                             training, f.sha(training))
            self.assertEqual(HISTORICAL_RECIPE_SHA256, o.identity(historical))
            expected = fit_recipe.full_label_recipe(checkpoint, f.sha(checkpoint / 'inference_payload_manifest.json'),
                training, f.sha(training), rollout_policy='previous_rollout_tokens_lpt_v1')
            self.assertEqual(expected, qual['correction'])
            self.assertEqual('previous_rollout_tokens_lpt_v1', qual['correction']['rollout_policy'])
            self.assertNotEqual(HISTORICAL_RECIPE_SHA256, o.identity(qual['correction']))
            self.assertNotIn('lr_profile', qual['correction']['optimizer'])
            self.assertEqual({'path': str(manifest), 'sha256': f.sha(manifest)}, qual['input_manifest'])
            self.assertEqual({str(x) for x in (o.INPUTS, training, o.p.POLICY, manifest)}, set(qual['sha256']))
            self.assertEqual({'profiles': [{'arm': 'treatment', 'microbatch': 1, 'activation_checkpointing': False}]},
                             qual['execution'])
            decoder = f.decoder_runtime_identity()
            self.assertEqual(decoder, qual['decoder_runtime_identity'])
            self.assertEqual('0.29.0+cu129', decoder['version'])
            self.assertTrue(any(path.endswith('/vllm/v1/sample/sampler.py') for path in decoder['source_sha256']))
            self.assertTrue(any(path.endswith('/vllm/sampling_params.py') for path in decoder['source_sha256']))
            self.assertEqual({'2': {'treatment': str(root / 'run-2')},
                              '16': {'treatment': str(root / 'run-16')}}, qual['pairs'])
            with patch('src.artifacts.git_identity.verify_source_identity'):
                binding = o.correction_binding(root, 'treatment', 1, checkpoint, .1,
                                               o.identity(qual['correction']), full_label_region=True)
            self.assertEqual(str(checkpoint), binding['checkpoint'])

    def test_cli_lr_profiles_bind_all_three_requested_shapes(self):
        from probes import online_row_credit as o

        profiles = ((1.0, 0), (1.0, 4), (0.90625, 0))
        for scale, warmup_updates in profiles:
            with self.subTest(lr_scale=scale, warmup_updates=warmup_updates), \
                    tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                checkpoint, qual = self._run_qualification_cli(
                    root, '--lr-scale', str(scale), '--warmup-updates', str(warmup_updates))
                training = f.UNIT / 'inputs/full-labels.json'
                expected_profile = {'lr_scale': scale, 'warmup_updates': warmup_updates}
                expected = fit_recipe.full_label_recipe(checkpoint,
                    f.sha(checkpoint / 'inference_payload_manifest.json'), training, f.sha(training),
                    lr_profile=expected_profile, rollout_policy='previous_rollout_tokens_lpt_v1')
                self.assertEqual(expected, qual['correction'])
                self.assertEqual(expected_profile, qual['correction']['optimizer']['lr_profile'])
                with patch('src.artifacts.git_identity.verify_source_identity'):
                    binding = o.correction_binding(root, 'treatment', 1, checkpoint, .1,
                                                   o.identity(qual['correction']), full_label_region=True)
                self.assertEqual(expected_profile, binding['lr_profile'])
                self.assertEqual(o.identity(expected), binding['recipe_sha256'])

    def test_cli_rejects_missing_or_malformed_lr_profile_before_source_capture(self):
        checkpoint = Path('/data/CoordExp/outputs/shared/checkpoints/start-loss-instance-margin-order17-step256/payload')
        base = ['full_label_fit.experiment', 'qualify', '--unit', str(f.UNIT)]
        with patch('src.artifacts.git_identity.capture_source_identity') as capture_source:
            for partial_pair in (('--lr-scale', '1'), ('--warmup-updates', '4')):
                with self.subTest(partial_pair=partial_pair), patch('sys.argv', base + [
                        '--root', str(Path('/cpu/lr-profile-test') / 'missing-profile'), '--qualification-run', str(Path('/cpu/lr-profile-test') / 'run-2'),
                        '--observation-run', str(Path('/cpu/lr-profile-test') / 'run-16'), '--checkpoint', str(checkpoint), *partial_pair]):
                    with self.assertRaises(SystemExit):
                        f.main()
                capture_source.assert_not_called()

            with patch('sys.argv', base + ['--root', str(Path('/cpu/lr-profile-test') / 'malformed-profile'),
                    '--qualification-run', str(Path('/cpu/lr-profile-test') / 'run-2'), '--observation-run', str(Path('/cpu/lr-profile-test') / 'run-16'),
                    '--checkpoint', str(checkpoint), '--lr-scale', 'nan', '--warmup-updates', '0']):
                with self.assertRaisesRegex(AssertionError, 'invalid LR scale'):
                    f.main()
            capture_source.assert_not_called()

    def test_annotation_metrics_and_recovery_tie(self):
        target_image = next(image for image in self.images if image['image_id'] == 4134)
        target = next(obj for obj in target_image['objects'] if obj['coco_ann_id'] == 294005)
        from src.eval.saved_rows import iou_xyxy
        other = next(obj for obj in target_image['objects'] if obj['coco_ann_id'] != target['coco_ann_id']
                     and iou_xyxy(obj['bbox_2d'], target['bbox_2d']) < 0.1)

        def row(image, objects=()):
            text = '<|im_end|>'
            if objects:
                text = ''.join(f"<|object_ref_start|>{obj['desc']}<|object_ref_end|><|box_start|>" +
                               ''.join(f'<|coord_{coord}|>' for coord in obj['bbox_2d']) + '<|box_end|>'
                               for obj in objects) + '<|im_end|>'
            return {'image_id': image['image_id'], 'request_id': f"test:{image['image_id']}",
                    'arm': 'treatment', 'width': image['width'], 'height': image['height'],
                    'crop': [0, 0, image['width'], image['height']], 'text': text,
                    'generated_tokens': 0, 'stop_reason': 'im_end'}

        zero = [row(image, [target] if image['image_id'] == 4134 else []) for image in self.images]
        two = [row(image, [other, other] if image['image_id'] == 4134 else []) for image in self.images]
        four = [row(image, [other] if image['image_id'] == 4134 else []) for image in self.images]
        eight = [row(image, [target] if image['image_id'] == 4134 else []) for image in self.images]
        sixteen = [row(image, [other] if image['image_id'] == 4134 else []) for image in self.images]
        scored = f.evaluate_versions(self.images, {'zero': zero, '2': two, '4': four, '8': eight, '16': sixteen})
        metrics = scored['scored']['2']['totals']
        self.assertEqual(570, metrics['denominator'])
        self.assertEqual(2, metrics['Nvalid'])
        self.assertEqual(1, metrics['tp'])
        self.assertAlmostEqual(2 / 572, metrics['f1'])
        self.assertEqual(569, metrics['fn'])
        self.assertEqual(1, metrics['fp_annotation'])
        self.assertEqual(1, metrics['literal_valid_repeats'])
        self.assertEqual(570, len(scored['denominator_annotation_ids']))
        self.assertEqual(569, sum(len(row['missed_annotation_ids']) for row in scored['scored']['2']['images'].values()))
        self.assertEqual(0, scored['transitions']['2']['retained'])
        self.assertEqual(1, scored['transitions']['2']['gained_count'])
        self.assertEqual(1, scored['transitions']['2']['lost_baseline_count'])
        self.assertEqual(0, scored['transitions']['4']['recovered_again_count'])
        self.assertEqual(1, scored['transitions']['8']['recovered_again_count'])
        self.assertEqual(1, scored['transitions']['16']['recovered_again_count'])
        self.assertEqual('2', scored['transitions']['4']['adjacent']['from_version'])
        self.assertEqual(1, scored['transitions']['4']['adjacent']['preserved_count'])
        self.assertEqual(1, scored['transitions']['8']['adjacent']['lost_count'])
        self.assertTrue(scored['transitions']['image_4134_annotation_294005']['zero']['version_correct'])
        self.assertFalse(scored['limitations'].endswith('physical false positives.'))


if __name__ == '__main__':
    unittest.main()
