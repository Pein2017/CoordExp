"""CPU falsifiers for the opt-in owner-region correction route."""
import copy
import importlib
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from probes import online_row_credit as o
from probes.owner_region_ranking import owner_slots, region_margin


class FullLabelRegionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = importlib.import_module('test_online_row_credit').OnlineCreditTest
        cls.helper.setUpClass()
        cls.q = cls.helper.q
        cls.t = cls.helper.t
        cls.vocab = cls.helper.vocab

    @classmethod
    def fixture(cls, boxes, objects=None):
        return cls.helper.fixture(cls.helper, boxes, objects)

    @classmethod
    def region_producer(cls, **extra):
        return dict(cls.helper.producer, owner_region=dict(o.OWNER_REGION),
                    training_sha256='training-fixture-sha256', **extra)

    def region_record(self, boxes, objects=None, **producer_fields):
        image, record, _ = self.fixture(boxes, objects)
        producer = self.region_producer(**producer_fields)
        return image, o.seal(record, producer)

    def persisted_full_label_fixture(self, root):
        helper = self.helper()

        def rewrite(path, value):
            Path(path).write_text(o.p.canonical(value) + '\n')

        def fixture_components(row_loss, reduction_weight):
            return dict(reduction_weight=float(reduction_weight),row_loss=float(row_loss),
                        lexical_schema_ce=float(row_loss),coordinate_region_hinge=0.,
                        type_gate_weighted=0.,conditional_order_weighted=0.)

        make_restored_qualifier = helper.restored_m_qualifier

        def two_update_qualifier(qual_root, checkpoint):
            qual = make_restored_qualifier(qual_root, checkpoint)
            qual['correction']['updates'] = [2, 16]
            qual['pairs'] = {'2': {'treatment': str(qual_root / 'full-label-run')}}
            return qual

        with patch.object(helper, 'restored_m_qualifier', side_effect=two_update_qualifier):
            output, old_qual, images, inputs = helper.correction_fixture_tree(
                root, 'treatment', updates=2, completion=True, identity_events=True,
                completion_weighting=o.RESTORED_M_WEIGHTING)
        checkpoint = root / 'anchor'
        training = root / 'training.json'
        manifest = root / 'input-manifest.json'
        training_images = [images[i] for i in sorted(images)]
        o.p.write(training, training_images)
        images = {image['image_id']: image for image in o.p.load(training)}
        o.p.write(manifest, dict(training=str(training)))
        manifest_sha = o.p.digest(checkpoint / 'inference_payload_manifest.json')
        training_sha = o.p.digest(training)
        recipe = o.full_label_recipe(checkpoint, manifest_sha, training, training_sha)
        from probes import full_label_self_rollout
        qual = dict(correction=recipe,
                    decoder_runtime_identity=full_label_self_rollout.decoder_runtime_identity(),
                    input_manifest=dict(path=str(manifest), sha256=o.p.digest(manifest)),
                    sha256={str(path): o.p.digest(path) for path in
                            (o.INPUTS, training, o.p.POLICY, manifest)},
                    source=old_qual['source'], rollout_backend='vllm', schema_geometry=True,
                    execution=dict(profiles=[dict(arm='treatment', microbatch=1,
                                                  activation_checkpointing=False)]),
                    pairs={'2': {'treatment': str(output)}})
        rewrite(root / 'qualification.json', qual)
        with patch('probes.full_label_self_rollout.verify_inputs'), \
             patch('src.artifacts.git_identity.verify_source_identity'), \
             patch.object(o, 'verify_anchor_payload'):
            binding = o.correction_binding(root, 'treatment', 1, checkpoint, .1,
                                           o.identity(recipe), full_label_region=True)

        for version in range(3):
            initial = o.p.load(output / 'rank-0' / f'producer-{version}.json')
            producer = dict(initial['producer'], recipe_sha256=o.identity(recipe),
                            owner_region=dict(o.OWNER_REGION), training_sha256=training_sha)
            for rank in range(8):
                directory = output / f'rank-{rank}'
                bound = o.p.load(directory / f'producer-{version}.json')
                bound['producer'] = producer
                rewrite(directory / f'producer-{version}.json', bound)
                rewrite(directory / 'correction.json', binding)
                rewrite(directory / 'geometry.json', binding)

                rawdir = output / f'rollout-{version}' / f'rank-{rank}'
                records = {}
                for path in sorted(rawdir.glob('*.json')):
                    if path.name == 'complete.json':
                        continue
                    record = o.seal(o.p.load(path), producer)
                    records[record['image_id']] = record
                    rewrite(path, record)
                raw_complete_path = rawdir / 'complete.json'
                raw_complete = o.p.load(raw_complete_path)
                raw_complete.update(producer=producer, artifacts={
                    path.name: o.p.digest(path) for path in sorted(rawdir.glob('*.json'))
                    if path.name != 'complete.json'})
                rewrite(raw_complete_path, raw_complete)

                local_ids = sorted(x['image_id'] for x in inputs)[rank::8]
                plans = {i: o.correction_plan(images[i], records[i], self.t, binding)
                         for i in local_ids}
                rewrite(directory / f'credit-{version}.json', [plans[i] for i in local_ids])
                if version == 2:
                    continue
                update_path = directory / f'update-{version + 1}.json'
                update = o.p.load(update_path)
                update['producer'] = producer
                for row in update['forwards']:
                    image_id = row['image_id']
                    image, record, plan = images[image_id], records[image_id], plans[image_id]
                    row.update(producer=producer, raw_identity=record['raw_identity'],
                               correction=dict(arm='treatment', duplicate_weight=1,
                                               plan_sha256=o.identity(plan)),
                               completion_arm='treatment',
                               completion_weighting=o.RESTORED_M_WEIGHTING,
                               owner_region=dict(o.OWNER_REGION), training_sha256=training_sha)
                    row['cached_replay'] = []
                    if row['branch'] == 'trace':
                        selected = o.bridge_trace_plan(plan, 'chain')
                        sequences = [o.r.positive_sequence(image, record, target, self.t)
                                     for target in selected['M']]
                        row['row_losses'] = [dict(atoms=[atom.to_artifact_dict() for atom in sequence.atoms],
                                                  owner_region=o.region_row_evidence(
                                                      sequence, target, image, self.vocab,
                                                      o.OWNER_REGION))
                                             for target, sequence in zip(selected['M'], sequences)]
                        reduction_weight=1/len(sequences) if sequences else 0
                        for value in row['row_losses']:
                            value['loss_components']=fixture_components(row['terms']['M'],reduction_weight)
                        row['schema_geometry'] = plan['schema_geometry']
                        prompt_length = len(record['prompt_token_ids'])
                        eligible = sorted({j for observed in selected['observations']
                                           for j in observed['coordinate_positions']})
                        checked = [j for j in eligible if prompt_length + j - 1 in row['positions']]
                        row['replay_ordering'] = dict(
                            scope='first_four_selected_original_trace_coordinate_positions',
                            eligible=len(eligible), eligible_in_selected=len(checked),
                            checked=min(4, len(checked)),
                            interpretation='descriptive_HF_replay_vs_native_emitted_token_not_deployment_argmax_certificate')
                        row['cached_replay'] = [dict(
                            index=j, causal_position=prompt_length + j - 1,
                            emitted_token=record['token_ids'][j],
                            prefix_sha256=o.identity(record['prompt_token_ids'] + record['token_ids'][:j]),
                            hf_argmax_token=record['token_ids'][j], max_minus_emitted=0.0)
                            for j in checked[:4]]
                    elif row['branch'] == 'bridge':
                        sequences = o.bridge_sequences(image, record, plan, self.t)
                        row['row_losses'] = o.bridge_row_evidence(
                            record, plan, sequences, None, image, self.vocab)
                        for value in row['row_losses']:
                            value['loss_components']=fixture_components(1.,value['weight'])
                        row['terms']['B']=sum(value['weight'] for value in row['row_losses'] if value['kind']=='B')
                        row['terms']['M_relocated']=sum(value['weight'] for value in row['row_losses'] if value['kind']=='M')
                        for name,value in row['terms'].items():row['logit_derivatives'][name]['loss']=value
                    else:
                        target = plan['redirects'][row['redirect']['event_index']]
                        sequence = o.redirect_sequence(image, record, target, self.t)
                        row['row_losses'] = [dict(
                            atoms=[atom.to_artifact_dict() for atom in sequence.atoms],
                            owner_region=o.region_row_evidence(
                                sequence, target, image, self.vocab, o.OWNER_REGION))]
                        row['row_losses'][0]['loss_components']=fixture_components(1.,target.get('event_weight',1))
                        row['redirect'].update(target=target,
                                               site_kind=sequence.atoms[target['site']['offset']].token_type)
                        if row['redirect']['site_kind'] == 'coordinate':
                            row['terms']['redirect_margin'] = 0.0
                            row['redirect'].update(good_derivative=0.0, bad_derivative=0.0)
                            row['logit_derivatives']['redirect_margin'].update(
                                loss=0.0, l2=0.0, linf=0.0, support_rows=0)
                    row['loss'] = sum(row['terms'].values())
                    for name, value in row['terms'].items():
                        row['logit_derivatives'][name]['loss'] = value
                rewrite(update_path, update)

        for rank in range(8):
            directory = output / f'rank-{rank}'
            complete_path = directory / 'complete.json'
            complete = o.p.load(complete_path)
            complete['artifacts'] = {path.name: o.p.digest(path)
                                     for path in sorted(directory.glob('*.json'))
                                     if path.name != 'complete.json'}
            rewrite(complete_path, complete)
        return output, root, training, images, inputs, qual

    def test_recipe_binding_source_closure_and_cli_delivery(self):
        checkpoint, training = Path('/cpu/anchor'), Path('/cpu/training.json')
        recipe = o.full_label_recipe(checkpoint, 'manifest-sha', training, 'training-sha')
        self.assertEqual(recipe['mode'], 'full-label-region-v1')
        self.assertEqual(recipe['owner_region'], o.OWNER_REGION)
        self.assertEqual((recipe['arms'], recipe['updates']), ({'treatment': 1}, [2, 16]))
        self.assertEqual(o.source_paths(True), [
            'probes/full_label_self_rollout.py', 'probes/owner_region_ranking.py',
            *o.source_paths(),
        ])

        root, manifest = Path('/cpu'), Path('/cpu/manifest.json')
        anchor_manifest = Path('/cpu/anchor/inference_payload_manifest.json')
        hashes = {str(o.INPUTS): 'inputs', str(training): 'training-sha',
                  str(o.p.POLICY): 'policy', str(manifest): 'input-manifest'}
        qual = dict(correction=recipe, input_manifest=dict(path=str(manifest), sha256='input-manifest'),
                    sha256=hashes, source={'commit': 'fixture', 'files': []},
                    decoder_runtime_identity=__import__(
                        'probes.full_label_self_rollout', fromlist=['decoder_runtime_identity']
                    ).decoder_runtime_identity(),
                    rollout_backend='vllm', schema_geometry=True)
        original_load = o.p.load

        def load(path):
            if path == root / 'qualification.json':
                return qual
            return original_load(path)

        with patch.object(o.p, 'load', side_effect=load), \
             patch.object(o.p, 'digest', side_effect=lambda path: hashes.get(str(path), 'manifest-sha')), \
             patch('probes.full_label_self_rollout.verify_inputs') as verify_inputs, \
             patch('src.artifacts.git_identity.verify_source_identity') as verify_source, \
             patch.object(o, 'verify_anchor_payload'):
            bound = o.correction_binding(root, 'treatment', 1, checkpoint, .1,
                                         o.identity(recipe), full_label_region=True)
            self.assertEqual(bound['owner_region'], o.OWNER_REGION)
            self.assertEqual(bound['training_sha256'], 'training-sha')
            verify_inputs.assert_called_once_with(training.parent.parent)
            verify_source.assert_called_once_with(qual['source'], required_paths=o.source_paths(True))
            for mixed, enabled in ((dict(qual, correction=None), True),
                                   (qual, False), (dict(qual, bridge={}), True)):
                with patch.object(o.p, 'load', return_value=mixed), self.assertRaises(AssertionError):
                    o.correction_binding(root, 'treatment', 1, checkpoint, .1,
                                         o.identity(recipe), full_label_region=enabled)

        argv = ['--root', str(root), '--output', '/cpu/out', '--updates', '2',
                '--geometry-weight', '.1', '--start-checkpoint', str(checkpoint),
                '--recipe-sha256', o.identity(recipe), '--rollout-backend', 'vllm',
                '--schema-geometry', '--activation-checkpointing', 'off',
                '--full-label-region', '--correction-arm', 'treatment', '--duplicate-weight', '1']
        for command in ('run', 'readback', 'offline'):
            with patch('sys.argv', ['online_row_credit', command, *argv]), patch.object(o, command) as caller:
                o.main()
                self.assertTrue(caller.call_args.kwargs['full_label_region'])
                self.assertEqual(caller.call_args.kwargs['correction_arm'], 'treatment')
                self.assertEqual(caller.call_args.args[2], 2)

    def test_owner_identity_translation_alternative_and_legacy_mutation(self):
        actual = [300, 10, 400, 100]
        decoy = [600, 300, 700, 500]
        objects = [dict(coco_ann_id=4, desc='person', bbox_2d=decoy),
                   dict(coco_ann_id=17, desc='person', bbox_2d=actual)]
        image, record = self.region_record([actual], objects)
        row = o.credit(image, record, self.t, record['producer'], redirect_enabled=False)['M'][0]
        self.assertEqual(row['annotation_id'], 17)
        sequence = o.r.positive_sequence(image, record, row, self.t)
        positions = tuple(atom.causal_logits_position for atom in sequence.atoms)
        logits = torch.full((1, len(positions), self.vocab.vocab_size), -10.)
        for j, atom in enumerate(sequence.atoms):
            if atom.token_type == 'coordinate':
                bins = {token: value for value, token in enumerate(self.vocab.coordinate)}
                emitted = bins[atom.token_id]
                logits[0, j, atom.token_id] = 0
                if atom.coordinate_target.slot_index == 0:
                    evidence = o.region_row_evidence(sequence, row, image, self.vocab, o.OWNER_REGION)
                    allowed = owner_slots(evidence['gt'], evidence['actual'])[1][0]
                    alternative = next(value for value in allowed if value != emitted)
                    logits[0, j, atom.token_id] = -7
                    logits[0, j, self.vocab.coordinate[alternative]] = 7
            else:
                logits[0, j, atom.token_id] = 0
        logits.requires_grad_()
        evidence = o.region_row_evidence(sequence, row, image, self.vocab, o.OWNER_REGION)
        self.assertEqual(evidence['gt'], actual)
        self.assertNotEqual(evidence['gt'], image['objects'][0]['bbox_2d'])
        slot0 = next(site for site in evidence['sites'] if site['slot'] == 0)
        self.assertNotEqual(slot0['position'], None)

        from src.losses.base_ce import BaseTokenCE
        from src.losses.conditional_order_gate import ConditionalOrderGateLoss
        from src.losses.context import LossContext
        from src.losses.token_type_gate import TokenTypeGateLoss

        context = LossContext(logits, sequence, self.vocab, positions)
        point_ce = BaseTokenCE().per_atom_loss(context)
        gate = TokenTypeGateLoss().per_atom_loss(context).mean()
        order = ConditionalOrderGateLoss().per_segment_loss(context).segment_losses.mean()
        actual_loss = o.owner_row_loss(logits, sequence, self.vocab, positions, row, image, o.OWNER_REGION)
        slots, allowed, _ = owner_slots(evidence['gt'], evidence['actual'], o.OWNER_REGION['tau'])
        allowed_by_slot = dict(zip(slots, allowed))
        lookup = {position: j for j, position in enumerate(positions)}
        expected_atoms = []
        for j, atom in enumerate(sequence.atoms):
            if atom.token_type != 'coordinate':
                expected_atoms.append(point_ce[j])
            elif atom.coordinate_target.slot_index in allowed_by_slot:
                accepted = [self.vocab.coordinate[b]
                            for b in allowed_by_slot[atom.coordinate_target.slot_index]]
                z = logits[0, lookup[atom.causal_logits_position]]
                expected_atoms.append(region_margin(z, accepted, o.OWNER_REGION['margin']))
            else:
                expected_atoms.append(logits[0, lookup[atom.causal_logits_position]].sum() * 0)
        expected_loss = torch.stack(expected_atoms).mean() + .1 * gate + .01 * order
        torch.testing.assert_close(actual_loss, expected_loss)
        self.assertGreater(float(point_ce[next(i for i, atom in enumerate(sequence.atoms)
                                               if atom.token_type == 'coordinate')].detach()), 10)
        self.assertTrue(torch.isfinite(gate) and torch.isfinite(order))
        with patch.object(o.p, 'image_loss', side_effect=AssertionError('legacy point CE reached')):
            torch.testing.assert_close(actual_loss, o.owner_row_loss(
                logits, sequence, self.vocab, positions, row, image, o.OWNER_REGION))

        translated = copy.deepcopy(image)
        translated['objects'][1]['bbox_2d'] = [750, 300, 850, 500]
        translated_evidence = o.region_row_evidence(sequence, row, translated, self.vocab, o.OWNER_REGION)
        self.assertEqual(translated_evidence['gt'], translated['objects'][1]['bbox_2d'])
        translated_loss = o.owner_row_loss(logits, sequence, self.vocab, positions, row,
                                           translated, o.OWNER_REGION)
        self.assertGreater(float(translated_loss.detach()), float(actual_loss.detach()))

        legacy = o.p.image_loss(logits, sequence, self.vocab, positions)[0]
        self.assertNotAlmostEqual(float(actual_loss.detach()), float(legacy.detach()), places=3)
        self.assertGreater(float(legacy.detach()), float(actual_loss.detach()))

    def test_first_owner_failure_trains_prefix_through_failure_only(self):
        gt, actual = [100, 100, 400, 400], [100, 100, 900, 200]
        image, record = self.region_record([actual], [dict(coco_ann_id=9, desc='person', bbox_2d=gt)])
        row = dict(o.observations(record, self.t)[0][0], annotation_id=9)
        sequence = o.r.positive_sequence(image, record, row, self.t)
        evidence = o.region_row_evidence(sequence, row, image, self.vocab, o.OWNER_REGION)
        self.assertEqual((evidence['first_failure'], evidence['eligible']), (2, 3))
        self.assertEqual([site['slot'] for site in evidence['sites']], [0, 1, 2])

        positions = tuple(atom.causal_logits_position for atom in sequence.atoms)
        logits = torch.zeros((1, len(positions), self.vocab.vocab_size))
        for site in evidence['sites']:
            z = logits[0, positions.index(site['position'])]
            z[0] = 4
            ids = [self.vocab.coordinate[value] for value in
                   owner_slots(evidence['gt'], evidence['actual'])[1][site['slot']]]
            z[ids] = 0
        logits.requires_grad_()
        from src.losses.base_ce import BaseTokenCE
        from src.losses.conditional_order_gate import ConditionalOrderGateLoss
        from src.losses.context import LossContext
        from src.losses.token_type_gate import TokenTypeGateLoss
        positions_lookup = {position: j for j, position in enumerate(positions)}
        context = LossContext(logits, sequence, self.vocab, positions)
        point_ce = BaseTokenCE().per_atom_loss(context)
        gate = TokenTypeGateLoss().per_atom_loss(context).mean()
        order = ConditionalOrderGateLoss().per_segment_loss(context).segment_losses.mean()
        valid_slots, valid_bins, _ = owner_slots(evidence['gt'], evidence['actual'], o.OWNER_REGION['tau'])
        allowed_by_slot = dict(zip(valid_slots, valid_bins))
        expected_atoms = []
        for j, atom in enumerate(sequence.atoms):
            if atom.token_type != 'coordinate':
                expected_atoms.append(point_ce[j])
            elif atom.coordinate_target.slot_index in allowed_by_slot:
                accepted = [self.vocab.coordinate[b]
                            for b in allowed_by_slot[atom.coordinate_target.slot_index]]
                z = logits[0, positions_lookup[atom.causal_logits_position]]
                expected_atoms.append(region_margin(z, accepted, o.OWNER_REGION['margin']))
            else:
                expected_atoms.append(logits[0, positions_lookup[atom.causal_logits_position]].sum() * 0)
        actual_loss = o.owner_row_loss(logits, sequence, self.vocab, positions, row,
                                       image, o.OWNER_REGION)
        expected_loss = torch.stack(expected_atoms).mean() + .1 * gate + .01 * order
        torch.testing.assert_close(actual_loss, expected_loss)
        owner_terms = [region_margin(logits[0, positions.index(site['position'])],
                                     [self.vocab.coordinate[value] for value in
                                      owner_slots(evidence['gt'], evidence['actual'])[1][site['slot']]],
                                     o.OWNER_REGION['margin']) for site in evidence['sites']]
        gradients, = torch.autograd.grad(torch.stack(owner_terms).sum(), logits)
        coord_atoms = {atom.coordinate_target.slot_index: atom for atom in sequence.atoms
                       if atom.token_type == 'coordinate'}
        for slot in range(3):
            self.assertGreater(float(gradients[0, positions.index(coord_atoms[slot].causal_logits_position)].abs().sum()), 0)
        self.assertEqual(float(gradients[0, positions.index(coord_atoms[3].causal_logits_position)].abs().sum()), 0)

    def test_persisted_two_update_full_label_readback_rejects_missing_region_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, root, training, images, inputs, qual = self.persisted_full_label_fixture(root)
            original_load = o.p.load

            def load(path):
                path = Path(path)
                if path == training:
                    return [images[i] for i in sorted(images)]
                if path == o.INPUTS:
                    return inputs
                if 'truth' in str(path) or 'evaluator' in str(path):
                    raise AssertionError('runtime read entered evaluator-owned data')
                return original_load(path)

            kwargs = dict(geometry_weight=.1, start_checkpoint=root / 'anchor',
                          recipe_sha256=o.identity(qual['correction']),
                          correction_arm='treatment', duplicate_weight=1,
                          rollout_backend='vllm', schema_geometry=True,
                          activation_checkpointing=False, full_label_region=True)
            with patch.object(o.p, 'load', side_effect=load), \
                 patch.object(o.r, 'frontend', return_value=self.q), \
                 patch('probes.full_label_self_rollout.verify_inputs'), \
                 patch('src.artifacts.git_identity.verify_source_identity'), \
                 patch.object(o, 'verify_start_export'):
                o.readback(output, root, 2, **kwargs)
                result = original_load(output / 'readback.json')
                self.assertEqual([row['update'] for row in result], [0, 1, 2])
                self.assertTrue(all(row['correction']['owner_region'] == o.OWNER_REGION
                                    and row['producer']['training_sha256'] ==
                                    qual['correction']['training_sha256']
                                    for row in result))

                update_path = output / 'rank-0' / 'update-1.json'
                complete_path = output / 'rank-0' / 'complete.json'
                original_update = update_path.read_bytes()
                original_complete = complete_path.read_bytes()
                evidence = original_load(update_path)
                bridge = next(row for row in evidence['forwards'] if row['branch'] == 'bridge')
                bridge['row_losses'][0].pop('owner_region')
                update_path.write_text(o.p.canonical(evidence) + '\n')
                complete = original_load(complete_path)
                complete['artifacts']['update-1.json'] = o.p.digest(update_path)
                complete_path.write_text(o.p.canonical(complete) + '\n')
                (output / 'readback.json').unlink()
                with self.assertRaises(AssertionError):
                    o.readback(output, root, 2, **kwargs)
                self.assertFalse((output / 'readback.json').exists())

                update_path.write_bytes(original_update)
                complete_path.write_bytes(original_complete)
                evidence = original_load(update_path)
                bridge = next(row for row in evidence['forwards'] if row['branch'] == 'bridge')
                bridge['row_losses'][0]['loss_components']['type_gate_weighted'] = -0.1
                update_path.write_text(o.p.canonical(evidence) + '\n')
                complete = original_load(complete_path)
                complete['artifacts']['update-1.json'] = o.p.digest(update_path)
                complete_path.write_text(o.p.canonical(complete) + '\n')
                with self.assertRaises(AssertionError):
                    o.readback(output, root, 2, **kwargs)
                self.assertFalse((output / 'readback.json').exists())

    def test_actual_full_label_chain_forwards_and_readback_binding(self):
        a, c = [300, 10, 400, 100], [800, 10, 900, 100]
        boxes = [a, a, c]
        objects = [dict(coco_ann_id=1, desc='person', bbox_2d=a),
                   dict(coco_ann_id=2, desc='traffic light', bbox_2d=c),
                   dict(coco_ann_id=3, desc='person', bbox_2d=[900, 10, 915, 100]),
                   dict(coco_ann_id=4, desc='person', bbox_2d=[920, 10, 935, 100]),
                   dict(coco_ann_id=5, desc='person', bbox_2d=[940, 10, 955, 100]),
                   dict(coco_ann_id=6, desc='person', bbox_2d=[960, 10, 975, 100])]
        image, record, _ = self.fixture(boxes, objects)
        text = ''.join([
            '<|object_ref_start|>person<|object_ref_end|><|box_start|>' + ''.join(f'<|coord_{x}|>' for x in a) + '<|box_end|>',
            '<|object_ref_start|>person<|object_ref_end|><|box_start|>' + ''.join(f'<|coord_{x}|>' for x in a) + '<|box_end|>',
            '<|object_ref_start|>traffic light<|object_ref_end|><|box_start|>' + ''.join(f'<|coord_{x}|>' for x in c) + '<|box_end|>',
            '<|im_end|>',
        ])
        raw = dict(record, text=text, token_ids=self.t.encode(text, add_special_tokens=False),
                   generated_tokens=len(self.t.encode(text, add_special_tokens=False)), stop_reason='im_end')
        producer = self.region_producer(completion_weighting=o.RESTORED_M_WEIGHTING,
                                        redirect_selection=o.IDENTITY_SELECTION)
        record = o.seal(raw, producer)
        plan = o.completion_credit(image, record, self.t, producer, 'treatment', True,
                                   o.RESTORED_M_WEIGHTING)
        self.assertEqual((plan['m'], plan['k']), (2, 4))
        self.assertEqual(plan['redirects'][0]['description'], 'traffic light')

        class Toy(torch.nn.Module):
            def __init__(self, vocab):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(vocab))
                self.histories = []

            def forward(self, input_ids, logits_to_keep, **kwargs):
                self.histories.append(input_ids.detach().tolist()[0])
                return SimpleNamespace(logits=self.w[None, None, :].expand(1, len(logits_to_keep), -1))

        model = Toy(self.vocab.vocab_size)
        q = SimpleNamespace(model=model, tokenizer=self.t)
        batch = SimpleNamespace(inputs={})
        histories = []

        def exact_history(_model, _inputs, full, pad_token_id):
            histories.append(list(full[0]))
            return {'input_ids': torch.tensor([full[0]])}

        tensor = torch.tensor
        rows = {}
        with patch('src.qwen.native.exact_history_inputs', side_effect=exact_history), \
             patch.object(torch, 'autocast', side_effect=lambda *args, **kwargs: nullcontext()), \
             patch.object(torch, 'tensor', side_effect=lambda data, **kwargs:
                          tensor(data, **{key: value for key, value in kwargs.items() if key != 'device'})):
            for branch, index in (('trace', None), ('bridge', None), ('redirect', 0)):
                model.zero_grad(set_to_none=True)
                loss, row = o.forward(q, model, batch, image, record, plan, None, self.vocab,
                                      branch, geometry_weight=.1, branch_index=index,
                                      correction_arm='treatment', duplicate_weight=1)
                row.update(image_weight=8/18, sync=True)
                loss.backward()
                self.assertTrue(torch.isfinite(model.w.grad).all())
                self.assertEqual(row['owner_region'], o.OWNER_REGION)
                self.assertEqual(row['training_sha256'], producer['training_sha256'])
                if branch == 'trace':
                    self.assertEqual(row['terms']['M'], 0)
                    self.assertEqual(row['row_losses'], [])
                    self.assertEqual(row['replay_ordering']['checked'], 4)
                    self.assertTrue(any(x['hf_argmax_token'] != x['emitted_token']
                                        for x in row['cached_replay']))
                elif branch == 'bridge':
                    self.assertEqual([entry['kind'] for entry in row['row_losses']], ['B'] * 4 + ['M'] * 2)
                    self.assertTrue(all('owner_region' in entry for entry in row['row_losses']))
                    self.assertTrue(all(atom['causal_logits_position'] == atom['target_position'] - 1
                                        for entry in row['row_losses'] for atom in entry['atoms']))
                else:
                    self.assertEqual(row['redirect']['site_kind'], 'desc_text')
                    self.assertGreater(row['terms']['redirect_margin'], 0)
                    self.assertTrue(row['row_losses'][0]['owner_region']['sites'])
                component_names = ('lexical_schema_ce', 'coordinate_region_hinge',
                                   'type_gate_weighted', 'conditional_order_weighted')
                for entry in row['row_losses']:
                    components = entry['loss_components']
                    self.assertEqual(set(components), {'reduction_weight', 'row_loss', *component_names})
                    self.assertAlmostEqual(sum(components[name] for name in component_names),
                                           components['row_loss'], places=5)
                    self.assertTrue(all(components[name] > 0 for name in component_names))
                if branch == 'trace':
                    self.assertAlmostEqual(sum(value['loss_components']['reduction_weight'] *
                                               value['loss_components']['row_loss']
                                               for value in row['row_losses']), row['terms']['M'], places=5)
                elif branch == 'bridge':
                    for kind, name in (('B', 'B'), ('M', 'M_relocated')):
                        self.assertAlmostEqual(sum(value['loss_components']['reduction_weight'] *
                                                   value['loss_components']['row_loss']
                                                   for value in row['row_losses'] if value['kind'] == kind),
                                               row['terms'][name], places=5)
                else:
                    self.assertAlmostEqual(row['row_losses'][0]['loss_components']['reduction_weight'] *
                                           row['row_losses'][0]['loss_components']['row_loss'],
                                           row['terms']['redirect_positive'], places=5)
                rows[branch] = row
                ids = [image['image_id']] + list(range(100000, 100017))
                job = dict(image_id=image['image_id'], branch=branch, sync=True, weight=8/18)
                if branch == 'redirect':
                    job['branch_index'] = 0
                with patch.object(o, 'jobs', return_value=[job]):
                    o.verify_correction_forwards([row], ids, 0, {image['image_id']: plan},
                                                 {image['image_id']: record}, {image['image_id']: image},
                                                 self.t, 'treatment', 1)
                if branch == 'trace':
                    changed = copy.deepcopy(row)
                    changed['cached_replay'][0]['prefix_sha256'] = 'wrong-prefix'
                    with patch.object(o, 'jobs', return_value=[job]), self.assertRaises(AssertionError):
                        o.verify_correction_forwards([changed], ids, 0, {image['image_id']: plan},
                                                     {image['image_id']: record}, {image['image_id']: image},
                                                     self.t, 'treatment', 1)
                for field in ('owner_region', 'training_sha256'):
                    changed = copy.deepcopy(row)
                    changed.pop(field)
                    with patch.object(o, 'jobs', return_value=[job]), self.assertRaises(AssertionError):
                        o.verify_correction_forwards([changed], ids, 0, {image['image_id']: plan},
                                                     {image['image_id']: record}, {image['image_id']: image},
                                                     self.t, 'treatment', 1)
                changed_plan = copy.deepcopy(plan)
                changed_plan.pop('owner_region')
                with patch.object(o, 'jobs', return_value=[job]), self.assertRaises(AssertionError):
                    o.verify_correction_forwards([row], ids, 0, {image['image_id']: changed_plan},
                                                 {image['image_id']: record}, {image['image_id']: image},
                                                 self.t, 'treatment', 1)
            complete_image=dict(image,objects=objects[:2])
            complete_producer=self.region_producer(completion_weighting=o.RESTORED_M_WEIGHTING,
                                                   redirect_selection=o.IDENTITY_SELECTION)
            complete_record=o.seal(raw,complete_producer)
            complete_plan=o.completion_credit(complete_image,complete_record,self.t,complete_producer,
                                               'treatment',True,o.RESTORED_M_WEIGHTING)
            self.assertEqual(complete_plan['k'],0)
            model.zero_grad(set_to_none=True)
            loss,row=o.forward(q,model,batch,complete_image,complete_record,complete_plan,None,self.vocab,
                               'trace',geometry_weight=.1,correction_arm='treatment',duplicate_weight=1)
            self.assertGreater(row['terms']['M'],0)
            self.assertTrue(row['row_losses'])
            names=('lexical_schema_ce','coordinate_region_hinge','type_gate_weighted','conditional_order_weighted')
            self.assertAlmostEqual(sum(value['loss_components']['reduction_weight']*
                                       value['loss_components']['row_loss'] for value in row['row_losses']),
                                   row['terms']['M'],places=5)
            self.assertTrue(all(all(value['loss_components'][name]>0 for name in names)
                                for value in row['row_losses']))
            trace_job=dict(image_id=image['image_id'],branch='trace',sync=True,weight=8/18)
            with patch.object(o,'jobs',return_value=[trace_job]):
                o.verify_correction_forwards([dict(row,image_weight=8/18,sync=True)],ids,0,
                    {image['image_id']:complete_plan},{image['image_id']:complete_record},
                    {image['image_id']:complete_image},self.t,'treatment',1)
            loss.backward()
            self.assertTrue(torch.isfinite(model.w.grad).all())
        self.assertEqual(len(histories), 4)
        coord_image, coord_record, _ = self.helper().multi_fixture()
        coord_producer = self.region_producer(completion_weighting=o.RESTORED_M_WEIGHTING,
                                              redirect_selection=o.IDENTITY_SELECTION)
        coord_record = o.seal(coord_record, coord_producer)
        coord_plan = o.completion_credit(coord_image, coord_record, self.t, coord_producer, 'treatment', True,
                                         o.RESTORED_M_WEIGHTING)
        coordinate_target = coord_plan['redirects'][0]
        sequence = o.redirect_sequence(coord_image, coord_record, coordinate_target, self.t)
        positions = tuple(atom.causal_logits_position for atom in sequence.atoms)
        logits = torch.zeros((1, len(positions), self.vocab.vocab_size), requires_grad=True)
        parts = {}
        value, terms = o.redirect_objective(logits, positions, sequence, coordinate_target,
                                            len(coord_record['prompt_token_ids']), self.vocab, coord_image,
                                            o.OWNER_REGION, parts)
        receipt = o.owner_loss_component_receipt(parts, coordinate_target.get('event_weight', 1))
        self.assertAlmostEqual(sum(receipt[name] for name in ('lexical_schema_ce', 'coordinate_region_hinge',
                                                               'type_gate_weighted', 'conditional_order_weighted')),
                               receipt['row_loss'], places=5)
        self.assertAlmostEqual(receipt['reduction_weight'] * receipt['row_loss'],
                               terms['redirect_positive'].item(), places=5)
        margin_grad, = torch.autograd.grad(terms['redirect_margin'], logits)
        self.assertEqual(float(terms['redirect_margin'].detach()), 0)
        self.assertEqual(float(margin_grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
