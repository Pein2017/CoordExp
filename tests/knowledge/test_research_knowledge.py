"""Synthetic CPU contracts for the flat research tree and path-only migration resolver."""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools import research_knowledge as CHECK

ROOT = Path(__file__).resolve().parents[2]






class KnowledgeContracts(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='research-knowledge-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.unit = '2026-01-01-example'
        self.original = f'research/investigations/example/experiments/{self.unit}/unit.md'
        self.archived = 'docs/history/first/' + self.original
        self.entry = self.snapshot(self.original, self.archived, '# Frozen question\n')
        self.capture = {'baseline_head': 'a' * 40, 'source_prefix': 'research/investigations',
                        'archive_prefix': 'docs/history/first/research/investigations',
                        'files': [self.entry]}
        self.bundle = {'canonical_root': '/original/repo', 'captures': [self.capture],
                       'exposure': {}}
        self.live = f'research/experiments/{self.unit}'
        self.state_path = self.live + '/state.json'
        self.result_path = self.live + '/results.md'
        for name in CHECK.ROOT_NAMES:
            if '.' in name:
                self.write('research/' + name, '# Entry\n')
            else:
                (self.root / 'research' / name).mkdir(parents=True)
        self.write('research/questions/example.md', '# Question\n')
        self.write(self.result_path, '# Accepted result\n')
        self.state = {'schema_version': 1, 'unit_id': self.unit, 'lifecycle': 'paused',
                      'evidence': 'accepted', 'disposition': 'stage_incomplete',
                      'state_as_of': '2026-01-01', 'protocol': self.archived,
                      'result': self.result_path, 'state_source': self.result_path,
                      'boundary': 'No resumption.', 'not_authorized': ['GPU'],
                      'next_action': 'Discuss retained evidence.'}
        self.save_state()
        self.write('research/index.md', f'[Catalog](experiments/catalog.jsonl)\n[State](experiments/{self.unit}/state.json)\n')
        self.row = {'id': self.unit, 'title': 'Example', 'kind': 'experiment',
                    'topics': ['example'], 'record_root': str(Path(self.archived).parent),
                    'reading_entry': self.result_path, 'protocols': [self.archived],
                    'result_records': [self.result_path], 'tracking': 'current',
                    'state': self.state_path}

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def snapshot(self, source: str, archive: str, text: str) -> dict:
        path = self.write(archive, text)
        return {'source': source, 'archive': archive}

    def save_state(self) -> None:
        self.write(self.state_path, json.dumps(self.state))

    def catalog_errors(self, rows: list | None = None) -> list:
        selected = rows or [self.row]
        return CHECK.check_catalog(self.root, selected) + CHECK.check_migration_catalog(self.root, selected, self.bundle)

    def test_clean_source_catalog_and_layout(self):
        self.assertEqual(self.catalog_errors(), [])
        self.assertEqual(CHECK.check_layout(self.root), [])

    def test_historical_neighbor_uses_original_coordinates(self):
        source = str(Path(self.original).with_name('results.md'))
        archived = str(Path(self.archived).with_name('results.md'))
        self.capture['files'].append(self.snapshot(source, archived, '# Result\n'))
        ref = CHECK.resolve_reference(self.root, self.bundle, self.archived, 'results.md#result')
        self.assertEqual(ref['path'], archived)
        self.assertTrue(ref['exists'])
        self.assertEqual(ref['fragment'], 'result')

    def test_history_uses_other_capture_when_old_neighbor_was_retired_later(self):
        source = 'research/ideas/old/overview.md'
        archived = 'docs/history/second/sources/' + source
        entry = self.snapshot(source, archived, '# Old idea\n')
        self.bundle['captures'].append({'baseline_head': 'b' * 40, 'files': [entry]})
        ref = CHECK.resolve_reference(self.root, self.bundle, self.archived,
                                      '/original/repo/' + source)
        self.assertEqual(ref['path'], archived)
        self.assertTrue(ref['exists'])

    def test_live_index_not_redirected_to_old_snapshot(self):
        old = self.snapshot('research/index.md', 'docs/history/second/index.md', '# Prior\n')
        self.bundle['captures'].append({'baseline_head': 'b' * 40, 'files': [old]})
        ref = CHECK.resolve_reference(self.root, self.bundle, 'research/questions/example.md', '../index.md')
        self.assertEqual(ref['path'], 'research/index.md')

    def test_live_missing_old_path_is_not_silently_repaired(self):
        ref = CHECK.resolve_reference(self.root, self.bundle, 'research/index.md',
                                      self.original.removeprefix('research/'))
        self.assertFalse(ref['exists'])
        self.assertEqual(ref['path'], self.original)

    def test_snapshot_self_reference_stays_in_own_capture(self):
        second = self.snapshot(self.original, 'docs/history/second/unit.md', '# Later\n')
        self.bundle['captures'].append({'baseline_head': 'b' * 40, 'files': [second]})
        first = CHECK.resolve_reference(self.root, self.bundle, self.archived, '#old')
        later = CHECK.resolve_reference(self.root, self.bundle, second['archive'], '#new')
        self.assertEqual(first['path'], self.archived)
        self.assertEqual(later['path'], second['archive'])

    def test_external_handles_remain_unverified(self):
        for target in ('https://example.org/a', '/another/worktree/a'):
            ref = CHECK.resolve_reference(self.root, self.bundle, 'research/index.md', target)
            self.assertEqual(ref['kind'], 'external')
            self.assertIsNone(ref['exists'])

    def test_traversal_and_symlink_escape_rejected(self):
        for path in ('../secret', '/etc/passwd', 'a\nnew'):
            with self.assertRaises(ValueError):
                CHECK.local_path(self.root, path)
        (self.root / 'outside').symlink_to(self.root.parent)
        with self.assertRaises(ValueError):
            CHECK.local_path(self.root, 'outside/unknown')

    def test_live_link_contract(self):
        path = self.write('research/story.md', '[Good](index.md) [Bad](missing.md)\n')
        errors, local, external = CHECK.check_live_links(self.root, self.bundle, [path])
        self.assertEqual(len(errors), 1)
        self.assertEqual((local, external), (1, 0))

    def test_fenced_examples_not_links(self):
        self.assertEqual(CHECK.links_in('```text\n[x](missing)\n```\n[x](index.md)'), ['index.md'])

    def test_removed_layers_and_empty_placeholder_fail(self):
        (self.root / 'research/mechanisms').mkdir()
        self.assertTrue(CHECK.check_layout(self.root))

    def test_root_alias_and_executable_fail(self):
        (self.root / 'research/alias').symlink_to(self.root / 'docs/history')
        self.write('research/questions/helper.py', 'pass\n')
        self.assertGreaterEqual(len(CHECK.check_layout(self.root)), 2)

    def test_duplicate_catalog_id_fails(self):
        self.assertTrue(self.catalog_errors([self.row, copy.deepcopy(self.row)]))

    def test_missing_protocol_in_catalog_fails(self):
        self.row['protocols'] = []
        self.assertTrue(self.catalog_errors())

    def test_missing_question_fails(self):
        self.row['topics'] = ['missing']
        self.assertTrue(self.catalog_errors())

    def test_missing_catalog_target_fails(self):
        self.row['result_records'] = ['research/missing.md']
        self.assertTrue(self.catalog_errors())

    def test_historical_row_cannot_own_current_state(self):
        self.row['tracking'] = 'historical'
        self.assertTrue(self.catalog_errors())

    def test_current_transport_is_not_owner(self):
        self.row['reading_entry'] = self.live + '/handoff.md'
        self.write(self.row['reading_entry'], '# Transport\n')
        self.assertTrue(self.catalog_errors())

    def test_closed_and_paused_states_need_catalog_not_frontier_duplication(self):
        for lifecycle in ('closed', 'paused'):
            with self.subTest(lifecycle=lifecycle):
                self.state['lifecycle'] = lifecycle
                self.save_state()
                self.write('research/index.md', '[Catalog](experiments/catalog.jsonl)\n')
                self.assertEqual(self.catalog_errors(), [])

    def test_catalog_requires_a_real_local_navigation_link(self):
        examples = (
            '# No catalog link\n',
            'experiments/catalog.jsonl\n',
            '[Catalog](https://example.org/experiments/catalog.jsonl)\n',
            '```markdown\n[Catalog](experiments/catalog.jsonl)\n```\n',
            '<!-- [Catalog](experiments/catalog.jsonl) -->\n',
            '![Not navigation](experiments/catalog.jsonl)\n',
            '`[Catalog](experiments/catalog.jsonl)`\n',
        )
        for text in examples:
            with self.subTest(text=text):
                self.write('research/index.md', text + f'[State](experiments/{self.unit}/state.json)\n')
                self.assertIn('frontier omits experiment catalog link', self.catalog_errors())

    def test_catalog_navigation_accepts_normalized_local_path(self):
        self.write('research/index.md', '[Catalog](./experiments/catalog.jsonl)\n')
        self.assertEqual(self.catalog_errors(), [])

    def test_orphan_state_fails(self):
        self.write('research/experiments/orphan/state.json', json.dumps(self.state))
        self.assertTrue(self.catalog_errors())

    def test_paused_accepted_incomplete_is_valid(self):
        self.assertEqual(CHECK.check_state(self.root, self.state_path, self.unit)[0], [])

    def test_invalid_lifecycle_and_evidence_fail(self):
        self.state.update(lifecycle='active_forever', evidence='probably')
        self.save_state()
        self.assertTrue(CHECK.check_state(self.root, self.state_path, self.unit)[0])

    def test_accepted_requires_result(self):
        self.state['result'] = None
        self.save_state()
        self.assertTrue(CHECK.check_state(self.root, self.state_path, self.unit)[0])

    def test_planned_can_have_no_result(self):
        self.state.update(result=None, evidence='none', lifecycle='planned')
        self.save_state()
        self.assertEqual(CHECK.check_state(self.root, self.state_path, self.unit)[0], [])

    def test_state_identity_and_boundary_fail(self):
        self.state.update(unit_id='other', boundary='', not_authorized='GPU')
        self.save_state()
        self.assertGreaterEqual(len(CHECK.check_state(self.root, self.state_path, self.unit)[0]), 3)

    def test_exposure_manifest_is_required(self):
        self.assertTrue(CHECK.check_exposure(self.root, self.bundle))

    def test_current_catalog_does_not_need_historical_migration_bundle(self):
        self.assertEqual(CHECK.check_catalog(self.root, [self.row]), [])
        self.bundle['captures'][0]['files'].append({
            'source': 'research/experiments/unknown/unit.md',
            'archive': 'docs/history/missing/unit.md'})
        self.assertTrue(CHECK.check_migration_catalog(self.root, [self.row], self.bundle))
        self.assertEqual(CHECK.check_catalog(self.root, [self.row]), [])

    def test_live_check_preserves_orphan_and_bad_link_failures_without_history(self):
        self.write('research/experiments/orphan/state.json', json.dumps(self.state))
        self.assertIn('orphan/missing current state', CHECK.check_catalog(self.root, [self.row]))
        document = self.write('research/story.md', '[broken](not-there.md)\n')
        bundle = {'canonical_root': str(self.root), 'captures': []}
        errors, _, _ = CHECK.check_live_links(self.root, bundle, [document])
        self.assertEqual(len(errors), 1)


if __name__ == '__main__':
    unittest.main()
