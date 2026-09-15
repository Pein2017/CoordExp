"""CPU-only real-consumer reads and isolated selection publication; no model calls."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from probes.parallel_owner_research import transfer

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'docs/history/research-records/2026-09-15-root-collapse/manifest.json'
ARCHIVE = Path('docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration')


class TransferExposureContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(MANIFEST.read_text())['research_json_exposure']

    def test_real_consumer_keeps_every_source_hash_and_image_id(self):
        records = transfer.research_exposure_sources()
        actual = {str(Path(r['path']).relative_to(ROOT)): r for r in records}
        expected = {r['archive_path']: r for r in self.baseline['records']}
        self.assertEqual(set(actual), set(expected))
        for name, row in actual.items():
            self.assertEqual(row['sha256'], expected[name]['sha256'], name)
            self.assertEqual(row['image_ids'], expected[name]['image_ids'], name)
        ids = set().union(*(set(row['image_ids']) for row in records))
        self.assertEqual(sorted(ids), self.baseline['excluded_image_ids'])
        self.assertEqual((len(records), len(ids)), (106, 135))

    def test_missing_corpus_fails_instead_of_empty_exclusions(self):
        with tempfile.TemporaryDirectory(prefix='exposure-missing-') as tmp:
            with patch.object(transfer, 'WORKTREE', Path(tmp)):
                with self.assertRaisesRegex(ValueError, 'corpus missing'):
                    transfer.research_exposure_sources()

    def test_empty_corpus_fails(self):
        with tempfile.TemporaryDirectory(prefix='exposure-empty-') as tmp:
            (Path(tmp) / ARCHIVE).mkdir(parents=True)
            with patch.object(transfer, 'WORKTREE', Path(tmp)):
                with self.assertRaisesRegex(ValueError, 'corpus empty'):
                    transfer.research_exposure_sources()

    def test_study_self_exclusion_and_bad_json(self):
        with tempfile.TemporaryDirectory(prefix='exposure-filter-') as tmp:
            corpus = Path(tmp) / ARCHIVE
            excluded = corpus / '2026-09-12-parallel-owner-research/self.json'
            excluded.parent.mkdir(parents=True)
            excluded.write_text('{"image_id": 900}')
            included = corpus / 'case.json'
            included.write_text('{"image_id": 42}')
            with patch.object(transfer, 'WORKTREE', Path(tmp)):
                records = transfer.research_exposure_sources()
                self.assertEqual([r['image_ids'] for r in records], [[42]])
                included.write_text('invalid JSON')
                with self.assertRaises(json.JSONDecodeError):
                    transfer.research_exposure_sources()

    def test_actual_selection_entry_publishes_equivalent_ids_in_temporary_root(self):
        # Keep the actual research archive reader. Only the output-manifest and
        # input universe are synthetic, and publication is confined to tmp.
        with tempfile.TemporaryDirectory(prefix='transfer-selection-fixture-') as tmp:
            base = Path(tmp)
            output = base / 'selection-output'
            source = base / 'synthetic.coord.jsonl'
            universe = list(range(1, 2001))
            source.write_text(''.join(json.dumps({'image_id': i}) + '\n' for i in universe))
            prior = base / 'prior-input-manifest.json'
            prior.write_text(json.dumps({'image_ids': [3, 7, 11]}))
            excluded = set(self.baseline['excluded_image_ids']) | {3, 7, 11}
            expected = sorted(set(universe) - excluded, key=lambda i: hashlib.sha256(
                f'{transfer.SALT}{i}'.encode()).hexdigest())[:256]
            with patch.object(transfer, 'ROOT', output), patch.object(transfer, 'SOURCE', source), \
                    patch.object(transfer.subprocess, 'check_output', return_value=str(prior) + '\n'), \
                    contextlib.redirect_stdout(io.StringIO()):
                transfer.freeze_selection()
                path = output / 'selection.json'
                before = path.read_bytes()
                result = json.loads(before)
                self.assertEqual(result['image_ids'], expected)
                self.assertEqual(result['excluded_image_ids'], sorted(excluded))
                self.assertEqual(len(result['exposure_sources']), 107)
                self.assertTrue(all(Path(r['path']).is_file() for r in result['exposure_sources']))
                with self.assertRaisesRegex(ValueError, 'already frozen'):
                    transfer.freeze_selection()
                self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
