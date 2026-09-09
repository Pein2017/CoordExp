"""Focused consumer corruption and terminal-action counterexamples (CPU only)."""

import copy
import json
import unittest

from probes.source_rweak_row_cross import prepare as p


class FrozenManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os

        manifest_path = os.environ.get("ROW_CROSS_MANIFEST")
        if not manifest_path:
            raise unittest.SkipTest("ROW_CROSS_MANIFEST selects the saved-input check")
        from pathlib import Path

        cls.manifest = json.loads(Path(manifest_path).read_text())
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.manifest["sources"]["source"]["config"]["model"]["base_model"],
            local_files_only=True,
        )

    def test_valid_frozen_consumer(self):
        p.validate_manifest(self.manifest)

    def test_corrupt_token_rejected(self):
        m = copy.deepcopy(self.manifest)
        m["cases"][0]["actions"]["source"]["token_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "token mismatch"):
            p.validate_manifest(m)

    def test_corrupt_row_identity_rejected(self):
        m = copy.deepcopy(self.manifest)
        m["cases"][0]["diagonals"]["source"]["raw_record"]["row_id"] = "wrong-image"
        with self.assertRaisesRegex(ValueError, "row identity"):
            p.validate_manifest(m)

    def test_corrupt_denominator_rejected(self):
        m = copy.deepcopy(self.manifest)
        m["selection"]["population_gt"] = 3758
        with self.assertRaisesRegex(ValueError, "denominator"):
            p.validate_manifest(m)

    def test_corrupt_remaining_budget_rejected(self):
        m = copy.deepcopy(self.manifest)
        m["cases"][0]["remaining_token_budgets"]["rweak"] += 1
        with self.assertRaisesRegex(ValueError, "remaining budget"):
            p.validate_manifest(m)

    def test_original_trace_token_corruption_rejected(self):
        diag = self.manifest["cases"][0]["diagonals"]["source"]
        entries = [
            {
                "row_id": diag["raw_record"]["row_id"],
                "generated_step_index": i,
                "token_id": token,
                "token_text": self.tokenizer.decode(
                    [token],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "is_pad": False,
                "is_stop": token == 151645,
            }
            for i, token in enumerate(diag["generated_token_ids"])
        ]
        self.assertEqual(
            p.verify_trace(entries, diag["raw_record"], self.tokenizer),
            diag["generated_token_ids"],
        )
        entries[0]["token_id"] += 1
        with self.assertRaisesRegex(ValueError, "token/text reconstruction"):
            p.verify_trace(entries, diag["raw_record"], self.tokenizer)

    def test_legal_terminal_eos_is_retained(self):
        case = self.manifest["cases"][0]
        row = case["diagonals"]["source"]["raw_record"]
        tokens = case["actions"]["source"]["token_ids"]
        prefix, actions = p.divergence([151645], tokens + [151645], self.tokenizer, row)
        self.assertEqual(prefix, [])
        self.assertEqual(actions["source"]["kind"], "eos")
        self.assertEqual(actions["source"]["token_ids"], [151645])
        with self.assertRaisesRegex(ValueError, "nonterminal_eos"):
            p.next_action([151645] + tokens, 0, self.tokenizer, row)

    def test_rweak_completed_update_is_bound_before_loading(self):
        import tempfile
        from pathlib import Path
        from probes.source_rweak_row_cross import run

        self.assertEqual(
            run.load_rweak_checkpoint(self.manifest)["completed_update"], 64
        )
        changed = copy.deepcopy(self.manifest)
        checkpoint = changed["sources"]["rweak_checkpoint"]["identity"]
        checkpoint["completed_update"] = 63
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(checkpoint))
            changed["sources"]["rweak_checkpoint"]["manifest"] = {
                "path": str(path),
                "sha256": run.sha(path),
                "bytes": path.stat().st_size,
            }
            with self.assertRaisesRegex(ValueError, "arm/surface/update"):
                run.load_rweak_checkpoint(changed)

    def test_direct_row_b_vs_c_terminal_is_not_later_recovery(self):
        # Both complete outcomes terminate immediately after the divergent row.
        # B -> C replaces a primary owner despite an unchanged empty suffix;
        # a suffix-only metric would erase the genuine full-output loss of B.
        gt = [("person", (0.0, 0.0, 10.0, 10.0)), ("person", (20.0, 0.0, 30.0, 10.0))]
        b, c = [gt[0]], [gt[1]]
        source = {i for i, _, _ in p._global_matches(gt, b, 0.5)}
        crossed = {i for i, _, _ in p._global_matches(gt, c, 0.5)}
        self.assertEqual(source - crossed, {0})
        self.assertEqual(crossed - source, {1})
        self.assertEqual(p._global_matches(gt, [], 0.5), [])
        self.assertEqual(len(source), len(crossed))


if __name__ == "__main__":
    unittest.main()
