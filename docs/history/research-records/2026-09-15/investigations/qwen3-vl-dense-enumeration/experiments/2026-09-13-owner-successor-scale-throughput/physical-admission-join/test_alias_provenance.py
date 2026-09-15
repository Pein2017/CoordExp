from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from alias_provenance import derive_required_alias_provenance


class AliasProvenanceTest(unittest.TestCase):
    def test_missing_consumer_fields_are_derived_from_exact_group(self):
        rows = [
            {"job_id": "1:h0:c0", "visual_group_id": "G1", "image_id": 1},
            {"job_id": "1:h1:c0", "visual_group_id": "G1", "image_id": 1},
        ]
        groups = [{"visual_group_id": "G1", "image_id": 1,
                   "job_ids": ["1:h0:c0", "1:h1:c0"]}]
        with self.assertRaises(KeyError):
            _ = rows[0]["same_image_aliases"]  # Exact old consumer failure.
        additions = derive_required_alias_provenance(rows, groups)
        self.assertEqual(len(additions), 2)
        self.assertEqual(rows[0]["same_image_aliases"], ["1:h1:c0"])
        self.assertEqual(rows[1]["same_image_aliases"], ["1:h0:c0"])
        self.assertEqual(rows[0]["execution_aliases_same_visual_group"], groups[0]["job_ids"])

    def test_same_image_different_group_does_not_create_alias(self):
        rows = [
            {"job_id": "1:h0:c0", "visual_group_id": "G1", "image_id": 1},
            {"job_id": "1:h0:c1", "visual_group_id": "G2", "image_id": 1},
        ]
        groups = [
            {"visual_group_id": "G1", "image_id": 1, "job_ids": ["1:h0:c0"]},
            {"visual_group_id": "G2", "image_id": 1, "job_ids": ["1:h0:c1"]},
        ]
        derive_required_alias_provenance(rows, groups)
        self.assertEqual(rows[0]["same_image_aliases"], [])
        self.assertEqual(rows[1]["same_image_aliases"], [])

    def test_existing_review_provenance_is_preserved_without_reinterpretation(self):
        rows = [
            {"job_id": "1:h0:c0", "visual_group_id": "G1", "image_id": 1,
             "same_image_aliases": ["1:h0:c0", "1:h1:c0"],
             "execution_aliases_same_visual_group": ["1:h0:c0"]},
            {"job_id": "1:h1:c0", "visual_group_id": "G2", "image_id": 1,
             "same_image_aliases": ["1:h0:c0", "1:h1:c0"],
             "execution_aliases_same_visual_group": ["1:h1:c0"]},
        ]
        original = deepcopy(rows)
        groups = [
            {"visual_group_id": "G1", "image_id": 1, "job_ids": ["1:h0:c0"]},
            {"visual_group_id": "G2", "image_id": 1, "job_ids": ["1:h1:c0"]},
        ]
        self.assertEqual(derive_required_alias_provenance(rows, groups), [])
        self.assertEqual(rows, original)


if __name__ == "__main__":
    unittest.main()
