from itertools import combinations, product
import unittest

import torch

from probes.owner_region_ranking import acceptable_bins, owner_slots, region_margin


def iou_at_least(box, gt, tau):
    if box[0] >= box[2] or box[1] >= box[3]:
        return False
    iw = max(0, min(box[2], gt[2]) - max(box[0], gt[0]))
    ih = max(0, min(box[3], gt[3]) - max(box[1], gt[1]))
    inter = iw * ih
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter)
    return union > 0 and inter >= tau * union


class OwnerRegionRankingTest(unittest.TestCase):
    def test_tiny_grid_matches_exhaustive_suffix_existence(self):
        max_bin = 3
        grid = range(max_bin + 1)
        boxes = [(x1, y1, x2, y2)
                 for x1, x2 in combinations(grid, 2)
                 for y1, y2 in combinations(grid, 2)]
        for gt in boxes:
            for size in range(4):
                for prefix in product(grid, repeat=size):
                    expected = []
                    for candidate in grid:
                        start = prefix + (candidate,)
                        if any(iou_at_least(start + suffix, gt, 0.5)
                               for suffix in product(grid, repeat=3-size)):
                            expected.append(candidate)
                    self.assertEqual(acceptable_bins(gt, prefix, max_bin=max_bin), expected,
                                     (gt, prefix))

    def test_joint_completion_can_fail_when_each_axis_overlaps(self):
        gt = (0, 0, 5, 5)
        # Each 1D overlap is 3/5, but the best joint box has IoU 9/25 < 1/2.
        self.assertEqual(acceptable_bins(gt, (2, 2), max_bin=7), [])
        # The y2 candidate 2 makes a half-width box with IoU exactly 0.5.
        self.assertIn(2, acceptable_bins((0, 0, 2, 2), (0, 0, 1), max_bin=3))

    def test_owner_stops_at_first_impossible_prefix(self):
        gt = (100, 100, 400, 400)
        slots, regions, failure = owner_slots(gt, [100, 100, 900, 200])
        self.assertEqual((slots, failure), ([0, 1, 2], 2))
        self.assertIn(100, regions[0])
        self.assertIn(100, regions[1])
        self.assertTrue(regions[2])
        self.assertEqual(acceptable_bins(gt, (999,)), [])
        self.assertEqual(owner_slots(gt, [100, 100, 100, 100])[0], [0, 1, 2])

    def test_integer_and_grid_boundaries(self):
        gt = (0, 0, 999, 999)
        self.assertEqual(owner_slots(gt, [0, 0, 999, 999])[0], [0, 1, 2, 3])
        for bad in ((False, 0, 999, 999), (0.0, 0, 999, 999),
                    (0, 0, 1000, 999), (1, 0, 1, 999)):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                acceptable_bins(bad)
        with self.assertRaises(ValueError):
            acceptable_bins(gt, tau=float("nan"))
        with self.assertRaises(ValueError):
            acceptable_bins(gt, tau=1.01)
        with self.assertRaises(ValueError):
            acceptable_bins(gt, tau=0)
        with self.assertRaises(ValueError):
            acceptable_bins(gt, max_bin=1000)
        with self.assertRaises(ValueError):
            owner_slots(gt, [0, 0, 999, 1000])
        with self.assertRaises(TypeError):
            owner_slots(gt, [0.0, 0, 999, 999])

    def test_margin_uses_max_valid_logit_and_full_vocabulary(self):
        valid_alternative_wins = region_margin(torch.tensor([0.0, 1.0, 0.9]), [1, 2])
        self.assertEqual(float(valid_alternative_wins), 0.0)
        # A log-sum-exp over the two acceptable tokens would hide this violation.
        nonmax_valid_mass = region_margin(torch.tensor([0.5, 0.0, 0.0]), [1, 2], margin=0)
        self.assertEqual(float(nonmax_valid_mass), 0.5)
        # Token 2 is outside the coordinate alternatives and still competes.
        noncoordinate_escape = region_margin(torch.tensor([0.0, 0.0, 5.0]), [0, 1], margin=0)
        self.assertEqual(float(noncoordinate_escape), 5.0)
        with self.assertRaises(ValueError):
            region_margin(torch.zeros(3), [], margin=0)


if __name__ == "__main__":
    unittest.main()
