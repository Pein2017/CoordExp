from pathlib import Path

import pytest

from src.eval.assignment import global_matches
from src.vis.matching import match_row
from src.vis.normalization import VisualObject, VisualRow


def test_cardinality_first_differs_from_greedy_visualization():
    gt = [('person', (0., 0., 100., 100.)), ('person', (40., 0., 140., 100.))]
    pred = [('person', (20., 0., 120., 100.)), ('person', (0., 0., 60., 100.))]
    assert [(i, j) for i, j, _ in global_matches(gt, pred, .5)] == [(0, 1), (1, 0)]
    def objects(rows):
        return tuple(VisualObject(i, c, c, box, box, 'pixel') for i, (c, box) in enumerate(rows))
    row = VisualRow('fixture', 0, Path('image'), 'image', 1000, 1000, objects(gt), objects(pred))
    assert len(match_row(row, match_iou_threshold=.5).matches) == 1


def test_category_threshold_empty_and_deterministic_ties():
    box = (0., 0., 100., 100.)
    assert global_matches([], [('person', box)], .5) == []
    assert global_matches([('Person', box)], [('person', box)], .5) == []
    assert global_matches([('person', box)], [('person', (0., 0., 50., 100.))], .5) == [(0, 0, .5)]
    assert global_matches([('person', box)], [('person', (0., 0., 50., 100.))], .500001) == []
    assert global_matches([('person', box)] * 2, [('person', box)] * 2, .5) == [(0, 0, 1.), (1, 1, 1.)]


def test_quantized_iou_is_secondary_objective():
    gt = [('person', (0., 0., 1., 1.))]
    pred = [('person', (0., 0., .75, 1.)), ('person', (0., 0., .7500000001, 1.))]
    # Equal integer cost retains earlier edge despite a slightly greater float IoU.
    assert global_matches(gt, pred, .5) == [(0, 0, .75)]
    pred[1] = ('person', (0., 0., .750000001, 1.))
    assert global_matches(gt, pred, .5) == [(0, 1, pytest.approx(.750000001))]
