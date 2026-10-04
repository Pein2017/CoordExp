from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from src.common.errors import ArtifactContractError
from src.vis import render_gt_vs_prediction, render_prediction_comparison
from src.vis.normalization import load_visual_rows


def _run(path: Path, *, size=(200, 100), image_path=None, gt=None, pred=None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path or path / 'image.png')
    if not image_path.exists():
        Image.new('RGB', size, (100, 100, 100)).save(image_path)
    row = {'row_id': 'dense', 'row_index': 0, 'image_path': str(image_path),
           'image_width': size[0], 'image_height': size[1], 'gt': gt or []}
    (path / 'gt_vs_pred.jsonl').write_text(json.dumps(row) + '\n')
    (path / 'gt_vs_pred_scored.jsonl').write_text(json.dumps({**row, 'pred': pred or []}) + '\n')
    return path


def _manifest(result):
    return json.loads(result.manifest_path.read_text())['items'][0]


def test_actual_image_dimensions_must_match_before_png(tmp_path):
    run = _run(tmp_path / 'run')
    Image.new('RGB', (201, 100)).save(run / 'image.png')
    with pytest.raises(ArtifactContractError) as error:
        render_gt_vs_prediction(run, tmp_path / 'out')
    assert error.value.code == 'vis.image_dimension_mismatch'
    assert not list((tmp_path / 'out').glob('*.png'))


def test_labels_and_lines_cannot_escape_image_viewport(tmp_path):
    run = _run(tmp_path / 'run', pred=[{'description': 'cat', 'bbox': [-4, 20, 220, 80]}])
    result = render_gt_vs_prediction(run, tmp_path / 'out')
    png = Image.open(result.image_paths[0])
    # The source image is 200x100, fitted to 852x426 at x=924,y=114.
    assert png.getpixel((910, 300)) == (240, 240, 240)
    assert png.getpixel((1780, 199)) == (240, 240, 240)
    assert png.getpixel((1000, 600)) == (240, 240, 240)


def test_crop_translation_focus_style_and_full_image_match(tmp_path):
    run = _run(tmp_path / 'run', gt=[
        {'description': 'cat', 'bbox': [300, 300, 500, 700]},
        {'description': 'dog', 'bbox': [750, 100, 900, 500]}], pred=[
        {'description': 'cat', 'bbox': [60, 30, 100, 70]},
        {'description': 'dog', 'bbox': [150, 10, 180, 50]},
        {'description': 'bird', 'bbox': [80, 40, 120, 80]}])
    source_image = Image.open(run / 'image.png')
    ImageDraw.Draw(source_image).rectangle((64, 35, 75, 65), fill=(10, 30, 230))
    source_image.save(run / 'image.png')
    baseline = _manifest(render_gt_vs_prediction(run, tmp_path / 'baseline'))
    result = render_gt_vs_prediction(run, tmp_path / 'focus', crop=(50, 20, 130, 100),
                                    focus_pred_indices=[0], context_alpha=.18)
    item = _manifest(result)
    assert item['match'] == baseline['match']
    assert item['gt_objects'] == baseline['gt_objects']
    assert item['pred_objects'] == baseline['pred_objects']
    view = item['view']['prediction']
    assert view['selected_pred_indices'] == [0]
    assert view['matching_scope'] == 'full_image_before_crop_and_focus'
    assert view['crop_transform']['source_crop_pixel_xyxy'] == [50, 20, 130, 100]
    assert view['crop_transform']['translate_source_pixel_xy'] == [-50, -20]
    assert view['crop_transform']['scale_xy'] == [8.775, 8.775]
    assert view['style']['context'] == {'dash': True, 'alpha': .18, 'width': 1, 'labels': False,
                                      'duplicate_hints': False}
    png = Image.open(result.image_paths[0])
    # Square crop fits 702x702 at x=999,y=114; P0 left edge is source x=60.
    assert png.getpixel((1087, 500)) == (0, 190, 90)
    # An image marker at source (70,50) follows the same crop transform as boxes.
    assert png.getpixel((1174, 377)) == (10, 30, 230)
    # P2 right edge is context, hence faint red dashes with genuine gaps.
    pixels = [png.getpixel((1613, y)) for y in range(480, 610)]
    assert (124, 90, 90) in pixels
    assert (100, 100, 100) in pixels
    assert 'full image' in result.summary.lower()


def test_focus_region_uses_centers_and_unions_exact_indices(tmp_path):
    run = _run(tmp_path / 'run', pred=[
        {'description': 'a', 'bbox': [0, 10, 100, 60]},
        {'description': 'b', 'bbox': [60, 20, 80, 40]},
        {'description': 'c', 'bbox': [150, 20, 180, 40]}])
    item = _manifest(render_gt_vs_prediction(run, tmp_path / 'out',
                    focus_region=(60, 10, 90, 50), focus_pred_indices=[2]))
    view = item['view']['prediction']
    assert view['selected_pred_indices'] == [1, 2]
    assert view['focus_region_selection_rule'] == 'box_center_in_region_inclusive'
    region_only = _manifest(render_gt_vs_prediction(run, tmp_path / 'region-only',
                            focus_region=(60, 10, 90, 50)))
    assert region_only['view']['prediction']['selected_pred_indices'] == [1]


@pytest.mark.parametrize('option', [
    {'crop': (0, 0, 201, 100)}, {'crop': (20, 0, 10, 100)},
    {'crop': (0, 0, float('nan'), 100)}, {'focus_region': (0, 0, 200, float('inf'))},
    {'focus_pred_indices': [3]}, {'focus_gt_indices': [-1]}, {'context_alpha': 1.1}])
def test_invalid_view_options_fail_before_png(tmp_path, option):
    run = _run(tmp_path / 'run')
    with pytest.raises(ArtifactContractError):
        render_gt_vs_prediction(run, tmp_path / 'out', **option)
    assert not list((tmp_path / 'out').glob('*.png'))


def test_focus_draws_selected_after_context_and_suppresses_context_dup_hint(tmp_path):
    run = _run(tmp_path / 'run', pred=[
        {'description': 'cat', 'bbox': [20, 20, 80, 60]},
        {'description': 'cat', 'bbox': [40, 20, 100, 60]}])
    result = render_gt_vs_prediction(run, tmp_path / 'out', focus_pred_indices=[0])
    png = Image.open(result.image_paths[0])
    # P0 and P1 overlap at the top edge: the selected solid red edge is last.
    assert png.getpixel((1190, 202)) == (235, 45, 45)
    # Only P0 may receive purple duplicate decoration; P1 right edge is faint red.
    assert png.getpixel((1350, 240)) != (160, 60, 220)


def test_cli_compare_has_independent_prediction_focus_and_shared_crop(tmp_path):
    left = _run(tmp_path / 'left', pred=[{'description': 'a', 'bbox': [20, 20, 80, 60]},
                                      {'description': 'b', 'bbox': [90, 20, 120, 60]}])
    right = _run(tmp_path / 'right', image_path=left / 'image.png',
                 pred=[{'description': 'b', 'bbox': [90, 20, 120, 60]},
                       {'description': 'a', 'bbox': [20, 20, 80, 60]}])
    output = tmp_path / 'out'
    command = [sys.executable, 'scripts/visualize_detection.py', 'compare', '--left-run-dir', str(left),
               '--right-run-dir', str(right), '--out-dir', str(output), '--crop', '10', '10', '140', '90',
               '--left-focus-pred', '0', '--right-focus-pred', '1', '--context-alpha', '.2']
    completed = subprocess.run(command, text=True, capture_output=True,
                               cwd=Path(__file__).resolve().parents[1])
    assert completed.returncode == 0, completed.stderr
    item = json.loads((output / 'manifest.json').read_text())['items'][0]
    assert item['view']['left']['selected_pred_indices'] == [0]
    assert item['view']['right']['selected_pred_indices'] == [1]
    assert item['view']['left']['crop_transform']['source_crop_pixel_xyxy'] == [10, 10, 140, 90]
    assert Image.open(output / '0000_dense_prediction_comparison.png').size == (1800, 830)


def test_comparison_reveals_selected_matched_gt_and_fades_both_panels(tmp_path):
    gt = [{'description': 'cat', 'bbox': [100, 200, 400, 600]}]
    left = _run(tmp_path / 'left', gt=gt, pred=[{'description': 'cat', 'bbox': [20, 20, 80, 60]}])
    right = _run(tmp_path / 'right', image_path=left / 'image.png', gt=gt,
                 pred=[{'description': 'cat', 'bbox': [20, 20, 80, 60]}])
    result = render_prediction_comparison(left, right, tmp_path / 'out',
                                         focus_gt_indices=[0], left_focus_pred_indices=[0])
    item = _manifest(result)
    assert item['view']['right']['focus_active'] is True
    assert item['view']['right']['selected_pred_indices'] == []
    assert item['view']['right']['comparison_gt_visibility']['revealed_matched_gt_indices'] == [0]
    assert 'Focused GT boxes are shown' in result.summary
    assert 'matched predictions and focused matched GT' in result.summary
    assert 'Green boxes are prediction boxes only' not in result.summary
    png = Image.open(result.image_paths[0])
    # On the right, the selected GT edge is opaque despite no prediction focus.
    assert png.getpixel((1009, 300)) == (0, 190, 90)


def test_invalid_endpoints_are_directed_glyph_and_preserved(tmp_path, monkeypatch):
    run = _run(tmp_path / 'run', pred=[{'description': 'bad', 'bbox': [20, 20, 80, 60]}])
    artifacts = load_visual_rows(run)
    row = artifacts.rows[0]
    invalid = replace(row.pred[0], bbox_pixel_xyxy=(80, 60, 20, 20),
                      source_bbox=(80, 60, 20, 20), geometry_valid=False)
    monkeypatch.setattr('src.vis.api.load_visual_rows',
                        lambda *args, **kwargs: replace(artifacts, rows=(replace(row, pred=(invalid,)),)))
    result = render_gt_vs_prediction(run, tmp_path / 'out', focus_pred_indices=[0])
    item = _manifest(result)
    assert item['pred_objects'][0]['bbox_pixel_xyxy'] == [80, 60, 20, 20]
    assert item['pred_objects'][0]['geometry_valid'] is False
    assert item['match']['tp'] == 0
    assert item['match']['fp'] == 0
    assert item['match']['invalid_pred_indices'] == [0]
    png = Image.open(result.image_paths[0])
    # Directed connector crosses its midpoint; a repaired rectangle would leave it empty.
    assert png.getpixel((1137, 284)) == (235, 45, 45)
    # The would-be repaired top right corner has no rectangle edge.
    assert png.getpixel((1264, 199)) == (100, 100, 100)


def test_sparse_original_prediction_index_is_the_focus_identity(tmp_path, monkeypatch):
    run = _run(tmp_path / 'run', gt=[{'description': 'cat', 'bbox': [100, 200, 400, 600]}],
               pred=[{'description': 'cat', 'bbox': [20, 20, 80, 60]}])
    artifacts = load_visual_rows(run)
    row = artifacts.rows[0]
    monkeypatch.setattr('src.vis.api.load_visual_rows',
                        lambda *args, **kwargs: replace(artifacts, rows=(replace(row, pred=(replace(row.pred[0], index=7),)),)))
    item = _manifest(render_gt_vs_prediction(run, tmp_path / 'out', focus_pred_indices=[7]))
    assert item['view']['prediction']['selected_pred_indices'] == [7]
    assert item['pred_objects'][0]['index'] == 7
    assert item['match']['matched_pairs'][0]['pred_index'] == 7


def test_selected_coincident_ids_have_distinct_visible_label_boxes(tmp_path, monkeypatch):
    run = _run(tmp_path / 'run', gt=[{'description': 'cat', 'bbox': [100, 200, 400, 600]}],
               pred=[{'description': 'cat', 'bbox': [20, 20, 80, 60]},
                     {'description': 'cat', 'bbox': [20, 20, 80, 60]}])
    artifacts = load_visual_rows(run)
    row = artifacts.rows[0]
    invalid = replace(row.pred[0], index=46, bbox_pixel_xyxy=(80, 60, 20, 20),
                      source_bbox=(80, 60, 20, 20), geometry_valid=False)
    row = replace(row, pred=(*row.pred, invalid, replace(invalid, index=48)))
    monkeypatch.setattr('src.vis.api.load_visual_rows',
                        lambda *args, **kwargs: replace(artifacts, rows=(row,)))
    labels_by_image = {}
    draw_text = ImageDraw.ImageDraw.text
    def record_text(draw, xy, text, *args, **kwargs):
        if text.startswith(('G0 ', 'P0->', 'FP P1 ', 'INVALID P46 ', 'INVALID P48 ')):
            bbox = draw.textbbox(xy, text, font=kwargs['font'])
            labels_by_image.setdefault(id(draw._image), []).append((text, tuple(bbox)))
        return draw_text(draw, xy, text, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, 'text', record_text)
    result = render_prediction_comparison(run, run, tmp_path / 'out', focus_gt_indices=[0],
                                         left_focus_pred_indices=[0, 1, 46, 48],
                                         right_focus_pred_indices=[0, 1, 46, 48])
    views = _manifest(result)['view']
    png = Image.open(result.image_paths[0])
    assert len(labels_by_image) == 2
    for panel, labels in zip(('left', 'right'), labels_by_image.values(), strict=True):
        assert len(labels) == 5
        vx, vy, _, _ = views[panel]['crop_transform']['viewport_canvas_xyxy']
        for index, (_, box) in enumerate(labels):
            for _, other in labels[index + 1:]:
                # Include the two-pixel label padding, not only glyph bounds.
                assert box[2] + 4 < other[0] or other[2] + 4 < box[0] or box[3] + 4 < other[1] or other[3] + 4 < box[1]
            glyph_pixels = png.crop(tuple(round(value + (vx if axis % 2 == 0 else vy))
                                         for axis, value in enumerate(box)))
            assert (0, 0, 0) in glyph_pixels.getdata()


def test_rollout_source_names_and_coordinate_summary_are_explicit(tmp_path, monkeypatch):
    run = _run(tmp_path / 'run')
    artifacts = load_visual_rows(run)
    monkeypatch.setattr('src.vis.api.load_visual_rows', lambda *args, **kwargs: artifacts)
    single = render_gt_vs_prediction(run, tmp_path / 'single', input_format='rollout', labels_json='labels.json')
    comparison = render_prediction_comparison(run, run, tmp_path / 'comparison', input_format='rollout', labels_json='labels.json')
    for result, expected in ((single, {'rollout_jsonl'}), (comparison, {'left_rollout_jsonl', 'right_rollout_jsonl'})):
        manifest = json.loads(result.manifest_path.read_text())
        assert expected <= manifest['inputs'].keys()
        assert not any('scored_jsonl' in key for key in manifest['inputs'])
        assert 'native rollout prediction boxes are converted from norm1000 bins to pixels' in result.summary
        assert 'prediction boxes are already pixels' not in result.summary
        assert manifest['coordinate_surfaces']['pred_bbox'].startswith('native rollout norm1000')
