"""Root-authorized four-case C/D review after both consumers were accepted."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path('/data/CoordExp/.worktrees/research-probes')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
OUT = ROOT / '2026-09-11-positive-progress-matched-control/visual-review-v2'
IDS = ['25274', '511251', '417044', '477415']
SOURCES = {
    'C': (ROOT / '2026-09-11-margin-preserved-positive-branch/endpoint-C/consumer.json',
          'c0f1f73c107956051c821b1d038a7434ea76a958c6c8a13a3716dde385c0533e',
          'margin_preserved_endpoint.natural.v1'),
    'D': (ROOT / '2026-09-11-positive-progress-matched-control/endpoint-D/consumer.json',
          'c25ce0157a6852a6f533d5544b05f900ea73f7af637aa2109711f0061e91fd93',
          'positive_progress_matched_endpoint.natural.v1'),
}
HELPER = REPO / ('research/investigations/qwen3-vl-dense-enumeration/experiments/'
                 '2026-09-11-margin-preserved-positive-branch/render_endpoint_comparison.py')
HELPER_SHA = '4998aed07e7e781b84c87c17d3d2f03bff1b65f7b9ae3437c9f319f734fe43b0'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert not OUT.exists(), 'Use the accepted existing outputs; never overwrite/rerender.'
    assert sha(HELPER) == HELPER_SHA
    spec = importlib.util.spec_from_file_location('accepted_visual_adapter', HELPER)
    assert spec and spec.loader
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    selected = {}
    for arm, (path, digest, schema) in SOURCES.items():
        assert sha(path) == digest
        rows = json.loads(path.read_text())
        assert len(rows) == 384 and all(r['arm'] == arm and r['schema'] == schema for r in rows)
        indexed = {str(r['image_id']): r for r in rows}
        selected[arm] = [indexed[i] for i in IDS]
    for c, d in zip(selected['C'], selected['D']):
        for key in ['row_id', 'image_path', 'image_width', 'image_height', 'gt']:
            assert c['parsed'][key] == d['parsed'][key], key
    row_ids = tuple(r['parsed']['row_id'] for r in selected['C'])
    provenance = {}
    for arm, rows in selected.items():
        raw, scored, info = [], [], []
        for row in rows:
            a, b, c = helper._adapter_rows(row, source_consumer_sha256=SOURCES[arm][1], arm=arm)
            raw.append(a)
            scored.append(b)
            info.append(c)
        adapter = OUT / f'adapter-{arm}'
        helper.write_immutable_jsonl(adapter / 'gt_vs_pred.jsonl', raw)
        helper.write_immutable_jsonl(adapter / 'gt_vs_pred_scored.jsonl', scored)
        provenance[arm] = {'consumer': str(SOURCES[arm][0]), 'sha256': SOURCES[arm][1],
                           'rows': info, 'adapter': helper._validate_adapter(adapter, row_ids)}
    from src.vis import render_prediction_comparison
    result = render_prediction_comparison(OUT / 'adapter-C', OUT / 'adapter-D', OUT / 'images',
                                         left_label='C: margin32', right_label='D: no-margin17',
                                         row_ids=row_ids, duplicate_iou_threshold=0.95)
    assert len(result.image_paths) == 4 and all(p.is_file() for p in result.image_paths)
    receipt = {'schema': 'positive_progress_matched_control.visual_render.v1',
               'status': 'rendered_pending_root_view', 'selection': 'root-selected two D caps plus two positive tradeoffs',
               'image_ids': IDS, 'row_ids': row_ids, 'sources': provenance,
               'producer': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__))},
               'adapter_helper': {'path': str(HELPER), 'sha256': HELPER_SHA},
               'manifest': {'path': str(result.manifest_path), 'sha256': sha(result.manifest_path)},
               'pngs': [{'path': str(p), 'sha256': sha(p)} for p in result.image_paths],
               'gt_edited': False, 'predictions_relabelled': False,
               'dup_hint_is_pair_count_not_strict_later_row_count': True}
    helper.write_immutable_json(OUT / 'render-receipt.json', receipt)
    print(json.dumps({'status': 'rendered', 'receipt': str(OUT / 'render-receipt.json'),
                      'pngs': receipt['pngs']}))


if __name__ == '__main__':
    main()
