#!/usr/bin/env python3
"""Build and validate all 45 corrected spatial inputs before model calls."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from PIL import Image

WORKTREE = Path('/data/CoordExp/.worktrees/research-probes')
ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source')
OUT = ROOT / 'final/corrected-pilot-v2'
PANEL = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json')
READINESS = ROOT / 'generic-entry-readiness.json'
SNAPSHOT = OUT / 'source-snapshot.json'
CELL_KEYS = {'00','10-','10+','01-','01+','11-','11+'}
COORD_BASE = 151670
COORD_LIMIT = COORD_BASE + 1000
OBJ_START = 151646

import sys
sys.path.insert(0, str(WORKTREE))
from probes.training_set_completion.recurrence_spatial import prepare_shared
from probes.training_set_completion.recurrence_spatial.state_entry import digest_json


def bind(path: Path) -> dict:
    path = path.resolve(strict=True)
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def load(path: Path):
    return json.loads(path.read_text())


def starts(tokens):
    return [i for i, token in enumerate(tokens) if token == OBJ_START]


def source_row_geometry(tokens, width, height):
    invalid = 0
    rows = 0
    for i, start in enumerate(starts(tokens)):
        stop = starts(tokens)[i+1] if i+1 < len(starts(tokens)) else len(tokens)
        coords = [token-COORD_BASE for token in tokens[start:stop] if COORD_BASE <= token < COORD_LIMIT]
        if len(coords) != 4:
            raise AssertionError(f'raw source row {i} has {len(coords)} coordinates')
        rows += 1
        invalid += int(not (coords[0] < coords[2] and coords[1] < coords[3]))
    return rows, invalid


def main():
    snapshot = load(SNAPSHOT)
    if not snapshot.get('captured_before_model_calls') or snapshot.get('model_calls_before_snapshot') != 0:
        raise AssertionError('source snapshot is not pre-model')
    panel_binding = bind(PANEL)
    if panel_binding != snapshot['source_files']['panel']:
        raise AssertionError('panel changed after immutable snapshot')
    readiness = load(READINESS)
    if readiness.get('status') != 'ready' or readiness.get('declared_state_count') != 45 or readiness.get('resolved_state_count') != 45:
        raise AssertionError('all45 readiness is not ready')
    if bind(READINESS) != snapshot['source_files']['readiness']:
        raise AssertionError('readiness changed after immutable snapshot')
    states = readiness['states']
    shared = panel_binding
    readiness_binding = bind(READINESS)
    snapshot_binding = bind(SNAPSHOT)
    manifests_dir = OUT / 'inputs/manifests'
    image_root = OUT / 'inputs/images'
    entries = []
    errors = []
    counts = {
        'states': 0, 'manifests': 0, 'cells': 0, 'images': 0,
        'history_rows': 0, 'invalid_source_rows': 0,
        'noncoordinate_drift': 0, 'y_coordinate_changes': 0,
        'image_crop_mismatches': 0, 'source_identity_mismatches': 0,
        'cell_shape_errors': 0,
    }
    image_seen = set()
    for state in states:
        state_id = state['id']
        manifest_path = manifests_dir / f'{state_id}.json'
        try:
            manifest = prepare_shared._manifest_for_state(
                state,
                shared_panel_binding=shared,
                readiness_binding=readiness_binding,
                snapshot_binding=snapshot_binding,
                manifest_path=manifest_path,
                image_root=image_root / state_id,
            )
            counts['states'] += 1
            counts['manifests'] += 1
            if set(manifest['cells']) != CELL_KEYS:
                raise AssertionError('cell set drift')
            prefix = [int(x) for x in load(Path(state['source']['raw']['path']))['rows'][int(state['source']['batch_index'])]['token_ids'][:int(state['prefix']['source_row_end'])]]
            source_image = Path(state['source']['image']['path']).resolve(strict=True)
            with Image.open(source_image) as original:
                original = original.convert('RGB')
                source_width, source_height = original.size
                for key, cell in manifest['cells'].items():
                    counts['cells'] += 1
                    history = [int(x) for x in cell['history']]
                    if len(history) != len(prefix):
                        raise AssertionError(f'{key}: history token length drift')
                    noncoord = sum(a != b and not (COORD_BASE <= a < COORD_LIMIT) for a,b in zip(prefix, history))
                    counts['noncoordinate_drift'] += noncoord
                    if noncoord:
                        raise AssertionError(f'{key}: noncoordinate history drift {noncoord}')
                    boxes = cell['history_boxes']
                    source_rows, invalid = source_row_geometry(prefix, source_width, source_height)
                    if len(boxes) != source_rows:
                        raise AssertionError(f'{key}: history row count drift')
                    counts['history_rows'] += len(boxes)
                    counts['invalid_source_rows'] += invalid
                    for box in boxes:
                        mapped = box['mapped_bins']; source = box['source_bins']
                        if mapped[1] != source[1] or mapped[3] != source[3]:
                            counts['y_coordinate_changes'] += 1
                            raise AssertionError(f'{key}: horizontal transform changed y')
                        if any(value < 0 or value > 999 for value in mapped):
                            raise AssertionError(f'{key}: mapped coordinate out of range')
                    image_path = Path(cell['image_path']).resolve(strict=True)
                    image_seen.add(str(image_path))
                    with Image.open(image_path) as transformed:
                        transformed = transformed.convert('RGB')
                        if transformed.size != (source_width + 256, source_height):
                            raise AssertionError(f'{key}: canvas dimensions drift')
                        offset = int(cell['visual_offset_px'])
                        if transformed.crop((offset, 0, offset + source_width, source_height)).tobytes() != original.tobytes():
                            counts['image_crop_mismatches'] += 1
                            raise AssertionError(f'{key}: source crop mismatch')
                    expected_sha = bind(image_path)
                    if expected_sha != cell['image']:
                        counts['source_identity_mismatches'] += 1
                        raise AssertionError(f'{key}: image binding drift')
            counts['images'] = len(image_seen)
            entries.append({'id': state_id, 'model': state['model'], 'kind': state.get('kind'), 'manifest': bind(manifest_path), 'cell_count': 7, 'cell_keys': sorted(CELL_KEYS)})
        except Exception as exc:
            errors.append({'id': state_id, 'error': repr(exc)})
    receipt = {
        'schema': 'recurrence_spatial_source.corrected_pilot_v2_input_construction.v1',
        'unit_id': '2026-09-19-recurrence-spatial-source',
        'attempt_id': 'corrected-pilot-v2',
        'status': 'ready' if not errors and counts['states'] == 45 and counts['cells'] == 315 else 'failed',
        'model_calls': 0,
        'gpu_forwards': 0,
        'panel': panel_binding,
        'readiness': readiness_binding,
        'source_snapshot': snapshot_binding,
        'declared_states': 45,
        'resolved_states': len(entries),
        'constructed_cells': counts['cells'],
        'pilot_state_ids': ['tied-417044-failure','untied-417044-failure'],
        'pilot_selection_rule': 'original first pilot boundary from tied/untied 417044 failure records; no outcome selection',
        'counts': counts,
        'errors': errors,
        'state_entries': entries,
    }
    path = OUT / 'input-construction-receipt.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    index = {'schema':'recurrence_spatial_source.corrected_pilot_v2_input_index.v1','unit_id':receipt['unit_id'],'attempt_id':receipt['attempt_id'],'panel':panel_binding,'source_snapshot':snapshot_binding,'declared_states':45,'states':entries,'cell_count':counts['cells']}
    (OUT/'input-execution-index.json').write_text(json.dumps(index, indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'states':counts['states'],'cells':counts['cells'],'images':counts['images'],'history_rows':counts['history_rows'],'invalid_source_rows':counts['invalid_source_rows'],'errors':len(errors)},indent=2))
    if receipt['status'] != 'ready':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
