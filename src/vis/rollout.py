"""Saved native rule-stability rollout adapter; no model or tokenizer calls."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image

from src.common.errors import ArtifactContractError
from src.inference.parsing import parse_compact_object_box_closed
from src.vis.normalization import ArtifactRows, VisualObject, VisualRow, _gt_object, _read_jsonl, _rows_by_id
from src.eval.detection_categories import normalize_coco_category_name

FORMAT = 'rule-stability-native-rollout-v1'
PROVENANCE = 'visualization_provenance'
_IDENTITY_KEYS = ('producer', 'request_id', 'image_id', 'token_ids', 'text', 'prompt_token_ids', 'media_sha256', 'image_grid_thw', 'stop_reason')
_CAP_REASONS = frozenset(('length', 'max_new_tokens', 'max_tokens'))
MATCHING_SEMANTICS = (
    'Visualization uses class-aware greedy matching at the configured pixel IoU threshold (default >= 0.5) '
    'and same-class duplicate hints at the configured pixel IoU threshold (default >= 0.3). '
    'This differs from saved research class-agnostic cardinality-first norm1000 matching with category credit, '
    'and chronological class-agnostic duplicate events at norm1000 IoU > 0.9. Invalid geometry is excluded from visualization matching and valid FP counts.'
)


def _require(condition: bool, message: str, code: str = 'vis.rollout_field', **context: Any) -> None:
    if not condition:
        raise ArtifactContractError(message, code=code, context=context)


def _finite_json(value: Any, *, field: str) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), 'native rollout metadata contains a nonfinite number', 'vis.rollout_nonfinite', field=field)
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, field=f'{field}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, field=f'{field}[{index}]')


def _json_file(path: Path) -> dict | list:
    _require(path.is_file(), 'bound native visualization input is missing', 'vis.rollout_missing_file', path=str(path))
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactContractError('bound native visualization JSON is malformed', code='vis.rollout_json', context={'path':str(path)}, cause=exc) from exc
    _finite_json(value, field=str(path))
    _require(isinstance(value, (dict, list)), 'native visualization JSON must be an object or list', path=str(path))
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_native_record(record: dict) -> None:
    """Validate only the declared full-image native record contract, not generated text quality."""
    _finite_json(record, field='record')
    for field in ('image_id', 'width', 'height'):
        _require(_integer(record.get(field)) and (field == 'image_id' or record[field] > 0), 'native rollout requires integer image identity/dimensions', field=field)
    for field in ('request_id', 'image_path', 'stop_reason'):
        _require(isinstance(record.get(field), str) and bool(record[field].strip()), 'native rollout requires a nonempty string', field=field)
    _require(Path(record['image_path']).is_absolute(), 'native rollout image_path must be absolute', field='image_path')
    _require(isinstance(record.get('text'), str), 'native rollout requires original generated text', field='text')
    for field in ('image_sha256', 'media_sha256', 'raw_identity'):
        _require(_sha(record.get(field)), 'native rollout requires a SHA256 identity', field=field)
    for field in ('token_ids', 'prompt_token_ids', 'image_grid_thw'):
        value = record.get(field)
        _require(isinstance(value, list) and all(_integer(x) for x in value), 'native rollout requires original integer token/media arrays', field=field)
    _require(len(record['image_grid_thw']) == 3, 'native rollout requires one original image grid', field='image_grid_thw')
    _require(_integer(record.get('generated_tokens')) and record['generated_tokens'] == len(record['token_ids']), 'native rollout generated token count differs', field='generated_tokens')
    for field in ('raw_logprobs', 'policy_logprobs'):
        value = record.get(field)
        _require(isinstance(value, list) and len(value) == len(record['token_ids']) and
                 all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value), 'native rollout requires aligned original log probabilities', field=field)
    producer = record.get('producer')
    _require(isinstance(producer, dict) and producer.get('kind') == 'rule_stability' and producer.get('engine') == 'native',
             'unsupported rollout schema: expected declared rule_stability native records', 'vis.rollout_unsupported_schema')
    crop = record.get('crop')
    _require(isinstance(crop, list) and all(_integer(value) for value in crop) and
             crop == [0, 0, record['width'], record['height']] and
             isinstance(record.get('view_scale'), (int, float)) and not isinstance(record['view_scale'], bool) and record['view_scale'] == 1,
             'native visualization supports only a full-image crop at view_scale 1', 'vis.rollout_view')
    identity = json.dumps({k:record[k] for k in _IDENTITY_KEYS}, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    _require(hashlib.sha256(identity.encode()).hexdigest() == record['raw_identity'], 'native rollout original identity differs', 'vis.rollout_raw_identity')


def _bound_source(binding: Any, *, base_dir: Path, field: str) -> tuple[Path, dict | list]:
    _require(isinstance(binding, dict) and isinstance(binding.get('path'), str) and bool(binding['path']) and _sha(binding.get('sha256')),
             'saved native analysis requires explicit path and SHA256 bindings', 'vis.rollout_binding', field=field)
    path = Path(binding['path'])
    if not path.is_absolute():
        path = base_dir / path
    value = _json_file(path)
    _require(sha256_file(path) == binding['sha256'], 'saved native source bytes differ from binding', 'vis.rollout_binding_sha', field=field, path=str(path))
    return path, value


def _parsed_rows(record: dict):
    parsed = parse_compact_object_box_closed(record['text'], row_id=record['request_id'], row_index=0,
                                             image_width=record['width'], image_height=record['height'])
    complete = [dict(row, valid=True, order=row['generated_order'], bbox=row['coord_bins']) for row in parsed.predictions]
    malformed = []
    for drop in parsed.dropped_predictions:
        bins = []
        if drop['reason'] == 'geometry_invalid':
            for span in drop['coord_token_spans']:
                match = re.fullmatch(r'<\|coord_(\d+)\|>', span['text'])
                if match is None:
                    break
                bins.append(int(match.group(1)))
        if len(bins) == 4:
            description = drop['raw_text'].split('<|object_ref_start|>', 1)[-1].split('<|object_ref_end|>', 1)[0]
            complete.append(dict(drop, valid=False, order=drop['generated_order'], bbox=bins, description=description))
        else:
            censored = record['stop_reason'] in _CAP_REASONS and drop['char_end'] == len(record['text']) and not drop['raw_text'].endswith('<|box_end|>')
            malformed.append(dict(drop, censored=censored))
    return parsed.parse_status, sorted(complete, key=lambda row:row['order']), malformed


def _span_consistent(row: dict, text: str) -> bool:
    a, z = row.get('char_start'), row.get('char_end')
    raw = row.get('raw_span_text', row.get('raw_text'))
    return (_integer(a) and _integer(z) and a <= z <= len(text) and isinstance(raw, str) and text[a:z] == raw and
            ('raw_span_sha256' not in row or row['raw_span_sha256'] == hashlib.sha256(raw.encode()).hexdigest()))


def _saved_rows(record: dict, *, base_dir: Path, expected: list[dict], malformed: list[dict]) -> tuple[list[dict], dict | None]:
    provenance = record.get(PROVENANCE)
    if provenance is None:
        return expected, None
    _require(isinstance(provenance, dict) and provenance.get('format') == FORMAT and provenance.get('coordinate_space') == 'norm1000',
             'native visualization provenance must declare norm1000 rule-stability records', 'vis.rollout_provenance')
    _, raw = _bound_source(provenance.get('raw_source'), base_dir=base_dir, field='raw_source')
    _require(raw == {k:v for k,v in record.items() if k != PROVENANCE}, 'native JSONL row differs from bound original raw record', 'vis.rollout_binding_record')
    if 'analysis_source' not in provenance:
        return expected, None
    _, analysis = _bound_source(provenance['analysis_source'], base_dir=base_dir, field='analysis_source')
    _require(isinstance(analysis, dict) and isinstance(analysis.get('rows'), list) and isinstance(analysis.get('malformed'), list),
             'bound native analysis has no row and malformed ledger', 'vis.rollout_analysis')
    rows = analysis['rows']
    _require(len(rows) == len(expected), 'saved native analysis omitted or added complete rows', 'vis.rollout_analysis_rows')
    for saved, parsed in zip(rows, expected, strict=True):
        _require(isinstance(saved, dict) and _span_consistent(saved, record['text']) and
                 all(saved.get(key) == parsed.get(key) for key in ('generated_order', 'order', 'description', 'bbox', 'valid', 'char_start', 'char_end')),
                 'saved native analysis row differs from original parser span', 'vis.rollout_analysis_span')
        positions = saved.get('positions')
        _require(isinstance(positions, list) and positions and all(_integer(p) and p < len(record['token_ids']) for p in positions) and
                 positions == list(range(positions[0], positions[-1]+1)) and saved.get('completion_position') == positions[-1],
                 'saved native analysis original action positions are invalid', 'vis.rollout_analysis_positions')
        coord = saved.get('coordinate_positions')
        _require(isinstance(coord, list) and all(_integer(p) and p in positions for p in coord) and
                 (not coord or (len(coord) == 4 and coord == list(range(coord[0], coord[0]+4)))),
                 'saved native analysis coordinate positions are invalid', 'vis.rollout_analysis_positions')
        for field in ('schema_spans', 'coord_token_spans'):
            _require(isinstance(saved.get(field), list) and all(isinstance(s, dict) and _span_consistent(dict(s, raw_text=s.get('text')), record['text']) for s in saved[field]),
                     'saved native analysis marker spans differ from original text', 'vis.rollout_analysis_span', field=field)
    saved_malformed = analysis['malformed']
    _require(len(saved_malformed) == len(malformed), 'saved native analysis malformed ledger differs', 'vis.rollout_analysis_span')
    for saved, parsed in zip(saved_malformed, malformed, strict=True):
        _require(isinstance(saved, dict) and _span_consistent(saved, record['text']) and
                 all(saved.get(k) == parsed.get(k) for k in ('generated_order', 'char_start', 'char_end', 'reason', 'censored')),
                 'saved native analysis malformed span differs', 'vis.rollout_analysis_span')
    return rows, analysis


def _prediction(row: dict, record: dict) -> VisualObject:
    bins = row['bbox']
    _require(isinstance(bins, list) and len(bins) == 4 and all(_integer(b) for b in bins), 'native complete box requires four integer norm1000 endpoints', 'vis.rollout_bbox')
    pixel = tuple(float(round(value * (record['width'] if axis % 2 == 0 else record['height']) / 1000)) for axis, value in enumerate(bins))
    valid = row['valid'] and pixel[0] < pixel[2] and pixel[1] < pixel[3]
    metadata = dict(row, prediction_id=f"{record['request_id']}:p{row['generated_order']}", raw_parser_order=row['generated_order'])
    if row['valid'] and not valid:
        metadata['geometry_diagnostic'] = 'pixel_rounding_collapsed'
    return VisualObject(index=row['generated_order'], description=row['description'], normalized_description=normalize_coco_category_name(row['description']),
                        bbox_pixel_xyxy=pixel, source_bbox=tuple(bins), source_coord_space='norm1000', source_coord_bins=tuple(bins),
                        geometry_valid=valid, source_metadata=metadata)


def load_rollout_rows(path: str | Path, *, labels_json: str | Path | None) -> ArtifactRows:
    path = Path(path).resolve()
    _require(path.suffix == '.jsonl' and path.is_file(), 'explicit rollout input must be an existing JSONL file', 'vis.rollout_input', path=str(path))
    _require(labels_json is not None, 'explicit rollout visualization requires --labels-json full labels', 'vis.rollout_labels_required')
    labels_path = Path(labels_json).resolve()
    labels = _json_file(labels_path)
    _require(isinstance(labels, list) and labels and all(isinstance(image, dict) and _integer(image.get('image_id')) for image in labels), 'full labels must be an image list with integer IDs', 'vis.rollout_labels')
    by_id = {image['image_id']:image for image in labels}
    _require(len(by_id) == len(labels), 'full labels contain duplicate image IDs', 'vis.rollout_labels')
    labels_binding = dict(path=str(labels_path), sha256=sha256_file(labels_path))
    rows = []
    checked_images = set()
    for index, record in enumerate(_read_jsonl(path)):
        # Identity/dimension join precedes view checks so a mismatched image never changes its coordinate frame.
        image = by_id.get(record.get('image_id')) if _integer(record.get('image_id')) else None
        _require(image is not None and all(record.get(k) == image.get(k) for k in ('image_path', 'image_sha256', 'width', 'height')),
                 'native rollout image identity/dimensions differ from explicit labels', 'vis.rollout_image_join', row_index=index, image_id=record.get('image_id'))
        validate_native_record(record)
        source_path = Path(record['image_path'])
        if record['image_id'] not in checked_images:
            _require(source_path.is_file() and sha256_file(source_path) == record['image_sha256'], 'native rollout image bytes differ', 'vis.rollout_image_sha', path=str(source_path))
            try:
                with Image.open(source_path) as source:
                    _require(source.size == (record['width'], record['height']), 'native rollout dimensions differ from image bytes', 'vis.rollout_image_size')
            except OSError as exc:
                raise ArtifactContractError('native rollout source image cannot be read', code='vis.rollout_image_size', cause=exc) from exc
            checked_images.add(record['image_id'])
        objects = image.get('objects')
        _require(isinstance(objects, list) and all(isinstance(o, dict) and isinstance(o.get('desc'), str) and 'coco_ann_id' in o for o in objects), 'full labels require explicit object category and owner IDs', 'vis.rollout_labels')
        owner_ids = [str(o['coco_ann_id']) for o in objects]
        _require(len(set(owner_ids)) == len(owner_ids), 'full labels contain duplicate owner IDs', 'vis.rollout_labels')
        gt = tuple(_gt_object(dict(obj, owner_id=str(obj['coco_ann_id'])), index=n, row_id=str(record['image_id']), image_width=record['width'], image_height=record['height']) for n,obj in enumerate(objects))
        status, complete, malformed = _parsed_rows(record)
        complete, analysis = _saved_rows(record, base_dir=path.parent, expected=complete, malformed=malformed)
        metadata = dict(input_format='rollout', request_id=record['request_id'], raw_identity=record['raw_identity'], stop_reason=record['stop_reason'],
                        generated_tokens=record['generated_tokens'], parser_status=status, malformed_outputs=malformed,
                        original_input_bounds=dict(crop=record['crop'], view_scale=record['view_scale'], width=record['width'], height=record['height']),
                        provenance=record.get(PROVENANCE), labels_source=labels_binding, matching_semantics=MATCHING_SEMANTICS,
                        token_positions_source='bound saved analysis' if analysis is not None else 'absent; no tokenizer or invented offsets')
        if analysis is not None:
            metadata['original_research_diagnostics'] = {k:analysis[k] for k in ('duplicate_events', 'event_positions', 'burdens', 'parser_status', 'action_family_burdens') if k in analysis}
        rows.append(VisualRow(row_id=str(record['image_id']), row_index=index, image_path=source_path, source_image_path=record['image_path'],
                              image_width=record['width'], image_height=record['height'], gt=gt, pred=tuple(_prediction(row, record) for row in complete), source_metadata=metadata))
    _rows_by_id(tuple(rows))
    return ArtifactRows(artifact_dir=path.parent, scored_jsonl=path, raw_jsonl=path, rows=tuple(rows))
