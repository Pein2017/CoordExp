"""Build the explicit synthetic, complete teacher bank from fixed reviewed owners.

This producer only prepares literal routes.  It does not treat edited suffixes as
native model observations and never launches a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from probes.training_set_completion.training import coordinate_token_table, validate_route

B = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum')
ROOT = B / 'third-complete-bank-preparation-v1'
PREP = B / 'first-fit-preparation-v1/manifest.json'
READBACK = B / 'first-fit-v1/readback-step-16.json'
DECISIONS = B / 'first-fit-review-extraction-v1/decisions.jsonl'
CATALOG = B / 'target-owners-complete-v4.json'
ADMISSIONS = B / 'second-fit-root-rulings-v1/admissions.json'
RULINGS = B / 'second-fit-root-rulings-v1/rulings.json'
ADDITIONS = B / 'second-fit-root-rulings-v1/parent-v4-additions.json'
STAGE1 = B / 'stage01-review-extraction-v2/proposal-rows'
FIRST_ADMISSIONS = B / 'first-fit-new-owner-admissions-v1/admissions.json'
IMAGE_IDS = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
EOS = 151645
SCHEMA = 'training_set_completion.complete_synthetic_teacher_bank.v1'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def fh(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': fh(path), 'size_bytes': path.stat().st_size}


def publish(path: Path, value: Any) -> None:
    require(not path.exists(), f'collision: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(value)
    with path.open('xb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    require(path.read_bytes() == content, 'publication readback')


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def decisions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _field_ids(tokenizer: Any, description: str, coords: list[int], coordinate_ids: list[int]) -> list[int]:
    description_ids = tokenizer.encode(description, add_special_tokens=False)
    require(description_ids and tokenizer.decode(description_ids, skip_special_tokens=False) == description, 'literal description tokenization')
    special = [tokenizer.convert_tokens_to_ids(token) for token in ('<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>')]
    require(all(type(token) is int and token >= 0 for token in special), 'schema tokens')
    return [special[0], *description_ids, special[1], special[2], *(coordinate_ids[value] for value in coords), special[3]]


def _eligible_source(row: Mapping[str, Any]) -> bool:
    return row.get('step') == 16 and row.get('identity_state') == 'fixed_target' and isinstance(row.get('owner_id'), str) and isinstance(row.get('generated_order'), int)


def _description(record: Mapping[str, Any], source: Mapping[str, Any] | None, admission: Mapping[str, Any] | None) -> tuple[str, bool, str]:
    # A current root mask ruling wins over every legacy/source-class value.
    if admission and admission.get('class_policy') == 'mask_description':
        observed = admission.get('observed_description') or record.get('legacy_observed_description') or (source or {}).get('raw', {}).get('description')
        require(isinstance(observed, str) and observed, 'unknown class requires observed description literal')
        return observed, False, 'root_unknown_observed_literal_masked'
    category = record.get('category')
    if isinstance(category, str) and category:
        return category, True, 'catalog_category'
    if source and source.get('class') == 'verified' and isinstance(source.get('raw', {}).get('description'), str):
        return source['raw']['description'], True, 'accepted_source_verified_class'
    if isinstance(record.get('legacy_verified_description'), str) and record['legacy_verified_description']:
        return record['legacy_verified_description'], True, 'accepted_source_verified_class'
    observed = (admission or {}).get('observed_description') or record.get('legacy_observed_description') or (source or {}).get('raw', {}).get('description')
    require(isinstance(observed, str) and observed, 'unknown class requires observed description literal')
    return observed, False, 'unknown_class_observed_literal_masked'


def _select(records: list[Mapping[str, Any]], source: list[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any] | None, bool]]:
    by_owner = {str(record['owner_id']): record for record in records}
    require(len(by_owner) == len(records), 'duplicate target owner')
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any] | None, bool]] = []
    seen: set[str] = set()
    for row in sorted(source, key=lambda item: int(item['generated_order'])):
        owner = str(row['owner_id'])
        if owner in by_owner and owner not in seen:
            selected.append((by_owner[owner], row, False))
            seen.add(owner)
    for owner in sorted(set(by_owner) - seen):
        selected.append((by_owner[owner], None, True))
    require(len(selected) == len(records) and {str(record['owner_id']) for record, _, _ in selected} == set(by_owner), 'complete coverage')
    return selected


def _stage1_observed() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    by_proposal: dict[str, str] = {}
    by_owner: dict[str, str] = {}
    source_by_owner: dict[str, str] = {}
    verified_by_owner: dict[str, str] = {}
    for path in sorted(STAGE1.glob('*.jsonl')):
        for row in decisions(path):
            proposal_id = str(row.get('proposal_id'))
            description = row.get('description_predicted') or row.get('description')
            if not isinstance(description, str) or not description:
                continue
            by_proposal[proposal_id] = description
            owner = row.get('root_owner_id') or row.get('reviewed_owner_id')
            if isinstance(owner, str):
                by_owner.setdefault(owner, description)
                source_by_owner.setdefault(owner, proposal_id)
                if row.get('effective_class') == 'verified':
                    verified_by_owner.setdefault(owner, description)
    return by_proposal, by_owner, source_by_owner, verified_by_owner


def _source_description_indexes(rows: list[dict[str, Any]], first_admissions: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    verified: dict[str, str] = {}
    observed: dict[str, str] = {}
    source_by_owner: dict[str, str] = {}
    by_proposal = {str(row['proposal_id']): row for row in rows}
    for entry in first_admissions['entries']:
        source = by_proposal.get(str(entry['reference_proposal_id']))
        if source and isinstance(source.get('raw', {}).get('description'), str):
            observed[str(entry['owner_id'])] = source['raw']['description']
            source_by_owner[str(entry['owner_id'])] = str(source['proposal_id'])
    for row in rows:
        owner, description = row.get('owner_id'), row.get('raw', {}).get('description')
        if not isinstance(owner, str) or not isinstance(description, str) or not description:
            continue
        observed.setdefault(owner, description)
        source_by_owner.setdefault(owner, str(row['proposal_id']))
        if row.get('class') == 'verified':
            verified.setdefault(owner, description)
    return verified, observed, source_by_owner


def _parser_roundtrip(routes: list[dict[str, Any]], acquisition_path: Path, tokenizer: Any) -> dict[str, Any]:
    """Cold-reparse synthetic literal token output using the production parser."""
    from probes.source_rweak_row_cross.run import native_record

    acquisition = read(acquisition_path)
    goldens = {int(record['image_id']): record['golden'] for record in acquisition['records']}
    parsed_rows = 0
    for route in routes:
        text = tokenizer.decode(route['continuation_token_ids'], skip_special_tokens=False)
        parsed = native_record(text, route['case'], goldens[route['image_id']], 'im_end')
        require(len(parsed['pred']) == len(route['trusted_boxes']), 'native parser row count')
        require(not parsed.get('malformed', False), 'native parser malformed')
        for parsed_row, box, card in zip(parsed['pred'], route['trusted_boxes'], route['provenance']['trace']):
            require(parsed_row['coord_bins'] == box['expected_bins'] == card['edited_fields']['catalog_reference_coord_bins_1000'], 'native parser coordinate identity')
            require(parsed_row['description'] == card['edited_fields']['selected_description'], 'native parser description identity')
        parsed_rows += len(parsed['pred'])
    require(parsed_rows == 232, 'native parser total rows')
    return {'parser': 'src.inference.parsing.parse_compact_object_box_closed', 'images': len(routes), 'parsed_rows': parsed_rows, 'malformed_rows': 0, 'dropped_rows': 0}


def build(*, catalog_path: Path = CATALOG, output: Path = ROOT) -> dict[str, Any]:
    from probes.training_set_completion.route_bank import _load_tokenizer

    require(not (output / 'bank.json').exists(), 'bank collision')
    catalog = read(catalog_path)
    require(catalog.get('status') == 'lead-accepted' and len(catalog.get('records', [])) == 232, 'accepted v4 catalog')
    prep, readback, source_rows, additions = read(PREP), read(READBACK), decisions(DECISIONS), read(ADDITIONS)
    root_admissions = {str(item['owner_id']): item for item in read(ADMISSIONS)['admissions']}
    first_admissions = read(FIRST_ADMISSIONS)
    admissions = {**root_admissions, **{str(item['owner_id']): item for item in first_admissions['entries']}}
    tokenizer = _load_tokenizer(Path(prep['model_config']['model']['base_model']))
    coordinate_ids = coordinate_token_table(prep['model_config']['model']['base_model'])['ids']
    legacy_verified, legacy_observed, legacy_sources = _source_description_indexes(source_rows, first_admissions)
    stage1_by_proposal, stage1_by_owner, stage1_sources, stage1_verified = _stage1_observed()
    # Direct root-supplied stage01 evidence wins over first-fit text for these owners.
    preferred_stage1 = {
        'stable-new-219546-transparent-serving-jar': 'stage01:image-000000219546:sample-t0p3:p15',
        'stablenew:rear-steering-wheel': 'stage01:image-000000388795:sample-t0p7:p7',
        'stableNew:right-white-spoon': 'stage01:image-000000528944:greedy:p3',
    }

    records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for original in catalog['records']:
        require(int(original['image_id']) in IMAGE_IDS and isinstance(original.get('owner_id'), str) and str(original.get('reference_status', '')).startswith('qualified_'), 'catalog record')
        record = dict(original)
        owner = record['owner_id']
        preferred = preferred_stage1.get(owner)
        require(preferred is None or preferred in stage1_by_proposal, 'root stage01 source missing')
        record['legacy_verified_description'] = legacy_verified.get(owner) or stage1_verified.get(owner)
        record['legacy_observed_description'] = (stage1_by_proposal.get(preferred) if preferred else None) or legacy_observed.get(owner) or stage1_by_owner.get(owner) or stage1_by_proposal.get(str(record.get('reference_proposal_id')))
        record['legacy_observed_source_proposal_id'] = preferred or legacy_sources.get(owner) or stage1_sources.get(owner) or record.get('reference_proposal_id')
        bins = list(record['reference_coord_bins_1000'])
        require(len(bins) == 4 and 0 <= bins[0] < bins[2] < 1000 and 0 <= bins[1] < bins[3] < 1000, 'reasonable reference axes')
        records[int(record['image_id'])].append(record)
    require(set(records) == set(IMAGE_IDS) and sum(map(len, records.values())) == 232, 'catalog coverage')

    source_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        if _eligible_source(row):
            source_by_image[int(row['image_id'])].append(row)
    for match in additions['new_owner_raw_matches']:
        source = next(row for row in source_rows if row.get('step') == 16 and int(row['image_id']) == int(match['image_id']) and row['prediction_id'] == match['prediction_id'])
        source = dict(source)
        source['owner_id'], source['class'] = match['owner_id'], match['class']
        source_by_image[int(match['image_id'])].append(source)

    planned = {image: _select(records[image], source_by_image[image]) for image in IMAGE_IDS}
    gaps = []
    for image, selections in planned.items():
        for record, source, _ in selections:
            try:
                _description(record, source, admissions.get(str(record['owner_id'])))
            except ValueError as exc:
                gaps.append({'image_id': image, 'owner_id': record['owner_id'], 'error': str(exc)})
    require(not gaps, f'unresolved descriptions across fixed catalog: {gaps}')

    prep_routes = {int(route['image_id']): route for route in prep['routes']}
    readback_rows = {int(row['image_id']): row for row in readback['rows']}
    routes: list[dict[str, Any]] = []
    for image in IMAGE_IDS:
        parent = prep_routes[image]
        require(readback_rows[image]['prompt_token_ids'] == parent['prompt_token_ids'], 'first-fit prompt source')
        continuation: list[int] = []
        weights: list[int] = []
        boxes, trace = [], []
        for order, (record, source, appended) in enumerate(planned[image]):
            admission = admissions.get(str(record['owner_id']))
            description, description_positive, source_kind = _description(record, source, admission)
            start = len(continuation)
            description_ids = tokenizer.encode(description, add_special_tokens=False)
            continuation.extend(_field_ids(tokenizer, description, list(record['reference_coord_bins_1000']), coordinate_ids))
            weights.extend([1, *([1] * len(description_ids) if description_positive else [0] * len(description_ids)), *([1] * 7)])
            coordinate_positions = list(range(start + 1 + len(description_ids) + 2, start + 1 + len(description_ids) + 6))
            boxes.append({'x1_position': coordinate_positions[0], 'y1_position': coordinate_positions[1], 'x2_position': coordinate_positions[2], 'y2_position': coordinate_positions[3], 'expected_bins': list(record['reference_coord_bins_1000'])})
            description_proposal = (admission or {}).get('reference_proposal_id') or record.get('legacy_observed_source_proposal_id') or (None if source is None else source['proposal_id'])
            if source_kind != 'catalog_category':
                require(isinstance(description_proposal, str) and description_proposal, 'description source provenance')
            trace.append({'owner_id': record['owner_id'], 'order': order, 'source_kind': 'appended_missing_owner' if appended else 'first_fit_step16_first_occurrence', 'source_decision': None if source is None else {'proposal_id': source['proposal_id'], 'generated_order': source['generated_order'], 'original_description': source['raw']['description'], 'original_coord_bins_1000': source['raw']['coord_bins_1000'], 'physical_status': source['physical_status'], 'class': source['class']}, 'edited_fields': {'description_source': source_kind, 'selected_description': description, 'description_ce_positive': description_positive, 'description_source_proposal_id': description_proposal, 'description_authority': 'catalog' if source_kind == 'catalog_category' else 'observed_proposal', 'description_token_positions': list(range(start + 1, start + 1 + len(description_ids))), 'catalog_reference_coord_bins_1000': list(record['reference_coord_bins_1000']), 'geometry_replaced': source is None or source['raw']['coord_bins_1000'] != record['reference_coord_bins_1000']}})
        continuation.append(EOS)
        weights.append(1)
        case, plan = parent['case'], parent['case']['image_plan']
        route = {'route_id': f'complete-synthetic:image-{image:012d}', 'image_id': image, 'example_id': parent['example_id'], 'case': case, 'image_identity': {'image_path': case['image_path'], 'image_content_sha256': plan['image_content_sha256'], 'executed_media_sha256': plan['executed_media_sha256'], 'observed_image_grid_thw': plan['observed_image_grid_thw']}, 'prompt_token_ids': parent['prompt_token_ids'], 'continuation_token_ids': continuation, 'ce_weights': weights, 'trusted_boxes': boxes, 'trusted_complete_support_endpoint': True, 'provenance': {'synthetic_teacher': True, 'fixed_owner_ids': [record['owner_id'] for record, _, _ in planned[image]], 'trace': trace}}
        validate_route(route, eos_token_id=EOS, coordinate_token_ids=coordinate_ids)
        routes.append(route)

    parser_receipt = _parser_roundtrip(routes, Path(prep['acquisition_manifest']['path']), tokenizer)
    bank = {'schema': SCHEMA, 'status': 'candidate_ready', 'sources': {'catalog': binding(catalog_path), 'first_fit_preparation': binding(PREP), 'first_fit_step16_readback': binding(READBACK), 'first_fit_decisions': binding(DECISIONS), 'admissions': binding(ADMISSIONS), 'first_fit_admissions': binding(FIRST_ADMISSIONS), 'rulings': binding(RULINGS), 'parent_v4_additions': binding(ADDITIONS), 'stage1_proposal_rows': [binding(path) for path in sorted(STAGE1.glob('*.jsonl'))], 'producer': binding(Path(__file__))}, 'fixed_owner_count': 232, 'parser_receipt': parser_receipt, 'routes': routes, 'content_sha256': None}
    bank['content_sha256'] = digest({key: value for key, value in bank.items() if key != 'content_sha256'})
    validate(bank)
    publish(output / 'bank.json', bank)
    publish(output / 'runtime-routes.json', routes)
    publish(output / 'runtime-routes-metadata.json', {'schema': SCHEMA + '.runtime_routes', 'status': 'candidate_ready', 'bank': binding(output / 'bank.json'), 'route_count': len(routes)})
    return bank


def validate(bank: Mapping[str, Any]) -> None:
    require(bank.get('schema') == SCHEMA and bank.get('content_sha256') == digest({key: value for key, value in bank.items() if key != 'content_sha256'}), 'bank identity')
    require(bank.get('fixed_owner_count') == 232 and len(bank.get('routes', [])) == 11, 'denominator')
    sources = bank['sources']
    for name in ('catalog', 'first_fit_preparation', 'first_fit_step16_readback', 'first_fit_decisions', 'admissions', 'first_fit_admissions', 'rulings', 'parent_v4_additions', 'producer'):
        require(sources[name] == binding(Path(sources[name]['path'])), f'source identity: {name}')
    require(all(item == binding(Path(item['path'])) for item in sources['stage1_proposal_rows']), 'source identity: stage1 proposal rows')
    catalog = read(Path(sources['catalog']['path']))
    expected: dict[int, dict[str, list[int]]] = defaultdict(dict)
    for item in catalog['records']:
        expected[int(item['image_id'])][item['owner_id']] = list(item['reference_coord_bins_1000'])
    all_owners: list[str] = []
    for route in bank['routes']:
        require(route['image_id'] in expected, 'route image outside catalog')
        trace, owners = route['provenance']['trace'], [item['owner_id'] for item in route['provenance']['trace']]
        require(set(owners) == set(expected[route['image_id']]) and len(owners) == len(set(owners)) and len(trace) == len(route['trusted_boxes']) and route['ce_weights'][-1] == 1 and route['continuation_token_ids'][-1] == EOS, 'route exact completeness')
        for card, box in zip(trace, route['trusted_boxes']):
            edited, positions = card['edited_fields'], [box['x1_position'], box['y1_position'], box['x2_position'], box['y2_position']]
            require(edited['catalog_reference_coord_bins_1000'] == expected[route['image_id']][card['owner_id']] == box['expected_bins'], 'trace geometry')
            require(all(route['ce_weights'][position] == 1 for position in positions), 'coordinate mask')
            expected_description_weight = 1 if edited['description_ce_positive'] else 0
            require(all(route['ce_weights'][position] == expected_description_weight for position in edited['description_token_positions']), 'description mask')
            if edited['description_authority'] == 'catalog':
                require(edited['description_source'] == 'catalog_category', 'catalog description authority')
            else:
                require(isinstance(edited['description_source_proposal_id'], str) and edited['description_source_proposal_id'], 'description provenance')
        all_owners.extend(owners)
    require(len(all_owners) == len(set(all_owners)) == 232, 'global ownership')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=CATALOG)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    build(catalog_path=args.catalog, output=args.output)


if __name__ == '__main__':
    main()
