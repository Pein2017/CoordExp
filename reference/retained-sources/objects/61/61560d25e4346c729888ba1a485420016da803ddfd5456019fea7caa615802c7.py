"""Freeze the eight-case history/image factorial; no model loading."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
from PIL import Image
from tokenizers import Tokenizer

ROOT = Path(__file__).parent
REPO = Path('/data/CoordExp/.worktrees/research-probes')
PREVIOUS = ROOT.parent / '2026-09-11-stable50-geometric-dedup/inputs.json'
DATA = Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox')
DONORS = {9813: 81205, 158044: 46432, 248167: 13043, 274509: 3125,
          351017: 13043, 417044: 46432, 477415: 13043, 502725: 3514}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def translate(box):
    x1, y1, x2, y2 = box
    # Move toward the largest available room; deterministic x+ / x- / y+ / y- ties.
    options = [(999-x2, 0, 1), (x1, 0, -1), (999-y2, 1, 1), (y1, 1, -1)]
    room, axis, sign = max(options, key=lambda x: x[0])
    assert room >= 250, 'no predeclared quarter-canvas translation available'
    result = list(box)
    for index in (axis, axis+2):
        result[index] += sign*250
    assert all(0 <= x <= 999 for x in result)
    assert (result[2]-result[0], result[3]-result[1]) == (x2-x1, y2-y1)
    return result


def main(draft):
    old = json.loads(PREVIOUS.read_text())
    census_path = ROOT/'census/census.json'
    census = json.loads(census_path.read_text())
    assert census['acceptance']['strict_repeat_counts_match']
    observed = {int(x['image_id']): x for x in census['cases']}
    records = {int(x['image_id']): x for x in old['eval_records']}
    source_cases = {int(x['image_id']): x for x in old['online_cases']}
    tokenizer_path = Path(old['model']['base_model_path'])/'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    extra = {}
    for index, line in enumerate((DATA/'train.jsonl').open()):
        record = json.loads(line)
        if record['image_id'] in {3125, 3514}:
            extra[record['image_id']] = (index, record)
        if len(extra) == 2:
            break
    paths = {PREVIOUS, census_path, ROOT/'census/build_census.py', Path(__file__), tokenizer_path}
    paths.update(REPO/p for p in ['probes/dora_owner_learning/runtime.py',
        'probes/dora_owner_learning/route_access.py', 'probes/source_rweak_row_cross/run.py',
        'probes/dora_owner_learning/geometric_dedup.py', 'src/qwen/native.py', 'src/qwen/generation.py'])
    for bundle in ('current_adapter', 'source_embedding'):
        identity = old['model'][bundle]
        for item in identity['files']:
            path = Path(identity['root'])/item['relative_path']
            assert sha(path) == item['sha256']
            paths.add(path)
    cases = []
    for image_id, donor_id in DONORS.items():
        source = source_cases[image_id]
        natural = observed[image_id]
        original = copy.deepcopy(source['case'])
        if donor_id in records:
            donor = copy.deepcopy(records[donor_id]['case'])
        else:
            index, record = extra[donor_id]
            record = copy.deepcopy(record)
            image_path = DATA/record['images'][0]
            record['images'] = [str(image_path)]
            plan = copy.deepcopy(original['image_plan'])
            plan.update(image_path=str(image_path), image_content_sha256=sha(image_path),
                        row_id=f'coco2017_train_{donor_id:012d}', row_index=index,
                        example_id=f'coco2017_train_{donor_id:012d}',
                        executed_media_sha256=None,
                        grid_provenance='expected_from_identical_image_dimensions_verify_live')
            donor = dict(image_height=record['height'], image_width=record['width'],
                         image_path=str(image_path), image_plan=plan, input_record=record,
                         row_id=f'coco2017_train_{donor_id:012d}', row_index=index)
        assert (original['image_width'], original['image_height']) == (donor['image_width'], donor['image_height'])
        for case in (original, donor):
            path = Path(case['image_path'])
            with Image.open(path) as image:
                assert image.size == (case['image_width'], case['image_height'])
            assert sha(path) == case['image_plan']['image_content_sha256']
            paths.add(path)
        ids = source['action_ids']
        assert ids == natural['action_token_ids'] == records[image_id]['stable_ids']
        seeds, jobs = {}, [dict(job_id='natural', image_condition='original',
            history_condition='native', boundary='natural', extension_ids=[], budget=3084)]
        donor_baseline = records[donor_id]['stable_ids'] if donor_id in records else None
        if donor_baseline is None:
            jobs.append(dict(job_id='donor_natural', image_condition='donor',
                history_condition='native', boundary='natural', extension_ids=[], budget=3084))
        for boundary in ('early', 'late'):
            if image_id == 9813:
                row_index = 0 if boundary == 'early' else 8
            else:
                key = ('early_last_complete_row_before_first_strict_repeat' if boundary == 'early'
                       else 'late_last_complete_row_after_up_to_8_repeat_description_emissions')
                row_index = natural['onset_summary']['probe_seed_candidates'][key]['raw_row_index']
            row = natural['raw_rows'][row_index]
            assert row['complete_canonical_row']
            start, stop = row['token_start'], row['token_end_exclusive']
            positions = [start+i for i, token in enumerate(row['token_texts']) if token.startswith('<|coord_')]
            assert len(positions) == 4 and ids[stop-1] == tokenizer.token_to_id('<|box_end|>')
            box = row['coord_bins']; translated = translate(box)
            extension = ids[:stop]; changed = list(extension)
            for position, coordinate in zip(positions, translated):
                changed[position] = tokenizer.token_to_id(f'<|coord_{coordinate}|>')
            assert len(extension) == len(changed) and extension != changed
            assert [i for i,(a,b) in enumerate(zip(extension,changed)) if a != b] == [p for p,b,t in zip(positions,box,translated) if b != t]
            assert 151645 not in extension and 151645 not in changed
            seeds[boundary] = dict(raw_row_index=row_index, original_bins=box,
                translated_bins=translated, description=row['description'],
                geometry_valid=row['geometry_valid'], coord_positions=positions,
                extension_length=stop, token_start=start)
            for image in ('original', 'donor'):
                for history in ('native', 'translated'):
                    jobs.append(dict(job_id=f'{boundary}_{image}_{history}', image_condition=image,
                        history_condition=history, boundary=boundary,
                        extension_ids=extension if history == 'native' else changed,
                        budget=min(512,3084-stop)))
        cases.append(dict(case_id=str(image_id), source_case=original, donor_case=donor,
            donor_image_id=donor_id, donor_baseline_action_ids=donor_baseline,
            donor_baseline_source='dedup_inputs.eval_records.stable_ids' if donor_baseline else 'new_donor_natural_job',
            prompt_token_ids=source['prompt_token_ids'],
            baseline_action_ids=ids, golden=records[image_id]['golden'], seeds=seeds, jobs=jobs))
    if not draft:
        paths.update([ROOT/'run_probe.py', ROOT/'test_run_probe.py', ROOT/'run_all.py'])
    packet = dict(schema='small_owner_repeat_origin.v1', config=old['config'],
        anchor_adapter=old['model']['current_adapter']['root'], model=old['model'],
        cases=cases, source_files={str(p):sha(p) for p in sorted(paths)},
        translation_rule='250 bins along axis/sign with largest available room, x+/x-/y+/y- tie order',
        clean_seed_rule='9813 rows0 and8; not repetitive-dose matched',
        claim_boundary='Conditional-history and image-dependence probe; no training or physical-owner truth from overlap',
        limits=dict(world_size=8, full_jobs=74, smoke_rank=4, smoke_jobs=5,
                    model_loads=9, total_continuations=79, total_new_token_upper_bound=68740,
                    seconds_per_worker=1500),
        donor_review=dict(reviewer='root-view_image', image_ids=sorted(set(DONORS.values())),
            note='Existing same-size photos personally viewed; no visible target bottle/book/vase/donut/chair/knife/horse/person as applicable. Absence is visual case review, not exhaustive annotation.'))
    output = ROOT/('packet-draft-v2.json' if draft else 'packet.json')
    assert not output.exists(), 'do not overwrite an existing packet'
    output.write_text(json.dumps(packet, indent=2, sort_keys=True)+'\n')
    cold=json.loads(output.read_text())
    assert cold == packet
    print(json.dumps(dict(path=str(output), sha256=sha(output), cases=len(cases),
        jobs=sum(len(c['jobs']) for c in cases), source_files=len(paths),
        seeds={c['case_id']:c['seeds'] for c in cases}), indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--draft', action='store_true')
    main(parser.parse_args().draft)
