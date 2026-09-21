import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from src.inference.parsing import parse_compact_object_box_closed
from src.vis import render_gt_vs_prediction, render_prediction_comparison
from src.vis.normalization import load_visual_rows
from src.vis.rendering import PANEL_LEFT_X, PANEL_RIGHT_X, PANEL_Y, PANEL_WIDTH, CANVAS_HEIGHT

base = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
study = base / '2026-09-16-source256-output-ranking-repair'
out = Path(__file__).parent
ids = [351017, 417044, 477415]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
prep_path = base / '2026-09-16-source256-fixed-prefix-completion/preparation/source256-admitted-v1/preparation.json'
prep = json.loads(prep_path.read_text())
train_path = Path(prep['sources']['train_jsonl']['path'])
assert sha(train_path) == prep['sources']['train_jsonl']['sha256']
result_path = study / 'runtime/main-v1/result.json'
assert sha(result_path) == '26f59a8409a64c23089b9bbfa4e5e3bb8af6b3f9de1971227fb55c17840fe783'
result = json.loads(result_path.read_text())
data = {x['image_id']: x for x in map(json.loads, train_path.read_text().splitlines())}
bank = {x['image_id']: x for x in prep['bank']['records']}
manifest = {'image_ids': ids, 'source_result': {'path': str(result_path), 'sha256': sha(result_path)}, 'frozen_bank': {'path': str(prep_path), 'sha256': sha(prep_path)}, 'renderer': 'src.vis public APIs', 'notes': ['GT is frozen admitted owner bank.', 'Shared renderer uses class-aware matching. Red/FP is annotation-unmatched, not confirmed physical false positive.', 'All parse-valid boxes are drawn without deduplication; invalid/unparseable rows remain in per-image source JSON.', 'G index is bank order; P index is parsed-valid order. Original generated order and owner IDs remain in input artifacts.'], 'inputs': {}, 'panels': []}
plots = {}
for arm in ['P', 'R']:
    score = {x['image_id']: x for x in result['scores'][arm + '16']['splits']['train']['per_image']}
    selected = {}
    for shard_path in sorted((study / f'runtime/main-v1/{arm}/readback/train').glob('shard-*.json')):
        shard = json.loads(shard_path.read_text())
        assert shard['status'] == 'completed_unscored' and shard['generation']['status'] == 'completed'
        for j, row in enumerate(shard['generation']['rows']):
            if row['image_id'] in ids:
                selected[row['image_id']] = row, shard_path, j
    assert set(selected) == set(ids)
    adir = out / f'{arm}-input'
    adir.mkdir()
    raw, scored, sources = [], [], []
    for idx, im in enumerate(ids):
        row, shard_path, j = selected[im]
        d = data[im]
        ip = (train_path.parent / d['images'][0]).resolve()
        assert ip.is_file() and Image.open(ip).size == (d['width'], d['height'])
        parsed = parse_compact_object_box_closed(row['raw_decode_text'], row_id=row['example_id'], row_index=row['row_index'], image_width=d['width'], image_height=d['height']).to_artifact_dict()
        assert parsed['valid_prediction_count'] == score[im]['valid_prediction_count']
        gt = [{'bbox': o['coord_bins'], 'desc': o['description'], 'owner_id': o['owner_id'], 'source': o['source']} for o in bank[im]['owners']]
        common = {'row_id': str(im), 'row_index': idx, 'image_path': str(ip), 'image_width': d['width'], 'image_height': d['height'], 'gt': gt, 'source_endpoint': arm + '16', 'source_shard': str(shard_path), 'source_shard_row_index': j}
        raw.append(common)
        scored.append({**common, 'pred': parsed['predictions']})
        src = out / f'{im}-{arm}-source.json'
        src.write_text(json.dumps({'saved_generation': row, 'parsed': parsed, 'evaluation': score[im]}, ensure_ascii=False, indent=2) + '\n')
        sources.append({'image_id': im, 'image_path': str(ip), 'image_sha256': sha(ip), 'shard': str(shard_path), 'shard_sha256': sha(shard_path), 'source_row_index': j, 'source_json': str(src), 'prompt_sha256': row['prompt_token_ids_sha256'], 'executed_media_sha256': row['executed_media_sha256'], 'gt_count': len(gt), 'valid_pred_count': len(parsed['predictions']), 'dropped_count': parsed['dropped_prediction_count']})
    for name, rows in [('gt_vs_pred.jsonl', raw), ('gt_vs_pred_scored.jsonl', scored)]:
        (adir / name).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    loaded = load_visual_rows(adir)
    assert [x.row_id for x in loaded.rows] == list(map(str, ids))
    for row in loaded.rows:
        assert len(row.gt) == bank[int(row.row_id)]['owner_count'] and len(row.pred) == score[int(row.row_id)]['valid_prediction_count']
    vr = render_gt_vs_prediction(adir, out / f'gt-vs-{arm}', row_ids=list(map(str, ids)), duplicate_iou_threshold=.95)
    assert len(vr.image_paths) == 3 and len(json.loads(vr.manifest_path.read_text())['items']) == 3
    plots[arm] = vr.image_paths
    manifest['inputs'][arm] = sources
    manifest['inputs'][arm + '_renderer_manifest'] = str(vr.manifest_path)
for i, im in enumerate(ids):
    p, r = manifest['inputs']['P'][i], manifest['inputs']['R'][i]
    assert p['prompt_sha256'] == r['prompt_sha256'] and p['executed_media_sha256'] == r['executed_media_sha256'] and p['image_sha256'] == r['image_sha256']
cmp = render_prediction_comparison(out / 'P-input', out / 'R-input', out / 'P-vs-R', left_label='P16: positive CE', right_label='R16: CE + ranking', row_ids=list(map(str, ids)), duplicate_iou_threshold=.95)
manifest['comparison_renderer_manifest'] = str(cmp.manifest_path)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
for i, im in enumerate(ids):
    p, r = Image.open(plots['P'][i]), Image.open(plots['R'][i])
    pieces = [('GT frozen bank', r, PANEL_LEFT_X), ('P16 - positive CE', p, PANEL_RIGHT_X), ('R16 - CE + ranking', r, PANEL_RIGHT_X)]
    individual = []
    for key, (label, source, x) in zip(['GT', 'P', 'R'], pieces):
        panel = source.crop((x, PANEL_Y, x + PANEL_WIDTH, CANVAS_HEIGHT))
        canvas = Image.new('RGB', (PANEL_WIDTH, panel.height + 36), 'white')
        canvas.paste(panel, (0, 36))
        ImageDraw.Draw(canvas).text((8, 5), f'{im} | {label}', font=font, fill='black')
        dest = out / f'{im}-{key}.png'
        canvas.save(dest)
        individual.append(dest)
    combined = Image.new('RGB', (PANEL_WIDTH * 3 + 24, Image.open(individual[0]).height), (230, 230, 230))
    for j, sp in enumerate(individual):
        combined.paste(Image.open(sp), (j * (PANEL_WIDTH + 12), 0))
    dest = out / f'{im}-GT-P-R.png'
    combined.save(dest)
    manifest['panels'].append({'image_id': im, 'combined': str(dest), 'GT': str(individual[0]), 'P': str(individual[1]), 'R': str(individual[2]), 'gt_vs_P': str(plots['P'][i]), 'gt_vs_R': str(plots['R'][i]), 'P_vs_R': str(cmp.image_paths[i])})
    for path in [*individual, dest, plots['P'][i], plots['R'][i], cmp.image_paths[i]]:
        with Image.open(path) as check:
            check.verify()
manifest['producer'] = {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}
(out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'manifest': str(out / 'manifest.json'), 'panels': manifest['panels'], 'verified': True}, ensure_ascii=False))
