"""Round11 CPU presentation derivative of accepted A_A/D_I saved outputs."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import resource
import subprocess
import time

from PIL import Image, ImageDraw, ImageFont

from probes import owner_transition_attribution as owner
from probes.hidden_human_recovery import canonical, candidates, write
from src.data.geometry import coord_bins_to_pixel_xyxy
from src.eval.saved_rows import iou_xyxy
from src.vis import render_prediction_comparison
from src.vis.normalization import load_visual_rows

ROOT = Path(__file__).resolve().parents[1]
EXEC = Path('/data/CoordExp/.worktrees/greedy-prefix-native-01')
PRE = EXEC / 'outputs/research/physical-fn-recovery/2026-10-03/dora-input-ablation-10'
OUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-03/known-owner-visual-audit-11'
FROZEN = {'lead-acceptance-01.json': '1ce79e06a96ab5b7389240bb6ddc3afe31927c0cca00b114ba213b89bfcf910b',
          'package-01/complete.json': '8f77594229d5cf986d4c82663fdd68a1a37252dd19bbaf0aec4badfd58e17157',
          'released-contract-01.json': '15bfa661e11056885dce687202f25029e9e5f4740c7567e8cffca8cb91769945'}
PANEL = {1584: ([541559], []), 2299: ([-17], [-27]), 2685: ([], [-86, -79]),
         4134: ([], [-99]), 5001: ([], [-107, 2017429]), 6040: ([228220], []),
         14439: ([1170759, 1720080], []), 16228: ([], [-53, -44, -39, 2018451]),
         417044: ([-8107737316336679, -7084232513636164, -6208576829732043, 1079910, 1083295], [-3069762157606252, 1083375]),
         477415: ([], [1586761])}
ARMS = ('A_A', 'D_I')
SOURCE = ['probes/known_owner_visual_audit.py', 'tests/probes/test_known_owner_visual_audit.py']
NOTE = ('CPU presentation derivative; zero new inference. Shared renderer uses its own class-aware pixel matching, '
        'not the frozen class-agnostic assignment. Colors, IoU and row order do not identify entities. '
        'Neutral blue rows are model candidates, cyan reference is a frozen label, not visual ground truth. '
        'Crop absence is not full-output absence; all valid and invalid rows remain in pools/raw records. '
        'Entity, category, localization and physical coverage judgments are pending lead review.')


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def pixel(box, image):
    return list(coord_bins_to_pixel_xyxy([int(v) for v in box], image_width=image['width'],
                                        image_height=image['height'], field='round11.reference_or_prediction'))


def projection(image, raw, pool, index):
    """Strict full-image adapter, preserving every valid parser row and order."""
    owner.require(raw['image_id'] == image['image_id'] and
                  all(raw[k] == image[k] for k in ('image_path', 'image_sha256', 'width', 'height')) and
                  raw['crop'] == [0, 0, image['width'], image['height']], 'adapter raw/image mismatch')
    parsed = owner.rows.parse(raw)
    owner.require(len(pool) == len(parsed.predictions), 'adapter dropped valid row')
    pred = []
    for p, row in zip(pool, parsed.predictions, strict=True):
        owner.require(p['prediction_id'] == f"{raw['request_id']}:p{row['generated_order']}" and
                      p['description'] == row['description'] and p['coord_bins_1000'] == row['coord_bins'],
                      'adapter prediction row mismatch')
        box = pixel(row['coord_bins'], image)
        owner.require(box[0] < box[2] and box[1] < box[3], 'pixel rounding collapsed valid row')
        pred.append(dict(description=p['description'], bbox=box, coord_bins=row['coord_bins'],
                         prediction_id=p['prediction_id'], raw_parser_order=row['generated_order']))
    return dict(row_id=str(image['image_id']), row_index=index, image_path=image['image_path'],
                image_width=image['width'], image_height=image['height'],
                gt=[dict(description=o['desc'], bbox=o['bbox_2d'], owner_id=str(o['coco_ann_id'])) for o in image['objects']],
                pred=pred, provenance='CPU derivative of accepted saved raw; not a new model/scored observation')


def check_projection(path, images, records, pools):
    normalized = load_visual_rows(path).rows
    owner.require([r.row_id for r in normalized] == [str(i['image_id']) for i in images], 'visual row order mismatch')
    for index, (v, image) in enumerate(zip(normalized, images, strict=True)):
        expected = projection(image, records[index]['raw'], pools[index], index)
        owner.require(len(v.pred) == len(expected['pred']) and len(v.gt) == len(image['objects']), 'visual pool count mismatch')
        owner.require([list(p.bbox_pixel_xyxy) for p in v.pred] == [p['bbox'] for p in expected['pred']] and
                      [p.description for p in v.pred] == [p['description'] for p in expected['pred']], 'visual prediction mismatch')
        owner.require([list(g.bbox_pixel_xyxy) for g in v.gt] == [pixel(o['bbox_2d'], image) for o in image['objects']], 'visual GT mismatch')
    return normalized


def inputs(out):
    bindings = {}
    def read(path, expected=None):
        path = Path(path); payload = path.read_bytes(); digest = sha(payload)
        owner.require(expected is None or expected == digest, f'input identity mismatch: {path}')
        bindings[str(path)] = digest
        return json.loads(payload), payload
    accepted, _ = read(PRE / 'lead-acceptance-01.json', FROZEN['lead-acceptance-01.json'])
    terminal, _ = read(PRE / 'package-01/complete.json', FROZEN['package-01/complete.json'])
    contract, _ = read(PRE / 'released-contract-01.json', FROZEN['released-contract-01.json'])
    owner.require(accepted['status'] == 'lead-accepted' and accepted['terminal']['sha256'] == FROZEN['package-01/complete.json'] and
                  accepted['release']['sha256'] == FROZEN['released-contract-01.json'], 'acceptance chain mismatch')
    labels, _ = read(contract['label_path'])
    label_positions = {i['image_id']: n for n, i in enumerate(labels)}
    images = [labels[label_positions[i]] for i in PANEL]
    tokenizer, _ = read(Path(contract['base_model']) / 'tokenizer.json', contract['tokenizer_sha256'])
    special = {t['content']: t['id'] for t in tokenizer['added_tokens'] if t['special']}
    owner.require([special[f'<|coord_{n}|>'] for n in range(1000)] == contract['coordinate_ids'], 'structural coordinate ID mismatch')
    records, pools, identities = {}, {}, []
    for arm in ARMS:
        phase_path = PRE / f'package-01/native-{arm}/complete.json'
        phase, _ = read(phase_path, accepted['verified_record_bindings'][str(phase_path)])
        records[arm], pools[arm] = [], []
        measured = {m['image_id']: m for m in terminal['natural'][arm]}
        for image in images:
            name = f"natural-{image['image_id']}.json"; path = phase_path.parent / name
            record, payload = read(path, phase['artifacts'][name]); raw = record['raw']
            expected_weight = contract['weights'][arm]['weight_identity'] if 'weight_identity' in contract['weights'][arm] else phase['weight_identity']
            owner.require(raw['producer']['weight_identity'] == expected_weight == phase['weight_identity'] and
                          raw['producer']['source'] == contract['source']['commit'], 'raw arm/source mismatch')
            owner.require(raw['token_ids'] == record['acquisition']['token_ids'] and len(raw['token_ids']) == raw['generated_tokens'], 'accepted token identity mismatch')
            sealed = {k:raw[k] for k in ('producer','request_id','image_id','token_ids','text','prompt_token_ids','media_sha256','image_grid_thw','stop_reason')}
            owner.require(sha(canonical(sealed).encode()) == raw['raw_identity'], 'raw identity seal mismatch')
            pool = owner.positioned_pool(raw, special)
            actual = owner.rows.assess_outputs([image], [], [raw])[0]
            owner.require(actual == record['measurement'] == measured[image['image_id']], 'accepted evaluator/witness mismatch')
            records[arm].append(dict(path=str(path), **record)); pools[arm].append(pool)
            destination = out / 'raw' / arm / name; destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(payload)
            identities.append(dict(arm=arm, image_id=image['image_id'], path=str(path), sha256=bindings[str(path)],
                                   raw_identity=raw['raw_identity'], copied_raw=str(destination), weight_identity=phase['weight_identity']))
    edge = terminal['natural_transitions']['joint_update']
    owner.require(edge['before'] == 'A_A' and edge['after'] == 'D_I', 'accepted edge mismatch')
    actual_keys = {(r['image_id'], str(o), d) for r in edge['images'] for d,k in [('gain','gained'),('loss','lost')] for o in r['owners']['category'][k]}
    frozen_keys = {(i,str(o),d) for i, pair in PANEL.items() for d, ids in zip(('gain','loss'),pair) for o in ids}
    owner.require(actual_keys == frozen_keys and len(actual_keys) == 23 and edge['owner_counts']['category']['gained'] == 10 and
                  edge['owner_counts']['category']['lost'] == 13, 'frozen 23 key mismatch')
    image_ids = []
    for image in images:
        path = Path(image['image_path']); payload = path.read_bytes(); digest = sha(payload)
        owner.require(digest == image['image_sha256'], 'source image identity mismatch')
        with Image.open(path) as img: owner.require(img.size == (image['width'],image['height']), 'source image dimension mismatch')
        destination = out / 'source-images' / path.name; destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(payload)
        image_ids.append(dict(image_id=image['image_id'], original=str(path), unmarked_copy=str(destination), sha256=digest,
                              width=image['width'], height=image['height']))
    return images, records, pools, label_positions, contract, bindings, identities, image_ids


def draw_rows(source, pool, image, crop, reference=None):
    view = source.crop(crop).convert('RGB'); draw = ImageDraw.Draw(view)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    visible = []
    for n,p in enumerate(pool):
        b = pixel(p['coord_bins_1000'], image)
        if b[2] <= crop[0] or b[0] >= crop[2] or b[3] <= crop[1] or b[1] >= crop[3]: continue
        visible.append(p['prediction_id']); local = [b[0]-crop[0],b[1]-crop[1],b[2]-crop[0],b[3]-crop[1]]
        draw.rectangle(local, outline='#3876db', width=2)
        xy = (max(0,local[0]), max(0,min(view.height-16,local[1])))
        label = f"P{n}/raw{p['raw_parser_order']} {p['description']}"
        bounds = draw.textbbox(xy,label,font=font); draw.rectangle(bounds,fill='white'); draw.text(xy,label,fill='#153d83',font=font)
    if reference:
        b = pixel(reference['bbox_2d'],image); local = [b[0]-crop[0],b[1]-crop[1],b[2]-crop[0],b[3]-crop[1]]
        draw.rectangle(local, outline='#00ffff',width=4)
        for offset,label in [(32,f"REF {reference['desc']}"),(16,str(reference['coco_ann_id']))]:
            xy=(2,view.height-offset); draw.rectangle(draw.textbbox(xy,label,font=font),fill='black');draw.text(xy,label,fill='#00ffff',font=font)
    return view, visible


def target_view(out, image, obj, pools, key):
    box = pixel(obj['bbox_2d'],image)
    pad = max(96, (box[2]-box[0])*.65, (box[3]-box[1])*.65)
    crop = [max(0, math.floor(box[0]-pad)), max(0, math.floor(box[1]-pad)),
            min(image['width'],math.ceil(box[2]+pad)), min(image['height'],math.ceil(box[3]+pad))]
    folder=out/'targets'/key;folder.mkdir(parents=True,exist_ok=True)
    with Image.open(image['image_path']) as src:
        src.crop(crop).save(folder/'unmarked-context.png')
        pair=Image.new('RGB',(2*(crop[2]-crop[0]),crop[3]-crop[1]+68),'white');draw=ImageDraw.Draw(pair)
        draw.text((4,4),key+' | cyan=reference; blue=model candidates',fill='black')
        visible={}
        for n,arm in enumerate(ARMS):
            panel,visible[arm]=draw_rows(src,pools[arm],image,crop,obj)
            pair.paste(panel,(n*panel.width,68));draw.text((n*panel.width+4,26),f'{arm} | P index/full pool',fill='black')
            draw.text((n*panel.width+4,44),'raw index/parser; judgment pending',fill='black')
        pair.save(folder/'comparison.png')
    return dict(comparison=str(folder/'comparison.png'),unmarked_context=str(folder/'unmarked-context.png'),
                crop_source_pixel_xyxy=crop, context_to_source=dict(scale=[1,1],translation=crop[:2]),
                pair_panel_origins=dict(A_A=[0,68],D_I=[crop[2]-crop[0],68]),
                visible_prediction_ids=visible, reference_pixel_xyxy=box, reference_drawn_both_arms=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    started=time.monotonic();out=args.output;owner.require(not (out/'receipt.json').exists(),'refuse existing package receipt')
    images,records,pools,label_positions,contract,bindings,raw_ids,image_ids=inputs(out)
    source=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                diff=subprocess.check_output(['git','diff','HEAD','--',*SOURCE],text=True),paths=SOURCE)
    source_file=out/'source.diff';source_file.write_text(source['diff']);source['diff_path']=str(source_file)
    for arm in ARMS:
        folder=out/'projection'/arm;folder.mkdir(parents=True,exist_ok=True)
        projected=[projection(image,records[arm][n]['raw'],pools[arm][n],n) for n,image in enumerate(images)]
        for name in ('gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl'):
            (folder/name).write_text(''.join(canonical(v)+'\n' for v in projected))
        check_projection(folder,images,records[arm],pools[arm])
    entries=[]; image_entries=[]; boundary=None
    for n,image in enumerate(images):
        ident=image['image_id']; per_arm={a:pools[a][n] for a in ARMS}; folder=out/'images'/str(ident)
        shared=render_prediction_comparison(out/'projection/A_A',out/'projection/D_I',folder/'shared',
                    row_ids=[str(ident)],left_label='A_A saved CPU projection',right_label='D_I saved CPU projection')
        owner.require(len(shared.image_paths)==1 and json.loads(shared.manifest_path.read_text())['items'][0]['row_id']==str(ident),'renderer inventory mismatch')
        with Image.open(shared.image_paths[0]) as png: owner.require(png.size==(1800,830),'renderer output dimension mismatch')
        full=[0,0,image['width'],image['height']]; overlays={}
        with Image.open(image['image_path']) as src:
            for arm in ARMS:
                view,visible=draw_rows(src,per_arm[arm],image,full)
                owner.require(visible==[p['prediction_id'] for p in per_arm[arm]],'source-resolution full pool omitted rows')
                path=folder/f'{arm}-source-resolution.png';view.save(path);overlays[arm]=str(path)
        pool_path=folder/'pools.json'
        write(pool_path,dict(image=image,arms={a:dict(raw_locator=records[a][n]['path']+'#/raw',
             raw_identity=records[a][n]['raw']['raw_identity'], valid_pool=per_arm[a],
             invalid_malformed_rows=candidates([records[a][n]['raw']])[1],accepted_measurement=records[a][n]['measurement']) for a in ARMS}))
        image_entries.append(dict(image_id=ident,source=image_ids[n],paired_full_image=str(shared.image_paths[0]),
                                  renderer_manifest=str(shared.manifest_path),source_resolution=overlays,pools=str(pool_path)))
        by_owner={str(o['coco_ann_id']):o for o in image['objects']}
        owner.require(len(by_owner)==len(image['objects']),'duplicate frozen reference owner')
        for direction,ids in zip(('gain','loss'),PANEL[ident]):
            for oid in ids:
                obj=by_owner[str(oid)];key=f'{ident}_{oid}_{direction}'
                sides={a:owner.side_witness(image,obj,records[a][n],records[a][n]['measurement'],per_arm[a]) for a in ARMS}
                owner.require(sides['A_A']['category_covered']==(direction=='loss') and sides['D_I']['category_covered']==(direction=='gain'),'owner transition witness mismatch')
                for a in ARMS:
                    assignments={m['prediction_id']:m for m in records[a][n]['measurement']['matches']}
                    sides[a]['all_candidate_overlap_category_eligibility']=[dict(prediction_id=p['prediction_id'],
                        iou_to_reference=iou_xyxy(obj['bbox_2d'],p['coord_bins_1000']),same_category=p['description']==obj['desc'],
                        eligible_geometry=iou_xyxy(obj['bbox_2d'],p['coord_bins_1000'])>=.5,
                        eligible_category=p['description']==obj['desc'] and iou_xyxy(obj['bbox_2d'],p['coord_bins_1000'])>=.5,
                        evaluator_assignment=assignments.get(p['prediction_id'])) for p in per_arm[a]]
                target=target_view(out,image,obj,per_arm,key)
                entry=dict(key=key,image_id=ident,owner_id=str(oid),direction=direction,reference=dict(category=obj['desc'],
                    norm1000_xyxy=obj['bbox_2d'],pixel_xyxy=pixel(obj['bbox_2d'],image),label_locator=contract['label_path']+f'#/{label_positions[ident]}/objects/{image["objects"].index(obj)}'),
                    sides=sides,uncovered_evaluator_class=owner.classify(sides['A_A' if direction=='gain' else 'D_I']),view=target,
                    full_image=image_entries[-1],judgments=dict(visible_entity_representation='pending_lead',category='pending_lead',localization='pending_lead',physical_coverage='pending_lead'))
                case_path=out/'targets'/key/'case.json';write(case_path,entry);entries.append(dict(key=key,case=str(case_path),view=target))
                if boundary is None:
                    owner.require(ident==1584 and oid==541559 and direction=='gain','first scheduled boundary mismatch')
                    boundary=dict(image_id=ident,key=key,renderer_manifest=str(shared.manifest_path),
                                  checked='real saved adapter -> supported JSONLs -> shared renderer -> PNG/manifest; explicit target reference and full pools',passed=True)
                    write(out/'first-boundary.json',boundary);print(canonical(dict(checkpoint='first_scheduled_image_and_target',status='passed',key=key)),flush=True)
    owner.require(len(image_entries)==10 and len(entries)==23 and len({e['key'] for e in entries})==23,'final inventory mismatch')
    write(out/'ledger.json',dict(schema='round11-known-owner-cpu-derivative-v1',note=NOTE,images=image_entries,cases=entries,
                                evaluator_semantics=owner.SEMANTICS,lead_judgments='pending',model_calls=0,GPU_calls=0))
    lines=['<!doctype html><meta charset="utf-8"><title>Round11 saved owner audit</title>',
           '<style>body{font:16px sans-serif;max-width:1200px;margin:24px auto}img{max-width:100%}li{margin:12px 0}</style>',
           '<h1>Round11: 23 saved known-owner transitions</h1>',f'<p>{html.escape(NOTE)}</p>','<p><a href="ledger.json">Complete ledger</a> | <a href="receipt.json">Receipt</a></p>']
    for start in range(0,len(entries),4):
        lines.append(f'<h2>Batch {start//4+1}: cases {start+1}–{min(start+4,len(entries))}</h2><ul>')
        for e in entries[start:start+4]:
            rel=Path(e['case']).relative_to(out);image=next(i for i in image_entries if i['image_id']==int(e['key'].split('_')[0]))
            lines.append(f'<li><b>{e["key"]}</b> <a href="{rel}">case / all overlap witnesses</a> | <a href="{Path(e["view"]["comparison"]).relative_to(out)}">paired target</a> | <a href="{Path(e["view"]["unmarked_context"]).relative_to(out)}">unmarked context</a> | <a href="{Path(image["paired_full_image"]).relative_to(out)}">full pair</a> | <a href="{Path(image["pools"]).relative_to(out)}">full pools / invalid rows</a> | <a href="{Path(image["source"]["unmarked_copy"]).relative_to(out)}">unmarked source</a> | '+ ' | '.join(f'<a href="{Path(image["source_resolution"][a]).relative_to(out)}">{a} source resolution</a>' for a in ARMS)+'</li>')
        lines.append('</ul>')
    (out/'index.html').write_text('\n'.join(lines))
    reused = {v['unmarked_copy']: v['sha256'] for v in image_ids}
    reused.update({v['copied_raw']: v['sha256'] for v in raw_ids})
    inventory={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=reused[str(p)] if str(p) in reused else sha(p.read_bytes()))
               for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('materialize.log','materialize-exit.json')}
    total=sum(v['bytes'] for v in inventory.values());wall=time.monotonic()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    owner.require(wall<1800 and rss<8*1024**3 and total<1024**3,'materialization resource bound exceeded')
    receipt=dict(status='CPU_EVIDENCE_CANDIDATE',lead_accepted=False,scientific_status='unmeasured',note=NOTE,
                 source=source,input_sha256=bindings,raw_identities=raw_ids,image_identities=image_ids,
                 checks=dict(exact_keys=23,gains=10,losses=13,images=10,raw_records=20,
                     accepted_measurements_equal=True,structural_token_text_mapping_equal=True,all_valid_rows_preserved=True,
                     full_image_pairs=10,focused_target_pairs=23,first_boundary=boundary),output_inventory=inventory,
                 resources=dict(active_wall_seconds=wall,peak_rss_bytes=rss,evidence_bytes=total),model_calls=0,GPU_calls=0,
                 execution_source=contract['source']['commit'],tokenizer_use='structural added_tokens JSON only; no tokenizer object, decoding, processor, model or checkpoint load',
                 next_unit_scheduled=False,stop='lead image audit / acceptance, then report and discussion; no round12')
    write(out/'receipt.json',receipt);print(canonical(dict(status=receipt['status'],checks=receipt['checks'],resources=receipt['resources'],receipt=str(out/'receipt.json'))))


if __name__=='__main__': main()
