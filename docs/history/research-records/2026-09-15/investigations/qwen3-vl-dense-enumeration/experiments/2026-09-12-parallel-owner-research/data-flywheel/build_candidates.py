"""CPU-only literal candidate extraction and single-image review cards.

No model load, source mutation, GT augmentation, or training is performed.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SOURCE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/instance-state/panel-v1')
OUT = SOURCE.parent.parent / 'data-flywheel'
HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def literal_rows() -> tuple[dict, dict, list[dict]]:
    receipt = json.loads((SOURCE / 'receipt.json').read_text())
    consumer = json.loads((SOURCE / 'consumer.json').read_text())
    assert consumer['receipt_sha256'] == sha(SOURCE / 'receipt.json')
    assert receipt['status'] == 'complete'
    record = next(r for r in consumer['records'] if (r['case_id'], r['arm'], r['carrier']) == ('417044', 'owner_a', 'coordinates'))
    raw = json.loads((SOURCE / '417044-owner_a-coordinates.json').read_text())
    assert (raw['case_id'], raw['arm'], raw['carrier']) == ('417044', 'owner_a', 'coordinates')
    tokens = raw['prefix_ids'] + raw['suffix_ids']
    starts = [i for i, t in enumerate(tokens) if t == 151646]
    assert len(tokens) == 340 and tokens[-1] == 151645
    assert len(starts) == len(record['parsed']['pred']) == 34
    rows = []
    for i, (start, pred) in enumerate(zip(starts, record['parsed']['pred'])):
        end = starts[i + 1] if i + 1 < len(starts) else len(tokens) - 1
        row_ids = tokens[start:end]
        assert row_ids[-1] == 151649
        assert pred['generated_order'] == i
        assert raw['text'][pred['char_start']:pred['char_end']] == pred['raw_span_text']
        assert hashlib.sha256(pred['raw_span_text'].encode()).hexdigest() == pred['raw_span_sha256']
        assert [v - 151670 for v in row_ids if 151670 <= v <= 152669] == pred['coord_bins']
        rows.append({
            'row_id': f'P{i}', 'generated_order': i,
            'object_span_id': pred['object_span_id'],
            'bbox_xyxy_pixels': pred['bbox'], 'coordinate_bins': pred['coord_bins'],
            'description': pred['description'], 'row_token_ids': row_ids,
            'source_token_start': start, 'source_token_end_exclusive': end,
            'source_prefix_token_ids': tokens[:start],
            'row_text': pred['raw_span_text'], 'row_text_sha256': pred['raw_span_sha256'],
            'source_role': 'supplied_native_prefix' if end <= len(raw['prefix_ids']) else ('common_opener_then_free_row' if start < len(raw['prefix_ids']) else 'free_suffix_row'),
        })
    assert ''.join(r['row_text'] for r in rows) + '<|im_end|>' == raw['text']
    return record, raw, rows


def render_cards(record: dict, rows: list[dict]) -> dict:
    image_path = Path(record['parsed']['image_path'])
    original = Image.open(image_path).convert('RGB')
    assert original.size == (1152, 864)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    cards = []
    for row in rows:
        x1, y1, x2, y2 = row['bbox_xyxy_pixels']
        mx, my = max(45, (x2-x1)//2), max(45, (y2-y1)//2)
        crop_box = (max(0,x1-mx), max(0,y1-my), min(original.width,x2+mx), min(original.height,y2+my))
        clean = original.crop(crop_box)
        marked = clean.copy()
        ImageDraw.Draw(marked).rectangle((x1-crop_box[0],y1-crop_box[1],x2-crop_box[0],y2-crop_box[1]), outline='red', width=2)
        card = Image.new('RGB',(700,300),'white')
        draw = ImageDraw.Draw(card)
        draw.text((8,5),f"{row['row_id']} {row['description']} {row['bbox_xyxy_pixels']}",fill='black',font=font)
        draw.text((8,28),'Original context | same crop with literal predicted box',fill='black',font=font)
        for offset, im in ((0,clean),(350,marked)):
            scale = min(342/im.width,244/im.height)
            resized = im.resize((round(im.width*scale),round(im.height*scale)))
            card.paste(resized,(offset+(350-resized.width)//2,54+(244-resized.height)//2))
        path = OUT / 'cards' / f"{row['row_id']}.png"
        path.parent.mkdir(parents=True,exist_ok=True)
        card.save(path)
        cards.append({'row_id':row['row_id'],'path':str(path),'sha256':sha(path),'crop_xyxy_pixels':crop_box})
    sheets = []
    for j in range(0,len(cards),8):
        sheet = Image.new('RGB',(1400,1200),'#eeeeee')
        for k,item in enumerate(cards[j:j+8]):
            sheet.paste(Image.open(item['path']),((k%2)*700,(k//2)*300))
        path = OUT / 'cards' / f'review-{j:02d}-{min(j+7,len(cards)-1):02d}.png'
        sheet.save(path)
        sheets.append({'path':str(path),'sha256':sha(path),'row_ids':[x['row_id'] for x in cards[j:j+8]]})
    result = {'image_path':str(image_path),'image_sha256':sha(image_path),'source_consumer_sha256':sha(SOURCE/'consumer.json'),'cards':cards,'same_image_review_sheets':sheets}
    save(OUT/'cards/manifest.json',result)
    return result


def iou(a: list[int], b: list[int]) -> float:
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def finalize(record: dict, raw: dict, rows: list[dict]) -> dict:
    review_path = HERE / 'adjudications.json'
    review = json.loads(review_path.read_text())
    assert set(review['rows']) == {r['row_id'] for r in rows}
    matches = {m['pred_index']: m for m in record['score']['50']['matches']}
    image_path = Path(record['parsed']['image_path'])
    image_sha = sha(image_path)
    packet = json.loads((SOURCE/'packet.json').read_text())
    case = next(c for c in packet['cases'] if c['case_id']=='417044')
    # Reuse the already prepared literal image identity, not a guessed adapter.
    identity_source = SOURCE.parent.parent/'history/preparation/training-inputs.json'
    identity = next(r for r in json.loads(identity_source.read_text())['positive_records'] if r['example_id'].endswith('417044'))
    assert identity['prompt_token_ids'] == case['prompt_ids']
    assert identity['image']['image_sha256'] == image_sha
    original_tokens = raw['prefix_ids'] + raw['suffix_ids']
    positive_mask = [0] * len(original_tokens)
    kept, removed, literal_records, seams = [], [], [], []
    clean_tokens, clean_text = [], ''
    for row in rows:
        values = review['rows'][row['row_id']]
        label, confidence, action, reason = values[:4]
        m = matches.get(row['generated_order'])
        assert (label=='gt_matched') == (m is not None)
        row.update({'candidate_label':label,'confidence':confidence,'proposed_action':action,'reason':reason,
                    'annotation_relative_match50':m,'review_card':str(OUT/'cards'/f"{row['row_id']}.png")})
        if label=='duplicate_of':
            earlier = next(x for x in rows if x['row_id']==values[4])
            assert earlier['generated_order'] < row['generated_order']
            row['duplicate_of'] = earlier['row_id']
            row['duplicate_pair_iou'] = iou(row['bbox_xyxy_pixels'], earlier['bbox_xyxy_pixels'])
            row['frozen_strict_gt_0p95_trigger'] = row['duplicate_pair_iou'] > .95
        if action == 'keep':
            row['candidate_owner_id'] = m['owner'] if m else f"417044:review:{row['row_id']}"
            kept.append(row['row_id'])
            for j in range(row['source_token_start'],row['source_token_end_exclusive']): positive_mask[j] = 1
            literal_records.append({'record_id':f"417044-clean-{row['row_id']}", 'example_id':identity['example_id'],
                'image':identity['image'], 'prompt_token_ids':case['prompt_ids'],
                'prompt_token_ids_sha256':identity['prompt_token_ids_sha256'],
                'prefix_token_ids':list(clean_tokens), 'target_token_ids':row['row_token_ids'],
                'source_row_id':row['row_id'], 'candidate_owner_id':row['candidate_owner_id'],
                'supervision_scope':'proposed_complete_row_no_eos',
                'condition_is_original':clean_tokens==row['source_prefix_token_ids']})
            if clean_tokens != row['source_prefix_token_ids']:
                seams.append({'row_id':row['row_id'],'original_prefix_token_count':len(row['source_prefix_token_ids']),
                              'clean_prefix_token_count':len(clean_tokens), 'removed_predecessors':list(removed)})
            clean_tokens.extend(row['row_token_ids'])
            clean_text += row['row_text']
        else:
            assert action=='neutral'
            removed.append(row['row_id'])
    assert len(kept)==25 and len(removed)==9
    assert len(clean_tokens)==249 and sum(positive_mask)==249 and len(positive_mask)==340
    assert all(x in kept for x in (f"P{m['pred_index']}" for m in matches.values()))
    assert kept[:3]==['P0','P1','P2'] and seams[0]['row_id']=='P6'
    clean = {'status':'candidate_not_training_authorized','source_row_ids':kept,'removed_row_ids':removed,
             'row_tokens':clean_tokens,'display_complete_token_ids':clean_tokens+[151645],
             'display_text':clean_text+'<|im_end|>', 'train_eos':False,
             'eos_policy':'Retained only as a display terminator. Known missing GT and unreviewed owners preclude claiming completeness; no EOS loss is proposed.',
             'condition_changed_row_count':len(seams),'changed_conditions':seams,
             'literal_positive_records_path':str(OUT/'proposed-literal-positive-records.json')}
    source_files = [SOURCE/'receipt.json',SOURCE/'consumer.json',SOURCE/'packet.json',
                    SOURCE/'417044-owner_a-coordinates.json',SOURCE/'runner.py',SOURCE/'model.json',
                    SOURCE/'visuals/0000_coco2017_train_000000417044_gt_vs_pred.png',identity_source,image_path,review_path,Path(__file__).resolve()]
    sidecar = {'schema':'owner-review-sidecar-v1','status':'candidate_not_ground_truth_or_training_grant',
               'source':{'case_id':'417044','arm':'owner_a','carrier':'coordinates','checkpoint':packet['anchor_adapter'],
                         'intervention':'Inference-only row-coordinate KV transplant after common native opener',
                         'source_files':{str(p):sha(p) for p in source_files}},
               'counts':dict(Counter(r['candidate_label'] for r in rows)),
               'rows':rows,'source_positive_token_mask':positive_mask,
               'mask_scope':'Illustrative original-token selection only; proposed cleaned histories are explicit in separate records. Zero local loss does not imply zero parameter/probability change.',
               'viewed_images':[{'path':p,'sha256':sha(Path(p))} for p in review['actually_viewed_paths']]}
    save(OUT/'candidate-sidecar.json',sidecar)
    save(OUT/'cleaned-trajectory.json',clean)
    save(OUT/'proposed-literal-positive-records.json',literal_records)
    summary = {'status':'candidate_cpu_validation_passed','source_rows':len(rows), 'original_tokens':len(original_tokens),
               'kept_rows':len(kept),'kept_ids':kept,'neutral_rows':len(removed),'neutral_ids':removed,
               'counts':sidecar['counts'],'original_gt_matches50':len(matches),'retained_gt_matches50':len(matches),
               'unlabeled_single_owner_candidates':sum(r['candidate_label']=='trusted_unlabeled_single_owner' for r in rows),
               'free_suffix_unlabeled_candidates':sum(r['candidate_label']=='trusted_unlabeled_single_owner' and r['source_role']!='supplied_native_prefix' for r in rows),
               'clean_row_tokens':len(clean_tokens),'display_total_tokens':len(clean_tokens)+1,
               'changed_condition_rows':len(seams),'unchanged_condition_rows':len(kept)-len(seams),
               'direct_source_token_loss_masked':len(original_tokens)-sum(positive_mask),
               'source_unchanged':all(sha(Path(p))==h for p,h in sidecar['source']['source_files'].items()),
               'gt_mutations':0,'gpu_calls':0,'model_calls':0,'train_updates':0,
               'artifacts':{str(p):sha(p) for p in [OUT/'candidate-sidecar.json',OUT/'cleaned-trajectory.json',OUT/'proposed-literal-positive-records.json',OUT/'cards/manifest.json']}}
    assert summary['free_suffix_unlabeled_candidates']==13 and summary['source_unchanged']
    save(OUT/'validation.json',summary)
    return summary


if __name__ == '__main__':
    record, raw, rows = literal_rows()
    save(OUT/'literal-rows.json',rows)
    manifest = render_cards(record,rows)
    summary = finalize(record,raw,rows)
    print(json.dumps({k:v for k,v in summary.items() if k!='artifacts'}))
