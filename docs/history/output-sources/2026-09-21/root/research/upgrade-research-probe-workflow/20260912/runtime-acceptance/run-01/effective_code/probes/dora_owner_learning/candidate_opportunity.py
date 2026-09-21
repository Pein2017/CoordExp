"""CPU-only, source-bound complete-output opportunity reduction (no model load)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from src.artifacts import publish_json_exclusive
from src.data.geometry import iou_xyxy, parse_coord_token
from src.eval.assignment import global_matches
from src.inference.parsing import parse_compact_object_box_closed
from .reward_rows import _gt_objects, _pred_objects, _pixel_box, _bbox


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def indexed(items, key):
    result = {str(row[key]): row for row in items}
    require(len(result) == len(items), f"duplicate {key}")
    return result


def score(row, *, seed, length, stop):
    gt = _gt_objects(row, row_id=row['row_id'])
    pred, invalid = _pred_objects(row)
    geometry = [box for obj in row['pred'] if (box := _pixel_box(_bbox(obj))) is not None]
    ids = [str(obj['object_id']) for obj in row['gt']]
    require(len(set(ids)) == len(ids), "duplicate GT owner")
    result = dict(seed=seed, prediction_count=len(pred), parsed_prediction_count=len(row['pred']), invalid_predictions=invalid,
                  parser_drops=row['dropped_prediction_count'],
                  complete_token_length=length, stop_reason=stop, cap=int(stop == 'length'),
                  strict_repeats=sum(any(iou_xyxy(box, old) > .95 for old in geometry[:i])
                                     for i, box in enumerate(geometry)))
    for threshold in (50, 60, 80):
        matches = global_matches(gt, pred, threshold / 100)
        tp = len(matches)
        fp, fn = len(pred) - tp + invalid, len(gt) - tp
        result[str(threshold)] = dict(tp=tp, fp=fp, fn=fn,
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., recall=tp/len(gt) if gt else 0.,
            owners=[ids[i] for i, _, _ in matches],
            matches=[dict(owner=ids[i], gt_index=i, pred_index=j, iou=v) for i,j,v in matches])
    return result


def compare(sample, baseline):
    old, new = set(baseline['50']['owners']), set(sample['50']['owners'])
    preserving = old < new
    return dict(gained=sorted(new-old), lost=sorted(old-new), retained=sorted(old & new),
                net_owner_improvement=len(new)>len(old), owner_preserving_improvement=preserving,
                strong_joint_witness=preserving and sample['50']['fp'] <= baseline['50']['fp']
                and all(sample[k] <= baseline[k] for k in ('strict_repeats','parser_drops','cap')))


def best_key(card):
    return (-card['50']['tp'], card['50']['fp'], card['strict_repeats'],
            card['parser_drops'], card['complete_token_length'], card['seed'])


def case_reduction(baseline, samples):
    for sample in samples:
        sample['comparison'] = compare(sample, baseline)
    union = set().union(*(set(s['50']['owners']) for s in samples))
    old = set(baseline['50']['owners'])
    best = min(samples, key=best_key)
    oracle = min([baseline, *samples], key=best_key)
    return dict(greedy=baseline, samples=samples, sampled_best_seed=best['seed'],
                oracle_seed=oracle['seed'], union_owners=sorted(union),
                union_gained=sorted(union-old), union_lost=sorted(old-union),
                union_gain_without_single_tp_improvement=bool(union-old) and
                not any(s['comparison']['net_owner_improvement'] for s in samples))


def geometry_diagnostics(row, sample_row, baseline, sample):
    gt = _gt_objects(row, row_id=row['row_id'])
    ids = [str(o['object_id']) for o in row['gt']]
    before, _ = _pred_objects(row)
    after, _ = _pred_objects(sample_row)
    changed = sample['comparison']['gained'] + sample['comparison']['lost']
    result = []
    for owner in changed:
        index = ids.index(owner)
        category, box = gt[index]
        record = dict(owner=owner, change='gained' if owner in sample['comparison']['gained'] else 'lost',
                      gt_category=category, gt_box=list(box))
        for name, predictions, card in [('baseline', before, baseline), ('sample', after, sample)]:
            overlaps = [(iou_xyxy(box, b), j, c, b) for j,(c,b) in enumerate(predictions)]
            def maximum(values):
                if not values:
                    return dict(iou=0., pred_index=None, category=None, box=None)
                v,j,c,b = max(values, key=lambda v:(v[0], -v[1]))
                return dict(iou=v, pred_index=j, category=c, box=list(b))
            record[name] = dict(best_any_category=maximum(overlaps),
                best_same_category=maximum([v for v in overlaps if v[2] == category]),
                selected_match=next((m for m in card['50']['matches'] if m['owner']==owner), None),
                eligible_same_category_predictions=[dict(iou=v,pred_index=j,box=list(b))
                    for v,j,c,b in overlaps if c==category and v>=.5])
        result.append(record)
    return result


def validate_parser(text, evidence, width, height):
    parsed = parse_compact_object_box_closed(text, row_id=evidence['row_id'],
        row_index=evidence['row_index'], image_width=width, image_height=height).to_artifact_dict()
    require(parsed == evidence, f"parser evidence changed: {evidence['row_id']}")


def validate_sample(row, action, group, meta, baseline, tokenizer, cap):
    body = row['generated_token_ids']
    require(body == action['generated_token_ids'] and digest(body) ==
            row['generated_token_ids_sha256'] == action['generated_token_ids_sha256'], 'sample token identity')
    require(row['stop_reason'] == action['stop_reason'] and row['stop_reason'] in ('im_end','length'), 'sample stop')
    complete = body + ([151645] if row['stop_reason']=='im_end' else [])
    require(151645 not in body and complete == action['action_token_ids'] and
            digest(complete)==action['action_token_ids_sha256'] and
            len(complete)==action['action_token_count'] and 0 < len(complete)<=cap and
            (row['stop_reason']!='length' or len(complete)==cap) and
            action['terminal_eos_included']==(row['stop_reason']=='im_end'), 'sample terminal/budget identity')
    require(tokenizer.decode(body, skip_special_tokens=False)==row['generated_text'] and
            hashlib.sha256(row['generated_text'].encode()).hexdigest()==action['generated_text_sha256'], 'sample token/text identity')
    require(row['prompt_token_ids']==meta['prompt_token_ids']==group['prompt_token_ids'] and
            digest(row['prompt_token_ids'])==row['prompt_token_ids_sha256']==
            meta['prompt_token_ids_sha256']==group['prompt_token_ids_sha256'], 'sample prompt identity')
    for key in ('executed_media_sha256','observed_image_grid_thw'):
        require(row[key]==group[key], f'sample {key}')
    require(meta['image_sha256']==group['image_content_sha256'] and
            meta['chat_text_sha256']==group['chat_text_sha256'] and
            Path(meta['image_path']).resolve()==Path(baseline['image_path']).resolve() and
            (meta['width'],meta['height'])==(baseline['image_width'],baseline['image_height']), 'sample media metadata')
    evidence = row['predictions']
    require(digest(evidence)==action['parser_evidence_sha256'], 'sample parser hash')
    validate_parser(row['generated_text'], evidence, baseline['image_width'], baseline['image_height'])
    sample_row = dict(baseline, pred=evidence['predictions'], dropped_prediction_count=evidence['dropped_prediction_count'])
    card = score(sample_row, seed=row['seed'], length=len(complete), stop=row['stop_reason'])
    matching = action['matching']
    require(card['50']['owners']==matching['matched_owner_refs'] and
            card['50']['tp']==matching['matched_owner_count'] and
            len(baseline['gt'])==matching['annotated_owner_count']==group['annotated_owner_count'] and
            card['prediction_count']==matching['valid_prediction_count'] and
            card['invalid_predictions']==matching['invalid_prediction_count'] and
            card['parser_drops']==matching['parser_dropped_prediction_count'] and
            card['50']['recall']==action['reward'], 'source plan reward/count identity')
    return card, sample_row


def aggregate(cards):
    out = dict(outputs=len(cards))
    for key in ('prediction_count','parsed_prediction_count','invalid_predictions','parser_drops','strict_repeats','cap','complete_token_length'):
        out[key] = sum(c[key] for c in cards)
    out['stop_reasons'] = dict(Counter(c['stop_reason'] for c in cards))
    for threshold in ('50','60','80'):
        metric = {k:sum(c[threshold][k] for c in cards) for k in ('tp','fp','fn')}
        tp,fp,fn = (metric[k] for k in ('tp','fp','fn'))
        metric.update(f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., recall=tp/(tp+fn) if tp+fn else 0.)
        out[threshold]=metric
    return out


def run(plan_path, greedy_root, output_dir, subset=None):
    from tokenizers import Tokenizer
    plan_path, greedy_root, output_dir = map(Path, (plan_path,greedy_root,output_dir))
    sources = {}
    def read(path, *, jsonl=False, expected=None):
        path=Path(path); sha=file_hash(path)
        require(expected is None or sha==expected, f'source hash changed: {path}')
        sources[str(path.resolve())]=sha
        return rows(path) if jsonl else json.loads(path.read_text())
    plan=read(plan_path)
    require(digest({k:v for k,v in plan.items() if k!='content_sha256'})==plan['content_sha256'], 'plan content hash')
    require(plan['round']==1 and plan['arm']=='rloo' and plan['population']['image_count']==256 and
            plan['population']['k']==4 and len(plan['population']['groups'])==256, 'frozen population')
    policy=plan['sampling']; cap=policy['max_new_tokens']; seeds=policy['seeds']
    require(cap==3084 and len(set(seeds))==4 and policy['temperature']==1 and policy['top_p']==1
            and policy['top_k']==0 and policy['repetition_penalty']==1 and policy['raw_softmax'] is True
            and policy['use_model_defaults'] is False and policy['terminal_token_id']==151645, 'frozen sampling policy')
    train_source=plan['sources']['train_jsonl']
    train=read(train_source['path'], jsonl=True, expected=train_source['sha256'])
    train=indexed(train,'image_id')
    manifest=read(greedy_root/'run_manifest.json')
    identity=manifest['model_identity']
    require(identity['base']['path']==plan['model']['base_model_path'] and
            identity['adapter']['adapter_path']==plan['model']['current_adapter']['root'] and
            identity['embedding_delta']['identity']['delta_path']==plan['model']['source_embedding']['root'] and
            identity['embedding_delta']['identity']['metadata']==plan['model']['source_embedding']['semantic_identity'],
            'plan/greedy checkpoint or embedding identity')
    require(identity['adapter']['enabled'] is True and not identity['adapter']['merged_adapters'], 'adapter execution state')
    summary=read(greedy_root/'summary.json')
    raw=indexed(read(greedy_root/'gt_vs_pred.jsonl',jsonl=True),'row_id')
    scored=indexed(read(greedy_root/'gt_vs_pred_scored.jsonl',jsonl=True),'row_id')
    images=indexed(read(greedy_root/'image_plan.jsonl',jsonl=True),'row_id')
    prompts=indexed(manifest['prompt_trace'],'row_id')
    traces=defaultdict(list)
    for token in read(greedy_root/'pred_token_trace.jsonl',jsonl=True):
        if token['trace_type']=='generated_token': traces[token['row_id']].append(token)
    groups=indexed(plan['population']['groups'],'example_id')
    require(set(raw)==set(scored)==set(images)==set(prompts)==set(traces)==set(groups) and
            len(raw)==256 and set(train)=={g['image_id'] for g in groups.values()}, 'population identity')
    generation=manifest['generation_policy']
    require(generation['do_sample'] is False and generation['repetition_penalty']==1 and
            generation['max_new_tokens']==cap and generation['top_p']==1, 'greedy policy mismatch')
    tokenizer_path=Path(plan['model']['base_model_path'])/'tokenizer.json'
    require(file_hash(tokenizer_path)==manifest['frontend_identity']['tokenizer_sha256'], 'tokenizer hash')
    sources[str(tokenizer_path)]=file_hash(tokenizer_path)
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    cells={}
    require(len(plan['sources']['rollout_artifacts'])==8, 'shard count')
    for source in plan['sources']['rollout_artifacts']:
        shard=read(source['path'],expected=source['sha256'])
        require(shard['rollout_count']==128 and len(shard['rollouts'])==128, 'shard cells')
        for key in ('max_new_tokens','temperature','top_p','top_k','repetition_penalty','raw_softmax','use_model_defaults','seeds'):
            require(shard['config'][key]==policy[key], f'shard sampling policy {key}')
        require(digest(shard['model_identity'])==policy['backend_receipt_sha256'], 'bank backend identity')
        for key in ('model_identity','processor_identity','tokenizer_identity'):
            require(shard['model_identity'][key]==manifest['backend_session'][key], f'greedy/bank {key}')
        require({k:v for k,v in shard['model_identity']['effective_settings'].items() if k!='performance'}==
                {k:v for k,v in manifest['backend_session']['effective_settings'].items() if k!='performance'},
                'greedy/bank effective execution settings')
        for row in shard['rollouts']:
            key=(row['example_id'],row['seed'])
            require(key not in cells, 'duplicate bank cell')
            cells[key]=(row,shard['prompt_metadata'][row['example_id']])
    require(set(cells)=={(eid,seed) for eid in groups for seed in seeds}, 'missing/extra bank cells')
    selected=set(groups) if subset is None else set(subset)
    require(selected and selected<=set(groups) and (subset is None or len(selected)==len(subset)), 'declared subset')
    cases=[]; all_owner_ids=[]
    for eid, group in groups.items():
        if eid not in selected: continue
        row=raw[eid]; t=train[group['image_id']]
        require((row['image_width'],row['image_height'])==(t['width'],t['height']), 'GT dimensions')
        native_gt=[dict(description=o['desc'],bbox=[parse_coord_token(v,field='GT') for v in o['bbox_2d']],
                        object_id=str(o['coco_ann_id'])) for o in t['objects']]
        require(_gt_objects(dict(row,gt=native_gt),row_id=eid)==_gt_objects(row,row_id=eid) and
                [o['object_id'] for o in native_gt]==[str(o['object_id']) for o in row['gt']], 'GT geometry/owner identity')
        all_owner_ids.extend(o['object_id'] for o in native_gt)
        require(Path(row['image_path']).resolve()==Path(group['image_path']).resolve()==
                (Path(train_source['path']).parent/t['images'][0]).resolve(), 'image path identity')
        require(file_hash(row['image_path'])==group['image_content_sha256'], 'image bytes identity')
        for key in ('executed_media_sha256','observed_image_grid_thw','image_content_sha256'):
            require(images[eid][key]==group[key], f'greedy image {key}')
        require(images[eid]['status']=='ok' and prompts[eid]['prompt_token_parity']=='verified' and
                prompts[eid]['backend_executed_prompt_token_ids_sha256']==group['prompt_token_ids_sha256'] and
                prompts[eid]['backend_executed_prompt_token_count']==len(group['prompt_token_ids']), 'greedy prompt identity')
        evidence={k:row[k] for k in ('row_id','row_index','parser_id','parser_policy','metric_bearing','parse_status',
                  'valid_prediction_count','dropped_prediction_count','dropped_predictions')}
        evidence['predictions']=row['pred']
        validate_parser(row['raw_decode_text'],evidence,row['image_width'],row['image_height'])
        tokens=sorted(traces[eid], key=lambda t:t['generated_step_index'])
        require([t['generated_step_index'] for t in tokens]==list(range(len(tokens))), 'greedy token trace coverage')
        # Native batched traces retain trailing batch padding, not generated actions.
        pad_count=sum(t['is_pad'] for t in tokens)
        if pad_count:
            require(all(t['is_pad'] and t['token_id']==151643 and not t['is_stop'] for t in tokens[-pad_count:])
                    and not any(t['is_pad'] for t in tokens[:-pad_count]), 'nonterminal greedy padding')
            tokens=tokens[:-pad_count]
        ids=[t['token_id'] for t in tokens]; stop=row['decode_stop_reason']
        require(tokenizer.decode(ids,skip_special_tokens=False)==row['raw_decode_text'] and
                ''.join(t['token_text'] for t in tokens)==row['raw_decode_text'], 'greedy token/text identity')
        require(stop in ('im_end','length') and 0<len(ids)<=cap and
                (stop!='length' or len(ids)==cap) and
                (ids[-1]==151645)==(stop=='im_end') and
                sum(t['is_stop'] for t in tokens)==int(stop=='im_end'), 'greedy terminal/budget identity')
        base=score(row,seed=-1,length=len(ids),stop=stop)
        base['batch_padding_trace_tokens']=pad_count
        require(_pred_objects(scored[eid])==_pred_objects(row) and scored[eid]['gt']==row['gt'], 'raw/scored predictions or GT')
        require(len(group['actions'])==4 and {a['seed'] for a in group['actions']}==set(seeds), 'plan action coverage')
        require(group['reward_values']==[a['reward'] for a in group['actions']], 'plan reward values')
        samples=[]; sample_rows=[]
        for action in group['actions']:
            source_row,meta=cells[(eid,action['seed'])]
            card,sample_row=validate_sample(source_row,action,group,meta,row,tokenizer,cap)
            samples.append(card);sample_rows.append(sample_row)
        case=case_reduction(base,samples)
        for card,sample_row in zip(samples,sample_rows):
            card['changed_owner_geometry']=geometry_diagnostics(row,sample_row,base,card)
        case.update(example_id=eid,image_id=group['image_id'],image_path=row['image_path'],
                    gt_sha256=digest(row['gt']),gt_owner_ids=[str(o['object_id']) for o in row['gt']])
        cases.append(case)
    if subset is None:
        require(summary['raw_row_count']==len(cases) and summary['dropped_prediction_count']==
                sum(c['greedy']['parser_drops'] for c in cases) and summary['truncated_decode_count']==
                sum(c['greedy']['cap'] for c in cases) and summary['scoreable_prediction_count']==
                sum(c['greedy']['prediction_count'] for c in cases), 'greedy summary counts')
    result=dict(schema_version='natural_candidate_opportunity.v1',scope='full256xK4' if subset is None else 'declared_cpu_subset',
                selected_example_ids=[c['example_id'] for c in cases],images=len(cases),samples=4*len(cases),
                owner_ids_globally_unique=len(all_owner_ids)==len(set(all_owner_ids)),
                owner_identity_sha256=digest([(c['example_id'],c['gt_owner_ids']) for c in cases]),
                source_files=sources,analysis_code_sha256=file_hash(__file__),
                analysis_dependency_sha256={str(Path(p).resolve()):file_hash(p) for p in
                    ('probes/dora_owner_learning/reward_rows.py','src/eval/assignment.py',
                     'src/inference/parsing.py','src/data/geometry.py','src/artifacts/json_values.py')},
                token_accounting='Sample bodies exclude observed EOS; append only im_end. Greedy trace trailing batch pads excluded, explicit counts retained.',
                prediction_accounting='prediction_count is native reward-projectable; parsed_prediction_count includes invalid projected categories/boxes; FP includes those invalid predictions.',
                source_plan_content_sha256=plan['content_sha256'],
                greedy=aggregate([c['greedy'] for c in cases]),
                all_samples=aggregate([s for c in cases for s in c['samples']]))
    for label,field in [('sampled_best','sampled_best_seed'),('oracle_with_greedy','oracle_seed')]:
        chosen=[next(s for s in [c['greedy'],*c['samples']] if s['seed']==c[field]) for c in cases]
        result[label]=aggregate(chosen)
        changes=[compare(s,c['greedy']) for s,c in zip(chosen,cases)]
        result[label]['owner_changes']={k:sum(len(change[k]) for change in changes) for k in ('gained','lost','retained')}
    result['witnesses']={level:dict(images=sum(any(s['comparison'][level] for s in c['samples']) for c in cases),
                                  samples=sum(s['comparison'][level] for c in cases for s in c['samples']))
                         for level in ('net_owner_improvement','owner_preserving_improvement','strong_joint_witness')}
    for level,value in result['witnesses'].items():
        witness_cards=[s for c in cases for s in c['samples'] if s['comparison'][level]]
        value['metrics']=aggregate(witness_cards)
        value['owner_changes']={k:sum(len(s['comparison'][k]) for s in witness_cards) for k in ('gained','lost','retained')}
    if subset is None:
        require([result['greedy'][t]['tp'] for t in ('50','60','80')]==[1259,1190,908]
                and result['greedy']['50']['tp']+result['greedy']['50']['fn']==1955,
                'recorded Source baseline TP/GT mismatch')
        result['recorded_baseline_crosscheck']='256 images, 1955 GT; TP50/60/80=1259/1190/908 reproduced'
    result['union']=dict(owners=sum(len(c['union_owners']) for c in cases),
        gained=sum(len(c['union_gained']) for c in cases),lost=sum(len(c['union_lost']) for c in cases),
        images_with_gain=sum(bool(c['union_gained']) for c in cases),
        images_gain_without_single_tp_improvement=sum(c['union_gain_without_single_tp_improvement'] for c in cases))
    output_dir.mkdir(parents=True,exist_ok=True)
    publish_json_exclusive(output_dir/'cases.json',cases)
    result['cases_sha256']=file_hash(output_dir/'cases.json')
    publish_json_exclusive(output_dir/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--greedy-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--subset',nargs='+',help='Declared CPU fixture example IDs; never a full-population result')
    args=parser.parse_args()
    result=run(args.plan,args.greedy_root,args.output_dir,args.subset)
    print(json.dumps({k:result[k] for k in ('scope','images','samples','witnesses','union','cases_sha256')},sort_keys=True))


if __name__=='__main__':
    main()
