"""CPU-only matched natural-greedy outcome after the registered RLOO update."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from src.artifacts import publish_json_exclusive
from .candidate_opportunity import (
    aggregate, compare, digest, file_hash, geometry_diagnostics, indexed,
    require, rows, score, validate_parser,
)
from .reward_rows import _pred_objects

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
SOURCE_ROOT = ROOT/'2026-09-05-sft256-dev128-baseline/natural-eval-v1/qwen3-vl-2b-sft256-source-train256-natural-v1'
CANDIDATE_ROOT = ROOT/'2026-09-09-natural-candidate-opportunity/full-v2'
UPDATE_ROOT = ROOT/'2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/rloo'
LEVELS = ('net_owner_improvement', 'owner_preserving_improvement', 'strong_joint_witness')


def validate_identity(post, source, receipt):
    """Compare semantic execution identities, not historical schema/runtime blobs."""
    require(post['terminal_status']=='completed', 'post inference is incomplete')
    model, old = post['model_identity'], source['model_identity']
    require(model['base']==old['base'] and model['qwen']==old['qwen'], 'base model identity')
    adapter = model['adapter']
    require(adapter['adapter_path']==receipt['saved_adapter']['root'], 'wrong post checkpoint')
    require(adapter['enabled'] is True and adapter['adapter_type']=='dora' and
            not adapter['merged_adapters'] and adapter['status']=='validated', 'post adapter state')
    semantic=receipt['saved_adapter']['semantic_identity']
    payload=adapter['adapter_payload_evidence']
    require(payload['rank']==semantic['r'] and payload['alpha']==semantic['lora_alpha'] and
            payload['key_count']==semantic['tensor_key_count'] and
            payload['lora_A_count']==semantic['lora_A_count'] and
            payload['lora_B_count']==semantic['lora_B_count'] and
            payload['lora_magnitude_vector_count']==semantic['lora_magnitude_vector_count'],
            'adapter receipt semantics')
    delta=model['embedding_delta']['identity']
    require(delta==old['embedding_delta']['identity'] and
            delta['delta_path']==receipt['source_embedding']['root'] and
            delta['metadata']==receipt['source_embedding']['semantic_identity'], 'embedding identity')
    require(post['dataset_identity']==source['dataset_identity'], 'dataset identity')
    for key in ('parser_policy','template_identity','processor_identity','tokenizer_identity'):
        require(post[key]==source[key], f'{key} identity')
    for key in ('do_sample','repetition_penalty','max_new_tokens','stop_policy','temperature','top_p','batch_size'):
        require(post['generation_policy'][key]==source['generation_policy'][key], f'greedy policy {key}')
    require(post['generation_policy']['do_sample'] is False and
            post['generation_policy']['max_new_tokens']==3084, 'natural greedy policy')
    # Ignore timing/performance counters and schema additions; enforce executed numerics.
    for key in ('observed_attn_implementation','observed_model_dtype','text_padding_side'):
        require(post['backend_session']['effective_settings'][key]==
                source['backend_session']['effective_settings'][key], f'execution {key}')
    require(post['frontend_identity']['tokenizer_sha256']==source['frontend_identity']['tokenizer_sha256'],
            'tokenizer content identity')


def validate_case_identity(post, source, image, source_image, prompt, source_prompt):
    require(post['row_id']==source['row_id'] and post['gt']==source['gt'] and
            (post['image_width'],post['image_height'],post['image_path'])==
            (source['image_width'],source['image_height'],source['image_path']), 'GT/image identity')
    for key in ('executed_media_sha256','observed_image_grid_thw','image_content_sha256'):
        require(image[key]==source_image[key], f'media identity {key}')
    require(image['status']=='ok' and prompt['prompt_token_parity']=='verified', 'prompt/media execution unverified')
    for key in ('backend_executed_prompt_token_ids_sha256','backend_executed_prompt_token_count'):
        require(prompt[key]==source_prompt[key], f'prompt identity {key}')


def trace_card(row, tokens, tokenizer):
    evidence={k:row[k] for k in ('row_id','row_index','parser_id','parser_policy','metric_bearing','parse_status',
                  'valid_prediction_count','dropped_prediction_count','dropped_predictions')}
    evidence['predictions']=row['pred']
    validate_parser(row['raw_decode_text'],evidence,row['image_width'],row['image_height'])
    tokens=sorted(tokens,key=lambda t:t['generated_step_index'])
    require([t['generated_step_index'] for t in tokens]==list(range(len(tokens))), 'trace coverage')
    pads=sum(t['is_pad'] for t in tokens)
    if pads:
        require(all(t['is_pad'] and t['token_id']==151643 and not t['is_stop'] for t in tokens[-pads:])
                and not any(t['is_pad'] for t in tokens[:-pads]), 'trace padding')
        tokens=tokens[:-pads]
    ids=[t['token_id'] for t in tokens]
    stop=row['decode_stop_reason']
    require(tokenizer.decode(ids,skip_special_tokens=False)==row['raw_decode_text'] and
            ''.join(t['token_text'] for t in tokens)==row['raw_decode_text'], 'trace token/text identity')
    require(stop in ('im_end','length') and 0<len(ids)<=3084 and
            (stop!='length' or len(ids)==3084) and (ids[-1]==151645)==(stop=='im_end') and
            sum(t['is_stop'] for t in tokens)==int(stop=='im_end'), 'trace stop/budget identity')
    card=score(row,seed=-1,length=len(ids),stop=stop)
    card['batch_padding_trace_tokens']=pads
    return card


def owner_change(post, source, threshold):
    new,old=set(post[threshold]['owners']),set(source[threshold]['owners'])
    return dict(gained=sorted(new-old),lost=sorted(old-new),retained=sorted(new&old),
                retention=len(new&old)/len(old) if old else None)


def witness_join(candidate, post, actions):
    """Each row remains one original complete sample, never an attainable union."""
    result=[]
    post_owners=set(post['50']['owners'])
    for sample in candidate['samples']:
        comp=sample['comparison']
        if not any(comp[level] for level in LEVELS):
            continue
        action=actions[sample['seed']]
        require(sample['50']['owners']==action['matching']['matched_owner_refs'], 'witness source matching')
        advantage=action['advantage']
        gained=set(comp['gained'])
        result.append(dict(seed=sample['seed'],levels={level:comp[level] for level in LEVELS},
            advantage=advantage,advantage_sign='positive' if advantage>0 else 'negative' if advantage<0 else 'zero',
            sample_gained_owners=sorted(gained),appears_in_post=sorted(gained&post_owners),
            absent_in_post=sorted(gained-post_owners),
            all_sample_owners_present=set(sample['50']['owners'])<=post_owners,
            complete_sample_owner_set_reproduced=set(sample['50']['owners'])==post_owners))
    return result


def run(post_root, output_dir, *, fixture_round4=False):
    from tokenizers import Tokenizer
    post_root, output_dir=Path(post_root),Path(output_dir)
    sources={}
    def read(path, *, jsonl=False, expected=None):
        path=Path(path); sha=file_hash(path)
        require(expected is None or sha==expected, f'source hash changed: {path}')
        sources[str(path.resolve())]=sha
        return rows(path) if jsonl else json.loads(path.read_text())
    candidate_summary=read(CANDIDATE_ROOT/'summary.json')
    candidate_cases=indexed(read(CANDIDATE_ROOT/'cases.json',expected=candidate_summary['cases_sha256']),'example_id')
    require(candidate_summary['scope']=='full256xK4' and len(candidate_cases)==256, 'fixed candidate population')
    plan_path=UPDATE_ROOT/'round-1/plan.json'
    plan=read(plan_path,expected=candidate_summary['source_files'][str(plan_path)])
    groups=indexed(plan['population']['groups'],'example_id')
    round_number=4 if fixture_round4 else 1
    receipt_path=UPDATE_ROOT/f'round-{round_number}/update/receipt.json'
    receipt=read(receipt_path)
    require(receipt['round']==round_number and receipt['arm']=='rloo' and
            receipt['mechanical_status']=='MECHANICALLY_VALID', 'registered update receipt')
    if not fixture_round4:
        require(receipt['plan']['sha256']==file_hash(plan_path) and
                receipt['plan']['content_sha256']==plan['content_sha256'], 'round1 update/source bank lineage')
    def source_read(name, jsonl=False):
        path=SOURCE_ROOT/name
        return read(path,jsonl=jsonl,expected=candidate_summary['source_files'][str(path)])
    source_manifest=source_read('run_manifest.json')
    source_rows=indexed(source_read('gt_vs_pred.jsonl',True),'row_id')
    source_images=indexed(source_read('image_plan.jsonl',True),'row_id')
    source_prompts=indexed(source_manifest['prompt_trace'],'row_id')
    manifest=read(post_root/'run_manifest.json')
    validate_identity(manifest,source_manifest,receipt)
    raw=indexed(read(post_root/'gt_vs_pred.jsonl',jsonl=True),'row_id')
    scored=indexed(read(post_root/'gt_vs_pred_scored.jsonl',jsonl=True),'row_id')
    images=indexed(read(post_root/'image_plan.jsonl',jsonl=True),'row_id')
    prompts=indexed(manifest['prompt_trace'],'row_id')
    summary=read(post_root/'summary.json')
    traces=defaultdict(list)
    for token in read(post_root/'pred_token_trace.jsonl',jsonl=True):
        if token['trace_type']=='generated_token': traces[token['row_id']].append(token)
    require(set(raw)==set(scored)==set(images)==set(prompts)==set(traces)==set(candidate_cases)==set(source_rows),
            'missing/extra/duplicate population cells')
    tokenizer_path=Path(source_manifest['model_identity']['base']['path'])/'tokenizer.json'
    require(file_hash(tokenizer_path)==source_manifest['frontend_identity']['tokenizer_sha256'], 'tokenizer bytes')
    sources[str(tokenizer_path)]=file_hash(tokenizer_path)
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    cases=[]
    for eid,candidate in candidate_cases.items():
        row,old=raw[eid],source_rows[eid]
        validate_case_identity(row,old,images[eid],source_images[eid],prompts[eid],source_prompts[eid])
        require(digest(old['gt'])==candidate['gt_sha256'], 'candidate GT binding')
        card=trace_card(row,traces[eid],tokenizer)
        require(_pred_objects(row)==_pred_objects(scored[eid]) and row['gt']==scored[eid]['gt'], 'raw/scored identity')
        baseline=candidate['greedy']
        require(score(old,seed=-1,length=baseline['complete_token_length'],stop=baseline['stop_reason'])==
                {k:v for k,v in baseline.items() if k!='batch_padding_trace_tokens'}, 'source score recomputation')
        card['comparison']=compare(card,baseline)
        cases.append(dict(example_id=eid,image_id=candidate['image_id'],source=baseline,post=card,
            owner_changes={t:owner_change(card,baseline,t) for t in ('50','60','80')},
            changed_owner_geometry=geometry_diagnostics(old,row,baseline,card),
            witness_realization=witness_join(candidate,card,{a['seed']:a for a in groups[eid]['actions']})))
    post=aggregate([c['post'] for c in cases]); source=aggregate([c['source'] for c in cases])
    require(summary['raw_row_count']==256 and summary['dropped_prediction_count']==post['parser_drops'] and
            summary['truncated_decode_count']==post['cap'] and summary['scoreable_prediction_count']==post['prediction_count'],
            'post inference summary count identity')
    require([source[t]['tp'] for t in ('50','60','80')]==[1259,1190,908], 'Source baseline counts')
    if fixture_round4:
        require([post[t]['tp'] for t in ('50','60','80')]==[1278,1209,923], 'historical round4 fixture counts')
    outcome=dict(schema_version='round1_greedy_realization.v1',
        scope='declared_historical_round4_cpu_fixture' if fixture_round4 else 'round1_train256_natural_greedy',
        round=round_number,images=len(cases),source=source,post=post,source_files=sources,
        code_sha256=file_hash(__file__),reused_consumer_sha256=file_hash(Path(__file__).with_name('candidate_opportunity.py')),
        checkpoint_binding='Exact persisted adapter path and semantic payload vs update receipt; no model weights read by this consumer.',
        limitations='One greedy outcome, not trajectory likelihood change. Witness pools below are descriptive owner-level joins, not single achievable outputs.')
    outcome['owner_changes']={}
    for t in ('50','60','80'):
        change={k:sum(len(c['owner_changes'][t][k]) for c in cases) for k in ('gained','lost','retained')}
        change['retention']=change['retained']/source[t]['tp']
        outcome['owner_changes'][t]=change
    outcome['delta']={t:{k:post[t][k]-source[t][k] for k in ('tp','fp','fn','f1','recall')} for t in ('50','60','80')}
    outcome['witness_realization']={}
    for level in LEVELS:
        strata={}
        for sign in ('all','positive','zero','negative'):
            selected=[(c['example_id'],w) for c in cases for w in c['witness_realization']
                      if w['levels'][level] and (sign=='all' or w['advantage_sign']==sign)]
            pool={(eid,owner) for eid,w in selected for owner in w['sample_gained_owners']}
            present={(eid,owner) for eid,w in selected for owner in w['appears_in_post']}
            strata[sign]=dict(samples=len(selected),images=len({eid for eid,_ in selected}),
                gained_owner_instances=sum(len(w['sample_gained_owners']) for _,w in selected),
                present_owner_instances=sum(len(w['appears_in_post']) for _,w in selected),
                distinct_gained_owner_pairs=len(pool),distinct_present_owner_pairs=len(present),
                samples_all_gained_owners_present=sum(not w['absent_in_post'] for _,w in selected),
                samples_all_sample_owners_present=sum(w['all_sample_owners_present'] for _,w in selected))
        outcome['witness_realization'][level]=strata
    output_dir.mkdir(parents=True,exist_ok=True)
    publish_json_exclusive(output_dir/'cases.json',cases)
    outcome['cases_sha256']=file_hash(output_dir/'cases.json')
    publish_json_exclusive(output_dir/'summary.json',outcome)
    return outcome


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--post-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fixture-round4',action='store_true',help='Historical consumer test only, never a round1 result')
    args=parser.parse_args()
    result=run(args.post_root,args.output_dir,fixture_round4=args.fixture_round4)
    print(json.dumps({k:result[k] for k in ('scope','images','delta','owner_changes','cases_sha256')},sort_keys=True))


if __name__=='__main__':
    main()
