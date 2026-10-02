"""Finite matched noisy-positive update; protocol owned by the 2026-09-27 unit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

from probes.hidden_human_recovery import (CHECKPOINT, load, write, digest, canonical, validate_visible,
                                         native_request, request_plan, candidates, original_coverage)
from src.eval.saved_rows import iou_xyxy

ROOT = Path(__file__).resolve().parents[1] / 'outputs/research/physical-fn-recovery/2026-09-27/iterative-positive-01'
OLD = Path('/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-09-26')
VISIBLE = OLD / 'preparation-v3/acquisition/visible.json'
POLICY = OLD / 'smoke-policy-01.json'
REPLAY = Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/train.coord.jsonl')
MAX_LENGTH = 16000


def stable(value):
    return hashlib.sha256(value.encode()).hexdigest()


def diverse_selection(bank):
    selected, dispositions = [], {}
    eligible = [c for c in bank['candidates'] if c['cohort'] == 'human13'
                and not c['quantization_error'] and not c['visible_conflict_ids']
                and not c['visible_same_category_ids']]
    for image_id in sorted({c['image_id'] for c in eligible}):
        rows = sorted((c for c in eligible if c['image_id'] == image_id),
                      key=lambda c: (-c['distinct_other_query_support'], stable('926-positive-v1:' + c['representative_prediction_id'])))
        kept, seen = [], set()
        for c in rows:
            key = (c['description'], tuple(c['quantized_full_image_bins']))
            if key in seen:
                reason = 'quantized_literal_duplicate'
            elif any(c['description'] == k['description'] and iou_xyxy(c['quantized_full_image_bins'], k['quantized_full_image_bins']) >= .8 for k in kept):
                reason = 'diversity_withheld'
            elif len(kept) == 4:
                reason = 'unused_budget'
            else:
                reason = 'selected_noisy_positive'
                kept.append(c)
            seen.add(key)
            dispositions[c['candidate_id']] = reason
        selected.extend(kept)
    return dict(selected=selected, dispositions=dispositions)


def novel_selection(bank, ordinary_predictions):
    """One prediction-only eligibility change; default diversity policy is unchanged."""
    witnesses={}
    for c in bank['candidates']:
        if c['cohort']!='human13' or c['quantization_error'] or c['visible_conflict_ids'] or c['visible_same_category_ids']:
            continue
        for prediction in ordinary_predictions:
            if prediction['image_id']!=c['image_id'] or prediction['description']!=c['description']:
                continue
            fractional=iou_xyxy(c['coord_bins_1000'],prediction['coord_bins_1000'])
            quantized=iou_xyxy(c['quantized_full_image_bins'],prediction['coord_bins_1000'])
            if max(fractional,quantized)>=.5:
                witnesses[c['candidate_id']]=dict(prediction_id=prediction['prediction_id'],fractional_iou=fractional,quantized_iou=quantized)
                break
    remaining=[c for c in bank['candidates'] if c['candidate_id'] not in witnesses]
    selection=diverse_selection({'candidates':remaining})
    selection['dispositions']={**{c['candidate_id']:c['disposition'] for c in bank['candidates']},
                               **selection['dispositions'],**{i:'ordinary_greedy_covered' for i in witnesses}}
    selection['ordinary_greedy_witnesses']=witnesses
    return selection


def prepare_novel(output_root):
    release=load(output_root/'lead-release-01.json')
    assert digest(output_root/'lead-release-01.json')=='b127483a01fc435b87a18ade1d0f4f82d40545f4b23e015ea2267bad6cbb0ae5'
    for path,expected in release['inputs'].items():
        assert digest(path)==expected,path  # Hash diagnostics as evidence; never deserialize them here.
    plan=load(ROOT/'learning-plan.json')
    bank=load(ROOT/'candidates.json')
    ordinary,_=candidates(read_evaluation(ROOT/'round-01/evaluation-zero'))
    plan['selection']=novel_selection(bank,ordinary)
    plan['inputs'].update({str(ROOT/'learning-plan.json'):digest(ROOT/'learning-plan.json'),
                           str(ROOT/'round-01/evaluation-zero/frozen.json'):digest(ROOT/'round-01/evaluation-zero/frozen.json')})
    plan['candidate_policy']='ordinary-greedy-novel: fractional OR quantized same-category IoU>=.5 withheld before unchanged ranking/diversity'
    assert not (output_root/'learning-plan.json').exists()
    write(output_root/'learning-plan.json',plan)
    return plan


def make_schedule(human_ids, replay_ids, steps=16):
    assert len(human_ids) == 13 and len(replay_ids) == 128
    return [dict(step=s+1, human=[human_ids[(s*8+j) % 13] for j in range(8)],
                 replay=[replay_ids[(s*16+j) % 128] for j in range(16)]) for s in range(steps)]


def rank_examples(step, rank, world=8, treatment=False):
    assert world == 8 and 0 <= rank < world
    result = [('visible', step['human'][rank], 1/3),
              ('replay', step['replay'][rank], 1/3),
              ('replay', step['replay'][rank+8], 1/3)]
    if treatment:
        result.append(('pseudo', step['human'][rank], .05))
    return result  # DDP averages eight ranks: common /24, pseudo .05/8.


def prepare():
    release = load(ROOT/'lead-release-01.json')
    for path, expected in release['sha256'].items():
        assert digest(path) == expected, path
    assert digest(REPLAY) == 'ecf07a40856ee96a92c9093139abe24facfa600136e04f3c1bcbd02e039aaad1'
    bank, visible = load(ROOT/'candidates.json'), load(VISIBLE)
    validate_visible(visible)
    selection = diverse_selection(bank)
    excluded = {r['image_id'] for r in visible}
    import heapq
    # One metadata scan; retain only the smallest 128 identities, then those rows.
    ids = []
    with REPLAY.open() as f:
        for line in f:
            i = int(json.loads(line)['image_id'])
            if i not in excluded:
                ids.append(i)
    chosen = heapq.nsmallest(128, ids, key=lambda i: stable('927-replay-v1:'+str(i)))
    assert len(set(chosen)) == 128
    rows = {}
    from src.data.examples import raw_example_from_jsonl_row
    with REPLAY.open() as f:
        for number, line in enumerate(f, 1):
            row = json.loads(line)
            if row['image_id'] in chosen:
                raw = raw_example_from_jsonl_row(row, jsonl_path=REPLAY, row_number=number, raw_line=line)
                rows[str(row['image_id'])] = dict(row=row, row_number=number, raw_line=line,
                    image_path=str(raw.image.path), image_sha256=digest(raw.image.path))
    human_ids = [r['image_id'] for r in bank['images'] if r['cohort']=='human13']
    for row in visible:
        assert digest(row['image_path']) == row['image_sha256']
    plan = dict(selection=selection, human_ids=human_ids, replay_ids=chosen, replay_rows=rows,
                schedule=make_schedule(human_ids, chosen), visible=visible,
                inputs={str(p):digest(p) for p in (ROOT/'candidates.json', VISIBLE, POLICY, REPLAY)},
                precision='base BF16 load; adapter/deltas FP32 master; eval base BF16 then FP32; no frozen-reference equivalence assumed')
    write(ROOT/'learning-plan.json', plan)
    return plan


def raw_for(plan, branch, image_id):
    from src.data.examples import (raw_example_from_jsonl_row, RawExample, RawObject, ImageRef, SourceProvenance)
    if branch == 'replay':
        r = plan['replay_rows'][str(image_id)]
        assert digest(r['image_path']) == r['image_sha256']
        return raw_example_from_jsonl_row(r['row'], jsonl_path=REPLAY, row_number=r['row_number'], raw_line=r['raw_line']), None
    r = next(r for r in plan['visible'] if r['image_id']==image_id)
    assert digest(r['image_path']) == r['image_sha256']
    objects = [RawObject('visible:'+str(o['coco_ann_id']), o['desc'], tuple(o['bbox_2d']), {}) for o in r['objects']]
    selected_ids = {o.object_id for o in objects}
    if branch == 'pseudo':
        additions = [RawObject('pseudo:'+c['candidate_id'], c['description'], tuple(c['quantized_full_image_bins']), {})
                     for c in plan['selection']['selected'] if c['image_id']==image_id]
        objects.extend(additions)
        selected_ids = {o.object_id for o in additions}
    objects.sort(key=lambda o:(o.bbox[0],o.bbox[1],o.object_id))
    raw = RawExample(str(image_id), ImageRef(r['image_path'],Path(r['image_path']),r['width'],r['height'],{}), tuple(objects),{},
                     SourceProvenance(VISIBLE, 1, digest(VISIBLE), 'prediction_visible_only'))
    return raw, selected_ids


def encode(raw, selected_ids, qwen):
    from src.config.models import TemplateConfig, ProcessorConfig
    from src.templates.renderer import render_example
    from src.qwen.encoding import encode_rendered_example
    from src.packing.planner import plan_packed_sequences
    from src.packing.supervision import build_packed_supervision
    from src.supervision.tokens import build_token_sequence_from_packed_supervision
    config = TemplateConfig(object_field_order='desc_first', object_ordering='geo_sorted_xy',
                            assistant_format='object_box_closed', prompt=load(POLICY)['prompt'])
    rendered = render_example(raw, config)
    encoded = encode_rendered_example(raw, rendered, components=qwen,
                processor_config=ProcessorConfig(max_raw_pixels=4000000,max_merged_visual_tokens=4096),
                global_max_length=MAX_LENGTH,materialize_image_pixels=False)
    pack, = plan_packed_sequences([encoded],global_max_length=MAX_LENGTH)
    supervision = build_packed_supervision([pack],[encoded])
    atoms = supervision.atoms if selected_ids is None else tuple(a for a in supervision.atoms if a.object_id in selected_ids)
    sequence = build_token_sequence_from_packed_supervision(pack,atoms)
    if selected_ids is not None:
        assert all(a.object_id in selected_ids and a.token_type != 'eos' for a in sequence.atoms)
        for oid in selected_ids:
            own = [a for a in sequence.atoms if a.object_id==oid]
            assert {a.field for a in own} >= {'object_ref_start','object_ref_end','box_start','box_end','bbox[0]','bbox[1]','bbox[2]','bbox[3]'}
    return encoded, sequence


def image_loss(logits, sequence, vocab, positions):
    from src.losses.context import LossContext
    from src.losses.base_ce import BaseTokenCE
    from src.losses.token_type_gate import TokenTypeGateLoss
    from src.losses.conditional_order_gate import ConditionalOrderGateLoss
    if not sequence.atoms:
        return logits.sum()*0, {'ce':0.,'type':0.,'order':0.}
    context = LossContext(logits,sequence,vocab,tuple(positions))
    ce = BaseTokenCE().per_atom_loss(context).mean()
    gate = TokenTypeGateLoss().per_atom_loss(context).mean()
    geometry = ConditionalOrderGateLoss().per_segment_loss(context).segment_losses.mean()
    return ce+.1*gate+.01*geometry, dict(ce=float(ce.detach()),type=float(gate.detach()),order=float(geometry.detach()))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def compose(checkpoint, evaluation=False):
    import torch
    from safetensors.torch import load_file
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
    from src.adapters.dora import load_live_dora_adapter, normalize_dora_state_key
    from peft import get_peft_model_state_dict
    from src.qwen.untied_embeddings import (SpecialTokenSelection,
        install_special_token_embedding_deltas, load_special_token_embedding_deltas)
    policy = load(POLICY)
    policy_checkpoint = Path(policy['checkpoint'])
    for payload, expected in policy['payload_sha256'].items():
        payload = Path(payload)
        if payload.is_relative_to(policy_checkpoint):
            payload = CHECKPOINT / payload.relative_to(policy_checkpoint)
        assert digest(payload)==expected,payload
    q = load_qwen_components_from_options(QwenLoadOptions(policy['base_model'],'bf16',
                      'sdpa' if evaluation else 'flash_attention_2',load_model=True))
    # Same fixed BF16 base conversion for zero/control/treatment natural evaluation.
    if evaluation:
        q.model.float()
    model,adapter = load_live_dora_adapter(q.model,adapter_path=checkpoint/'adapter',base_model_path=q.base_model_path)
    q = replace(q,model=model)
    metadata = load(checkpoint/'special_token_embeddings/special_token_embeddings.json')
    selection = SpecialTokenSelection(token_strings=metadata['token_strings'],token_ids=metadata['token_ids'])
    assert metadata['tie_word_embeddings'] is False
    delta = install_special_token_embedding_deltas(q.model,selection,tie_word_embeddings=False)
    loaded = load_special_token_embedding_deltas(delta,checkpoint/'special_token_embeddings',expected_base_model_path=q.base_model_path,
        expected_base_config_sha256=q.base_config_sha256,expected_tokenizer_sha256=q.tokenizer_sha256)
    source = load_file(str(checkpoint/'adapter/adapter_model.safetensors'))
    live = {normalize_dora_state_key(k,adapter_name='default'):v for k,v in get_peft_model_state_dict(q.model,adapter_name='default').items()}
    for key,t in source.items():
        v = live[normalize_dora_state_key(key,adapter_name='default')]
        assert v.dtype==t.dtype and torch.equal(v.cpu(),t),key
    saved_delta = load_file(str(checkpoint/'special_token_embeddings/special_token_embeddings.safetensors'))
    for key,v in delta.delta_tensors().items():
        assert torch.equal(v.cpu(),saved_delta[key]) and v.dtype==saved_delta[key].dtype,key
    for name,p in q.model.named_parameters():
        p.requires_grad_(('lora_' in name or name in delta.receipt.delta_parameter_names) and not evaluation)
    groups = {name:dict(shape=list(p.shape),dtype=str(p.dtype),sha256=tensor_hash(p))
              for name,p in q.model.named_parameters() if p.requires_grad or 'lora_' in name or name in delta.receipt.delta_parameter_names}
    assert not any('visual' in name for name in groups)
    assert len(delta.delta_tensors())==2
    q.model.to('cuda')
    return q,delta,dict(adapter=adapter,delta=loaded.to_artifact_dict(),parameters=groups,base_conversion='BF16->FP32' if evaluation else 'BF16',components=q.to_artifact_dict())


def torch_equal_cpu(a,b):
    import torch
    return a.dtype==b.dtype and torch.equal(a.cpu(),b.detach().cpu())


def save_checkpoint(q, delta, output):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from src.qwen.untied_embeddings import save_special_token_embedding_deltas
    output.mkdir(parents=True,exist_ok=False)
    q.model.save_pretrained(output/'adapter',safe_serialization=True,save_embedding_layers=False)
    saved=load_file(str(output/'adapter/adapter_model.safetensors'))
    live=get_peft_model_state_dict(q.model,adapter_name='default')
    assert saved.keys()==live.keys()
    assert all(torch_equal_cpu(saved[k],live[k]) for k in saved)
    save_special_token_embedding_deltas(delta,output/'special_token_embeddings',base_model_path=q.base_model_path,
        base_config_sha256=q.base_config_sha256,tokenizer_sha256=q.tokenizer_sha256)
    write(output/'identity.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})


def runtime_start(output, root=ROOT):
    import torch
    from src.artifacts.git_identity import capture_source_identity
    rank,world = int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    assert world==8
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(92701)
    qualified=load(root/'cpu-qualification-02.json')
    for path,expected in qualified['sha256'].items():
        assert digest(path)==expected,path
    sources = ['probes/iterative_positive.py','probes/hidden_human_recovery.py',*sorted(str(p) for p in Path('src').rglob('*.py'))]
    identity = capture_source_identity(sources)
    out = output/f'rank-{rank}'
    out.mkdir(parents=True,exist_ok=False)
    write(out/'entry.json',dict(pid=os.getpid(),rank=rank,world=world,source=identity,start=time.time()))
    return rank,world,out,sources,identity


def forward_example(q, wrapped, raw, selected, vocab):
    import torch
    from src.qwen.native import NativeRequest,prepare_native_inputs,exact_history_inputs
    encoded,sequence = encode(raw,selected,q)
    batch = prepare_native_inputs(q.processor,[NativeRequest(raw.example_id,encoded.chat_text,raw.image.path,
              expected_token_ids=encoded.input_ids,expected_image_grid=encoded.image_encoding.plan.image_grid_thw)],device='cuda')
    positions = tuple(a.causal_logits_position for a in sequence.atoms) or (len(sequence.input_ids)-1,)
    kwargs = exact_history_inputs(q.model,batch.inputs,[encoded.input_ids],pad_token_id=q.tokenizer.pad_token_id)
    kwargs['logits_to_keep'] = torch.tensor(positions,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits = wrapped(**kwargs).logits
    loss,terms = image_loss(logits,sequence,vocab,positions)
    evidence = dict(image_id=raw.example_id,selected_ids=None if selected is None else sorted(selected),
                    input_sha256=stable(canonical(list(encoded.input_ids))),grid=list(encoded.image_encoding.plan.image_grid_thw),
                    tokens=len(encoded.input_ids),visual_tokens=encoded.image_token_count,
                    supervised_atoms=[a.to_artifact_dict() for a in sequence.atoms],logits_sha256=tensor_hash(logits),terms=terms)
    return loss,evidence


def lr_factor(step):
    return step/2 if step<=2 else .5*(1+math.cos(math.pi*(step-2)/14))


def train(output,arm,slice_run=False,root=ROOT):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from src.losses.vocab import build_token_vocabulary_groups
    from src.artifacts.git_identity import verify_source_identity
    rank,world,out,sources,identity = runtime_start(output,root=root)
    dist.init_process_group('nccl')
    start=time.monotonic()
    plan=load(root/'learning-plan.json')
    q,delta,composition=compose(Path(load(POLICY)['checkpoint']))
    load_seconds=time.monotonic()-start
    write(out/'composition.json',composition)
    vocab=build_token_vocabulary_groups(q.token_identity,tokenizer=q.tokenizer)
    params=[p for p in q.model.parameters() if p.requires_grad]
    delta_ids={id(p) for p in delta.delta_tensors().values()}
    optimizer=torch.optim.AdamW([dict(params=[p for p in params if id(p) not in delta_ids],lr=2e-5),
                                dict(params=list(delta.delta_tensors().values()),lr=1e-5)],betas=(.9,.999),eps=1e-8,weight_decay=0)
    assert not optimizer.state
    q.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    q.model.enable_input_require_grads()
    q.model.train()
    model=DistributedDataParallel(q.model,device_ids=[int(os.environ['LOCAL_RANK'])],broadcast_buffers=False)
    if rank==0:
        save_checkpoint(q,delta,output/'checkpoint-0')
    dist.barrier()
    schedule=plan['schedule'][:1] if slice_run else plan['schedule']
    history=[]
    for step in schedule:
        actual=step if not slice_run else dict(step=1,human=[1584,2299]*4,replay=[1584,2299]*8)
        items=rank_examples(actual,rank,treatment=arm=='treatment')
        if slice_run:
            items=[('visible' if b=='replay' else b,i,w) for b,i,w in items]
        optimizer.zero_grad(set_to_none=True)
        for micro,(branch,i,weight) in enumerate(items):
            from contextlib import nullcontext
            micro_start=time.monotonic()
            raw,selected=raw_for(plan,branch,i)
            with model.no_sync() if micro+1<len(items) else nullcontext():
                loss,evidence=forward_example(q,model,raw,selected,vocab)
                assert torch.isfinite(loss)
                (loss*weight).backward()
            torch.cuda.synchronize()
            evidence.update(branch=branch,weight=weight,step=step['step'],forward_backward_seconds=time.monotonic()-micro_start)
            write(out/f"step-{step['step']}-micro-{micro}.json",evidence)
            del loss
        norms={name:float(p.grad.float().norm()) if p.grad is not None else None for name,p in q.model.named_parameters() if p.requires_grad}
        assert all(v is not None and math.isfinite(v) for v in norms.values())
        assert all(any(v>0 for name,v in norms.items() if tag in name) for tag in ('lora_','embed_tokens.shared_embed_delta','lm_head.shared_embed_delta'))
        total=float(torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True))
        for group,base_lr in zip(optimizer.param_groups,(2e-5,1e-5)):
            group['lr']=base_lr*lr_factor(step['step'])
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in params)
        history.append(dict(step=step['step'],gradient_norm=total,gradient_norms=norms,lrs=[g['lr'] for g in optimizer.param_groups]))
        if slice_run or step['step'] in (4,8,16):
            if rank==0:
                save_checkpoint(q,delta,output/f"checkpoint-{step['step']}")
            dist.barrier()
    # Fresh-process BF16 forward parity anchor, independent of natural FP32 evaluation.
    q.model.eval()
    raw,selected=raw_for(plan,'pseudo',1584 if rank%2==0 else 2299)
    with torch.no_grad():
        _,evidence=forward_example(q,q.model,raw,selected,vocab)
    write(out/'final-forward.json',evidence)
    verify_source_identity(identity,required_paths=sources)
    write(out/'complete.json',dict(status='complete',arm=arm,slice=slice_run,steps=history,wall_seconds=time.monotonic()-start,
              load_seconds=load_seconds,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),source=identity))
    dist.destroy_process_group()


def reload_slice(checkpoint,output,training_root,root=ROOT):
    import torch
    from src.losses.vocab import build_token_vocabulary_groups
    from src.artifacts.git_identity import verify_source_identity
    rank,world,out,sources,identity=runtime_start(output,root=root)
    q,delta,composition=compose(checkpoint)
    q.model.eval()
    plan=load(root/'learning-plan.json')
    raw,selected=raw_for(plan,'pseudo',1584 if rank%2==0 else 2299)
    vocab=build_token_vocabulary_groups(q.token_identity,tokenizer=q.tokenizer)
    with torch.no_grad():
        _,evidence=forward_example(q,q.model,raw,selected,vocab)
    previous=load(training_root/f'rank-{rank}/final-forward.json')
    assert evidence==previous,'fresh reload forward differs'
    verify_source_identity(identity,required_paths=sources)
    write(out/'complete.json',dict(status='complete',composition=composition,forward=evidence,source=identity))


def evaluate_checkpoint(checkpoint,output,root=ROOT):
    import torch
    from src.qwen.generation import generate_continuations,NativeGenerationPolicy
    from src.artifacts.git_identity import verify_source_identity
    rank,world,out,sources,identity=runtime_start(output,root=root)
    start=time.monotonic()
    q,delta,composition=compose(checkpoint,evaluation=True)
    q.model.eval()
    policy=load(POLICY)
    plan=[r for r in request_plan(load(VISIBLE),policy) if r['arm']=='greedy']
    records=[]
    for item in plan[rank::world]:
        before=time.monotonic()
        batch=native_request(item,policy,q.processor)
        prepared=time.monotonic()-before
        before=time.monotonic()
        with torch.inference_mode():
            result=generate_continuations(q.model,batch,extensions=[()],budgets=[3084],eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),pad_token_id=q.tokenizer.pad_token_id,
                 policy=NativeGenerationPolicy(temperature=0,top_p=1,top_k=0,repetition_penalty=1,use_model_defaults=False),seed=None)[0]
        record=dict(item,token_ids=list(result.token_ids),text=q.tokenizer.decode(result.token_ids,skip_special_tokens=False),stop_reason=result.stop_reason,
                    generated_tokens=len(result.token_ids),generation_seconds=time.monotonic()-before,prepare_seconds=prepared,
                    prompt_token_ids=list(batch.prompt_token_ids[0]),image_grid_thw=batch.image_grids[0],media_sha256=batch.media_sha256[0])
        write(out/(str(item['image_id'])+'.json'),record)
        records.append(str(item['image_id'])+'.json')
    verify_source_identity(identity,required_paths=sources)
    write(out/'complete.json',dict(status='complete',composition=composition,source=identity,wall_seconds=time.monotonic()-start,
                                 artifacts={r:digest(out/r) for r in records}))


def read_evaluation(root):
    records=[]
    for rank in range(8):
        directory=root/f'rank-{rank}'
        receipt=load(directory/'complete.json')
        assert receipt['status']=='complete'
        for name,expected in receipt['artifacts'].items():
            assert digest(directory/name)==expected
            records.append(load(directory/name))
    assert len(records)==18 and len({r['image_id'] for r in records})==18
    assert {r['image_id'] for r in records}=={r['image_id'] for r in load(VISIBLE)}
    records.sort(key=lambda r:r['image_id'])
    frozen={str(path.relative_to(root)):digest(path) for path in sorted(root.rglob('*.json')) if path.name!='frozen.json'}
    if (root/'frozen.json').exists():
        assert load(root/'frozen.json')==frozen
    else:
        write(root/'frozen.json',frozen)
    return records


def offline_results(run_root, reference_root=None):
    """Only this post-freeze evaluator opens hidden truth; never called by training."""
    from collections import Counter
    from src.eval.saved_rows import one_to_one_matches
    roots={'zero':run_root/'evaluation-zero',**{f'{arm}-{step}':run_root/f'evaluation-{arm}-{step}' for arm in ('control','treatment') for step in (4,8,16)}}
    if reference_root is not None:
        roots.update({'zero':reference_root/'evaluation-zero',**{f'control-{step}':reference_root/f'evaluation-control-{step}' for step in (4,8,16)}})
    frozen={key:read_evaluation(root) for key,root in roots.items()}
    baseline={r['image_id']:r for r in frozen['zero']}
    for records in frozen.values():
        for r in records:
            for field in ('prompt_token_ids','image_grid_thw','media_sha256','crop','seed','temperature'):
                assert r[field]==baseline[r['image_id']][field],field
    truth=load(OLD/'preparation-v3/evaluator/truth.json')
    results={}
    for key,records in frozen.items():
        predictions,invalid=candidates(records)
        ids={}
        for reference in truth:
            pool=[p for p in predictions if p['image_id']==reference['image_id']]
            refs=[dict(owner_id=str(o['coco_ann_id']),reference_coord_bins_1000=o['bbox_2d']) for o in reference['objects']]
            matches=one_to_one_matches(refs,pool,.5)
            by={p['prediction_id']:p for p in pool}
            ids[str(reference['image_id'])]={mode:{kind:[] for kind in ('hidden','visible')} for mode in ('raw','category')}
            for match in matches:
                obj=reference['objects'][match['reference_index']]
                kind='hidden' if obj['coco_ann_id']<0 else 'visible'
                ids[str(reference['image_id'])]['raw'][kind].append(obj['coco_ann_id'])
                if obj['desc']==by[match['prediction_id']]['description']:
                    ids[str(reference['image_id'])]['category'][kind].append(obj['coco_ann_id'])
        results[key]=dict(coverage=original_coverage(truth,predictions),matched_ids=ids,
            stops=dict(Counter(r['stop_reason'] for r in records)),generated_tokens=sum(r['generated_tokens'] for r in records),
            generation_seconds=sum(r['generation_seconds'] for r in records),invalid_rows=invalid,
            invalid_reasons=dict(Counter(r.get('reason','unknown') for r in invalid)),
            literal_duplicates=len(predictions)-len({(p['image_id'],p['description'],tuple(p['coord_bins_1000'])) for p in predictions}))
    for key,result in results.items():
        changes=[]
        for reference in truth:
            i=str(reference['image_id'])
            for mode in ('raw','category'):
                old=results['zero']['matched_ids'][i][mode];new=result['matched_ids'][i][mode]
                row=dict(image_id=int(i),cohort=reference['cohort'],mode=mode)
                for kind in ('hidden','visible'):
                    row[kind+'_gains']=sorted(set(new[kind])-set(old[kind]))
                    row[kind+'_losses']=sorted(set(old[kind])-set(new[kind]))
                row['utility']=len(row['hidden_gains'])-len(row['hidden_losses'])-len(row['visible_losses'])
                changes.append(row)
        result['changes']=changes
    write(run_root/'offline-results.json',dict(results=results,truth_sha256=digest(OLD/'preparation-v3/evaluator/truth.json'),
        limitation='Annotation-ID proxy; refined5 redraw uncertainty; noisy positives are not gold; no checkpoint selection.'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','prepare-novel','train','reload','evaluate','offline'])
    p.add_argument('--output',type=Path)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--reference-root',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--training-root',type=Path)
    p.add_argument('--arm',choices=['control','treatment'])
    p.add_argument('--slice',action='store_true')
    a=p.parse_args()
    if a.command=='prepare': prepare()
    elif a.command=='prepare-novel': prepare_novel(a.root)
    elif a.command=='train': train(a.output,a.arm,a.slice,root=a.root)
    elif a.command=='offline': offline_results(a.output,a.reference_root)
    elif a.command=='reload': reload_slice(a.checkpoint,a.output,a.training_root,root=a.root)
    else: evaluate_checkpoint(a.checkpoint,a.output,root=a.root)


if __name__=='__main__': main()
