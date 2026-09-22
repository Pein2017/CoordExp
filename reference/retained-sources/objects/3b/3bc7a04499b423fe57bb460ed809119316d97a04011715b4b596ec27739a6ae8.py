"""Bounded real-model train/save qualification and independent fresh reload."""
from __future__ import annotations
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import torch
from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.untied_shared import load_model, config_for
from probes.training_set_completion.coordinate_address_readout.bridge import CoordinateAddressReadout, coordinate_role_from_prefix
from probes.training_set_completion.coordinate_address_readout.runtime import RunLedger, write_once, replay, generate
from probes.training_set_completion.coordinate_address_readout.geometry_check import check as check_geometry
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs, exact_history_inputs
from src.qwen.input_identity import tensor_hash
from src.qwen.generation import generate_continuations, NativeGenerationPolicy
from probes.training_set_completion.coordinate_address_readout.runtime import NativeCapture
from src.data.examples import raw_example_from_jsonl_row
from src.templates.renderer import render_example
from src.config.models import TemplateConfig
from src.inference.parsing import parse_compact_object_box_closed


def grammar(tokenizer):
    return dict(coordinate_token_ids=[tokenizer.convert_tokens_to_ids(f'<|coord_{b}|>') for b in range(1000)],
                object_ref_start_id=tokenizer.convert_tokens_to_ids('<|object_ref_start|>'),
                object_ref_end_id=tokenizer.convert_tokens_to_ids('<|object_ref_end|>'),
                box_start_id=tokenizer.convert_tokens_to_ids('<|box_start|>'),
                box_end_id=tokenizer.convert_tokens_to_ids('<|box_end|>'),
                eos_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'))


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            raise ValueError(f'original parameter is trainable: {name}')
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_batch(q, case, config, device):
    requests, _ = build_bound_native_requests(q, config, [case])
    return prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)


def target_tokens(q, case, config, *, row_limit=2):
    raw = raw_example_from_jsonl_row(case['input_record'], jsonl_path=Path(config['data']['input_jsonl']),
                                   row_number=case['row_index']+1, raw_line=json.dumps(case['input_record']))
    template = TemplateConfig(**{k:config['template'][k] for k in
                               ('object_field_order','object_ordering','assistant_format','prompt')})
    rendered = render_example(raw, template)
    tokens = q.tokenizer.encode(rendered.supervised_response_text, add_special_tokens=False)
    # Qualification only: first two complete positive rows, never a training denominator change.
    end = q.tokenizer.convert_tokens_to_ids('<|box_end|>')
    ends = [i+1 for i,t in enumerate(tokens) if t == end]
    if not ends:
        raise ValueError('qualification requires a positive full box')
    return tokens if row_limit is None else tokens[:ends[min(row_limit-1, len(ends)-1)]]


def prediction(q, batch, case, bridge, parser, extension=(), cap=64, capture_logits=False):
    result = generate(q.model,batch,q.tokenizer,bridge,parser,extension=extension,max_new_tokens=cap,capture_logits=capture_logits)
    result['parsed'] = parse_compact_object_box_closed(q.tokenizer.decode(list(extension)+result['token_ids'],skip_special_tokens=False),row_id=case['row_id'],
        row_index=case['row_index'],image_width=case['image_width'],image_height=case['image_height']).to_artifact_dict()
    return result


def alignment_evidence(q, batch, tokens, logits, hidden, positions):
    history = list(batch.prompt_token_ids[0])+tokens
    selected_positions = torch.tensor([len(batch.prompt_token_ids[0])+i-1 for i in positions],device=logits.device)
    inputs = exact_history_inputs(q.model,batch.inputs,[history],pad_token_id=q.tokenizer.pad_token_id)
    inputs['logits_to_keep'] = selected_positions
    with NativeCapture(q.model,batch.image_grids[0]) as captured, torch.no_grad():
        selected = q.model(**inputs).logits[0]
    with torch.no_grad():
        same_shape_reference = q.model.get_output_embeddings()(hidden[positions][None])[0]
    return dict(target_positions=positions,selected_model_positions=selected_positions.tolist(),
                raw_logits_max=float((selected-logits[positions]).abs().max()),
                hidden_max=float((captured.hidden[0]-hidden[positions]).abs().max()),
                shape_matched_head_logits_max=float((selected-same_shape_reference).abs().max()),
                native_head_shape_effect_max=float((same_shape_reference-logits[positions]).abs().max()))


def run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    launch = json.loads(args.launch.read_text())
    admission_path = Path(launch['admission']['path'])
    assert binding(admission_path) == launch['admission'], 'admission drift'
    for entry in launch['input_bindings']:
        assert binding(Path(entry['path'])) == entry, 'input/source payload drift'
    config = launch['model_config']
    case = launch['qualification_cases'][0]
    ledger = RunLedger(args.output, args.device)
    try:
        write_once(args.output/'launch-binding.json', dict(launch=binding(args.launch), mode=args.mode))
        q, identity = load_model('tied',torch.device(args.device))
        q.processor.image_processor.do_resize = False
        write_once(args.output/'source-identity.json', dict(loader=identity, runtime=q.to_artifact_dict()))
        ledger.capture_sources()
        ledger.attach(q.model)
        write_once(args.output/'geometry.json',check_geometry(q.processor))
        batch = build_batch(q,case,config,torch.device(args.device))
        write_once(args.output/'actual-input.json',dict(case_id=case['row_id'],prompt_token_ids=list(batch.prompt_token_ids[0]),
            image_grid=list(batch.image_grids[0]),media_sha256=list(batch.media_sha256),
            tensors={k:tensor_hash(v) for k,v in batch.inputs.items() if isinstance(v,torch.Tensor)}))
        g = grammar(q.tokenizer)
        parser = partial(coordinate_role_from_prefix, **g)
        ids = g['coordinate_token_ids']
        torch.manual_seed(launch['training']['seeds'][0])
        sidecar = CoordinateAddressReadout(ids).to(args.device)
        tokens = target_tokens(q,case,config)
        roles = torch.tensor([parser(tokens[:i]) for i in range(len(tokens))],device=args.device)
        assert all((role >= 0) == (token in ids) for role,token in zip(roles.tolist(), tokens, strict=True))
        if args.mode == 'alignment':
            raw,h,_ = replay(q.model,batch,tokens)
            evidence=alignment_evidence(q,batch,tokens,raw[:-1],h[:-1],[i for i,t in enumerate(tokens) if t in ids])
            write_once(args.output/'alignment.json',evidence)
        elif args.mode == 'reload':
            saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            assert saved['launch'] == binding(args.launch)
            assert saved['source_identity'] == identity
            assert saved['coordinate_ids'] == ids
            sidecar.load_state_dict(saved['state'])
            result = prediction(q,batch,case,sidecar,parser,cap=launch['qualification']['greedy_cap'])
            expected = json.loads(args.expected.read_text())
            assert result['token_ids'] == expected['token_ids'], 'fresh reload greedy differs'
            # Free all four coordinate values after a description-only prefix.
            diagnostic = launch['diagnostic_case']
            diagnostic_batch = build_batch(q,diagnostic,launch['calibration_config'],torch.device(args.device))
            extension = q.tokenizer.encode('<|object_ref_start|>'+launch['diagnostic_referent']+'<|object_ref_end|><|box_start|>',add_special_tokens=False)
            assert not any(t in ids for t in extension)
            box = prediction(q,diagnostic_batch,diagnostic,sidecar,parser,extension=extension,cap=8)
            assert len(box['token_ids']) >= 4 and all(t in ids for t in box['token_ids'][:4]), 'no freely generated complete coordinate box'
            write_once(args.output/'reload.json',dict(status='candidate',greedy=result,free_box=box,
                       checkpoint=binding(args.checkpoint),description_only_prefix=extension,diagnostic_case_id=diagnostic['row_id']))
        else:
            before = frozen_digest(q.model)
            inference_start = time.time()
            logits, hidden, visual = replay(q.model,batch,tokens)
            logits, hidden = logits[:-1], hidden[:-1]
            hm, wm = (v//2 for v in batch.image_grids[0][1:])
            zero = sidecar(logits,hidden,visual,roles,hm,wm)
            zero_max = float((zero-logits).abs().max())
            assert torch.equal(zero,logits), 'zero gain complete-vocabulary parity'
            base = prediction(q,batch,case,None,parser,cap=launch['qualification']['greedy_cap'])
            source_result, = generate_continuations(q.model,batch,extensions=[[]],budgets=[launch['qualification']['greedy_cap']],
                eos_token_id=g['eos_token_id'],pad_token_id=q.tokenizer.pad_token_id,policy=NativeGenerationPolicy())
            assert list(source_result.token_ids) == base['token_ids'], 'caller differs from maintained native greedy source'
            zero_gen = prediction(q,batch,case,sidecar,parser,cap=launch['qualification']['greedy_cap'])
            assert base['token_ids'] == zero_gen['token_ids'], 'zero gain greedy differs'
            write_once(args.output/'source-parity.json',dict(full_vocab_zero_gain_max=zero_max,baseline=base,
                source_native_tokens=list(source_result.token_ids),zero_gain_tokens=zero_gen['token_ids']))
            inference_seconds = time.time()-inference_start
            # Future target mutation: score rows through the last mutated target's predecessor.
            mutated = list(tokens)
            coord_positions = [i for i,t in enumerate(tokens) if t in ids]
            j = coord_positions[-1]
            mutated[j] = ids[(ids.index(mutated[j])+137)%1000]
            mutated_logits, mutated_h, mutated_v = replay(q.model,batch,mutated)
            causal_max = float((mutated_logits[:j+1]-logits[:j+1]).abs().max())
            assert causal_max <= 1e-5
            assert torch.equal(mutated_v,visual)
            assert torch.allclose(mutated_h[:j+1],hidden[:j+1],atol=1e-6,rtol=0)
            # Deliberately wrong same-position alignment must disagree at mutated target.
            wrong_alignment_delta = float((mutated_h[j+1]-replay(q.model,batch,tokens)[1][j+1]).abs().max())
            assert wrong_alignment_delta > 1e-6, 'causal mutation lacks sensitivity'
            # Explicit compact selected-row mapping must equal j-1 teacher-forced rows.
            aligned = alignment_evidence(q,batch,tokens,logits,hidden,coord_positions)
            compact_max = aligned['shape_matched_head_logits_max']
            write_once(args.output/'causal-alignment.json',dict(**aligned,roles=roles.tolist(),
                suffix_mutation_position=j,causal_max=causal_max,wrong_same_position_delta=wrong_alignment_delta))
            assert aligned['hidden_max'] == 0.0 and compact_max <= 2e-5
            assert aligned['raw_logits_max'] == aligned['native_head_shape_effect_max']
            del zero, mutated_logits, mutated_h, mutated_v
            active = roles >= 0
            coord = logits[active][:,ids].detach()
            full_lse = torch.logsumexp(logits[active],-1).detach()
            h = hidden[active].detach()
            rr = roles[active]
            targets = torch.tensor([ids.index(tokens[i]) for i in coord_positions],device=args.device)
            native_ce = torch.nn.functional.cross_entropy(logits[active],torch.tensor([tokens[i] for i in coord_positions],device=args.device))
            cached_ce = (full_lse-coord.gather(1,targets[:,None]).squeeze(1)).mean()
            ce_accounting_max = float((native_ce-cached_ce).abs())
            assert ce_accounting_max <= 1e-5, 'cached full-vocabulary CE mismatch'
            del logits, hidden
            initial = {k:v.detach().clone() for k,v in sidecar.state_dict().items()}
            optimizer = torch.optim.AdamW(sidecar.parameters(),lr=launch['training']['learning_rate'],
                                         weight_decay=0,betas=(.9,.999),eps=1e-8)
            assert {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in sidecar.parameters()}
            losses, gradients = [], []
            started = time.time()
            for step in range(launch['qualification']['updates']):
                optimizer.zero_grad(set_to_none=True)
                adjusted = sidecar.adjust_coordinate_logits(coord,h,visual,rr,hm,wm)
                loss = (full_lse-adjusted.gather(1,targets[:,None]).squeeze(1)).sum()/targets.numel()
                loss.backward()
                norms = {name:float(p.grad.norm()) for name,p in sidecar.named_parameters()}
                assert all(torch.isfinite(p.grad).all() for p in sidecar.parameters())
                if step == 0:
                    assert norms['gain'] > 0
                    assert all(value == 0 for name,value in norms.items() if name != 'gain')
                if step == 1:
                    assert all(value > 0 for value in norms.values()), 'dead sidecar after gain update'
                gradients.append(norms)
                losses.append(float(loss))
                optimizer.step()
            fit_seconds = time.time()-started
            final_coord = sidecar.adjust_coordinate_logits(coord,h,visual,rr,hm,wm)
            final_loss = float((full_lse-final_coord.gather(1,targets[:,None]).squeeze(1)).mean())
            assert final_loss < losses[0], 'tiny fit did not learn'
            gate_only = CoordinateAddressReadout(ids).to(args.device)
            gate_only.load_state_dict(initial)
            with torch.no_grad():
                gate_only.gain.copy_(sidecar.gain)
                gate_logits = gate_only.adjust_coordinate_logits(coord,h,visual,rr,hm,wm)
                gate_only_loss = float((full_lse-gate_logits.gather(1,targets[:,None]).squeeze(1)).mean())
            assert final_loss < gate_only_loss, 'no measured Q/K/role learning contribution at learned gain'
            deltas = {name:float((p.detach()-initial[name]).norm()) for name,p in sidecar.named_parameters()}
            assert all(v > 0 for v in deltas.values())
            after = frozen_digest(q.model)
            assert after == before, 'frozen original changed'
            write_once(args.output/'fit.json',dict(losses=losses,final_loss=final_loss,initial_QK_at_final_gain_loss=gate_only_loss,
                gradients=gradients,parameter_deltas=deltas,frozen_before=before,frozen_after=after,ce_accounting_max=ce_accounting_max,
                updates=len(losses),tokens_per_update=targets.numel(),seconds=fit_seconds,
                initial_bin_mae=float((coord.argmax(-1)-targets).abs().float().mean()/1000),
                final_bin_mae=float((final_coord.argmax(-1)-targets).abs().float().mean()/1000)))
            # Family mass and non-coordinate identity at a fixed real prefix.
            raw,hfull,vfull = replay(q.model,batch,tokens)
            all_roles = torch.cat((roles,torch.tensor([parser(tokens)],device=args.device)))
            changed = sidecar(raw,hfull,vfull,all_roles,hm,wm)
            mask = torch.ones(raw.shape[-1],device=args.device,dtype=torch.bool); mask[ids] = False
            assert torch.equal(changed[:,mask],raw[:,mask])
            mass_max = float((torch.logsumexp(changed[:,ids],-1)-torch.logsumexp(raw[:,ids],-1)).abs().max())
            assert mass_max <= 2e-5
            assert torch.equal(changed[all_roles<0],raw[all_roles<0])
            write_once(args.output/'slot-isolation.json',dict(family_logsumexp_max=mass_max,noncoordinate_bitwise_equal=True,
                unadmitted_bitwise_equal=True,admitted_changed_max=float((changed[all_roles>=0]-raw[all_roles>=0]).abs().max())))
            checkpoint = dict(state={k:v.detach().cpu() for k,v in sidecar.state_dict().items()},
                              launch=binding(args.launch),source_identity=identity,coordinate_ids=ids)
            torch.save(checkpoint,args.output/'sidecar.pt')
            decode_started = time.time()
            trained = prediction(q,batch,case,sidecar,parser,cap=launch['qualification']['greedy_cap'],capture_logits=True)
            decode_seconds = time.time()-decode_started
            cached_logits = trained.pop('saved_logits')
            torch.save(cached_logits,args.output/'cached-selected-logits.pt')
            write_once(args.output/'trained-generation.json',trained)
            # Full-prefix replay versus independently cached generated histories.
            maxima, full_maxima = [], []
            for length,cached in cached_logits.items():
                prefix = trained['token_ids'][:length]
                full,hf,vf = replay(q.model,batch,prefix)
                role = torch.tensor([parser(prefix)],device=args.device)
                next_logits = sidecar(full[-1:],hf[-1:],vf,role,hm,wm)
                full_maxima.append(float((next_logits-cached.to(args.device)).abs().max()))
                assert int(next_logits.argmax()) == trained['token_ids'][length]
                selected = trained['token_ids'][length]
                lp = float(next_logits[0,selected]-torch.logsumexp(next_logits[0],0))
                maxima.append(abs(lp-trained['trace'][length]['logprob']))
            assert max(maxima) <= 2e-4 and max(full_maxima) <= 2e-4, 'cached/full logits differ'
            # A different admitted image, then A again: no per-request visual-bank leakage.
            second = launch['qualification_cases'][1]
            second_batch = build_batch(q,second,config,torch.device(args.device))
            _,_,other_visual = replay(q.model,second_batch,[])
            again,again_h,again_visual = replay(q.model,batch,tokens)
            assert torch.equal(again_visual,visual), 'visual bank leaked between requests'
            assert tensor_hash(other_visual) != tensor_hash(visual)
            assert torch.allclose(again,raw,atol=2e-5,rtol=0)
            # Largest admitted train target exercises full teacher history and supplies forecast bounds.
            long_tokens = target_tokens(q,second,config,row_limit=None)
            long_start = time.time()
            long_logits,long_h,long_v = replay(q.model,second_batch,long_tokens)
            feature_seconds = time.time()-long_start
            long_roles = torch.tensor([parser(long_tokens[:i]) for i in range(len(long_tokens))],device=args.device)
            long_active = long_roles >= 0
            long_targets = torch.tensor([ids.index(t) for t in long_tokens if t in ids],device=args.device)
            long_coord = long_logits[:-1][long_active][:,ids].detach()
            long_lse = torch.logsumexp(long_logits[:-1][long_active],-1).detach()
            long_h = long_h[:-1][long_active].detach()
            del long_logits
            optimizer.zero_grad(set_to_none=True)
            shape = second_batch.image_grids[0]
            throughput_start = time.time()
            long_out = sidecar.adjust_coordinate_logits(long_coord,long_h,long_v,long_roles[long_active],shape[1]//2,shape[2]//2)
            long_loss = (long_lse-long_out.gather(1,long_targets[:,None]).squeeze(1)).mean()
            long_loss.backward()  # throughput/gradient check only, no extra optimizer update
            torch.cuda.synchronize()
            sidecar_backward_seconds = time.time()-throughput_start
            assert all(torch.isfinite(p.grad).all() for p in sidecar.parameters())
            write_once(args.output/'throughput.json',dict(full_teacher_case=second['row_id'],full_teacher_tokens=len(long_tokens),
                coordinate_tokens=long_targets.numel(),visual_tokens=long_v.shape[0],feature_forward_seconds=feature_seconds,
                sidecar_forward_backward_seconds=sidecar_backward_seconds,greedy_tokens=len(trained['token_ids']),
                greedy_seconds=decode_seconds,tiny_fit_seconds=fit_seconds,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                training_cache_bytes_per_largest_case=sum(t.numel()*t.element_size() for t in [long_coord,long_lse,long_h,long_v,long_targets]),
                additional_optimizer_updates=0))
            write_once(args.output/'qualification.json',dict(status='candidate',parameter_count=sum(p.numel() for p in sidecar.parameters()),
                zero_gain_max=zero_max,causal_suffix_max=causal_max,family_logsumexp_max=mass_max,
                cached_replay_logprob_max=max(maxima),cached_replay_full_vocab_max=max(full_maxima),cached_selected_steps=sorted(cached_logits),
                losses=losses,final_loss=final_loss,initial_QK_at_final_gain_loss=gate_only_loss,gradients=gradients,
                parameter_deltas=deltas,frozen_model_sha256=before,updates=len(losses),tokens_per_update=targets.numel(),
                tiny_fit_seconds=fit_seconds,baseline=base,bank_isolation=True,checkpoint=binding(args.output/'sidecar.pt')))
        ledger.finish('candidate')
    except BaseException as exc:
        ledger.finish('failed',repr(exc))
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--launch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--mode',choices=('train','reload','alignment'),default='train')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--expected',type=Path)
    run(p.parse_args())

if __name__ == '__main__':
    main()
