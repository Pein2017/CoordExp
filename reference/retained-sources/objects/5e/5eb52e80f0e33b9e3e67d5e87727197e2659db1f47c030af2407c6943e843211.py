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
from src.qwen.native import prepare_native_inputs
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


def target_tokens(q, case, config):
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
    return tokens[:ends[min(1, len(ends)-1)]]


def prediction(q, batch, case, bridge, parser, extension=(), cap=64):
    result = generate(q.model,batch,q.tokenizer,bridge,parser,extension=extension,max_new_tokens=cap)
    result['parsed'] = parse_compact_object_box_closed(result['text'],row_id=case['row_id'],
        row_index=case['row_index'],image_width=case['image_width'],image_height=case['image_height']).to_artifact_dict()
    return result


def run(args):
    launch = json.loads(args.launch.read_text())
    admission_path = Path(launch['admission']['path'])
    assert binding(admission_path) == launch['admission'], 'admission drift'
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
        g = grammar(q.tokenizer)
        parser = partial(coordinate_role_from_prefix, **g)
        ids = g['coordinate_token_ids']
        torch.manual_seed(launch['training']['seeds'][0])
        sidecar = CoordinateAddressReadout(ids).to(args.device)
        tokens = target_tokens(q,case,config)
        roles = torch.tensor([parser(tokens[:i]) for i in range(len(tokens))],device=args.device)
        assert all((role >= 0) == (token in ids) for role,token in zip(roles.tolist(), tokens, strict=True))
        if args.mode == 'reload':
            saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            assert saved['launch'] == binding(args.launch)
            assert saved['source_identity'] == identity
            assert saved['coordinate_ids'] == ids
            sidecar.load_state_dict(saved['state'])
            result = prediction(q,batch,case,sidecar,parser,cap=launch['qualification']['greedy_cap'])
            expected = json.loads(args.expected.read_text())
            assert result['token_ids'] == expected['token_ids'], 'fresh reload greedy differs'
            # Free all four coordinate values after a description-only prefix.
            first = tokens.index(g['box_start_id'])+1
            box = prediction(q,batch,case,sidecar,parser,extension=tokens[:first],cap=8)
            assert len(box['token_ids']) >= 4 and all(t in ids for t in box['token_ids'][:4]), 'no freely generated complete coordinate box'
            write_once(args.output/'reload.json',dict(status='candidate',greedy=result,free_box=box,
                       checkpoint=binding(args.checkpoint), description_only_prefix=tokens[:first]))
        else:
            before = frozen_digest(q.model)
            logits, hidden, visual = replay(q.model,batch,tokens)
            logits, hidden = logits[:-1], hidden[:-1]
            hm, wm = (v//2 for v in batch.image_grids[0][1:])
            zero = sidecar(logits,hidden,visual,roles,hm,wm)
            zero_max = float((zero-logits).abs().max())
            assert torch.equal(zero,logits), 'zero gain complete-vocabulary parity'
            base = prediction(q,batch,case,None,parser,cap=launch['qualification']['greedy_cap'])
            zero_gen = prediction(q,batch,case,sidecar,parser,cap=launch['qualification']['greedy_cap'])
            assert base['token_ids'] == zero_gen['token_ids'], 'zero gain greedy differs'
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
            del zero, mutated_logits, mutated_h, mutated_v
            active = roles >= 0
            coord = logits[active][:,ids].detach()
            full_lse = torch.logsumexp(logits[active],-1).detach()
            h = hidden[active].detach()
            rr = roles[active]
            targets = torch.tensor([ids.index(tokens[i]) for i in coord_positions],device=args.device)
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
                if step == 1:
                    assert all(value > 0 for value in norms.values()), 'dead sidecar after gain update'
                gradients.append(norms)
                losses.append(float(loss))
                optimizer.step()
            final_coord = sidecar.adjust_coordinate_logits(coord,h,visual,rr,hm,wm)
            final_loss = float((full_lse-final_coord.gather(1,targets[:,None]).squeeze(1)).mean())
            assert final_loss < losses[0], 'tiny fit did not learn'
            deltas = {name:float((p.detach()-initial[name]).norm()) for name,p in sidecar.named_parameters()}
            assert all(v > 0 for v in deltas.values())
            assert frozen_digest(q.model) == before, 'frozen original changed'
            # Family mass and non-coordinate identity at a fixed real prefix.
            raw,hfull,vfull = replay(q.model,batch,tokens)
            all_roles = torch.cat((roles,torch.tensor([parser(tokens)],device=args.device)))
            changed = sidecar(raw,hfull,vfull,all_roles,hm,wm)
            mask = torch.ones(raw.shape[-1],device=args.device,dtype=torch.bool); mask[ids] = False
            assert torch.equal(changed[:,mask],raw[:,mask])
            mass_max = float((torch.logsumexp(changed[:,ids],-1)-torch.logsumexp(raw[:,ids],-1)).abs().max())
            assert mass_max <= 2e-5
            assert torch.equal(changed[all_roles<0],raw[all_roles<0])
            checkpoint = dict(state={k:v.detach().cpu() for k,v in sidecar.state_dict().items()},
                              launch=binding(args.launch),source_identity=identity,coordinate_ids=ids)
            torch.save(checkpoint,args.output/'sidecar.pt')
            trained = prediction(q,batch,case,sidecar,parser,cap=launch['qualification']['greedy_cap'])
            write_once(args.output/'trained-generation.json',trained)
            # Full-prefix replay versus independently cached generated histories.
            maxima = []
            for length in sorted({0,min(4,len(trained['token_ids'])-1),min(12,len(trained['token_ids'])-1)}):
                prefix = trained['token_ids'][:length]
                full,hf,vf = replay(q.model,batch,prefix)
                role = torch.tensor([parser(prefix)],device=args.device)
                next_logits = sidecar(full[-1:],hf[-1:],vf,role,hm,wm)
                assert int(next_logits.argmax()) == trained['token_ids'][length]
                selected = trained['token_ids'][length]
                lp = float(next_logits[0,selected]-torch.logsumexp(next_logits[0],0))
                maxima.append(abs(lp-trained['trace'][length]['logprob']))
            assert max(maxima) <= 2e-4, 'cached/full chosen-token logprob differs'
            write_once(args.output/'qualification.json',dict(status='candidate',parameter_count=sum(p.numel() for p in sidecar.parameters()),
                zero_gain_max=zero_max,causal_suffix_max=causal_max,family_logsumexp_max=mass_max,
                cached_replay_logprob_max=max(maxima),losses=losses,final_loss=final_loss,gradients=gradients,
                parameter_deltas=deltas,frozen_model_sha256=before,updates=len(losses),tokens_per_update=targets.numel(),
                tiny_fit_seconds=time.time()-started,baseline=base,checkpoint=binding(args.output/'sidecar.pt')))
        ledger.finish('candidate')
    except BaseException as exc:
        ledger.finish('failed',repr(exc))
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--launch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--mode',choices=('train','reload'),default='train')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--expected',type=Path)
    run(p.parse_args())

if __name__ == '__main__':
    main()
