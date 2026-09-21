"""Finite candidate-row scoring on exact multimodal histories; no generation."""
from __future__ import annotations
import argparse
import copy
import json
import os
import time
from pathlib import Path
import torch
from probes.dora_owner_learning.runtime import load_policy
from probes.training_set_completion.artifacts import literal_binding as _binding, ascii_json_digest as _json_hash, write_pretty_json as _write
from src.qwen.input_identity import input_identity as _input_identity, tensor_hash as _tensor_hash
from src.config.inference import InferConfig
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import exact_history_inputs, prepare_native_inputs


def execute(panel_path: Path, image_id: int, out: Path):
    panel = json.loads(panel_path.read_text())
    case = next(c for c in panel['cases'] if c['image_id'] == image_id)
    for binding in panel['sources']:
        assert _binding(Path(binding['path'])) == binding, binding['path']
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    receipt = dict(status='running', panel=_binding(panel_path), producer=_binding(Path(__file__)),
                   pid=os.getpid(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   model_forwards=0, image_id=image_id)
    _write(out/'receipt.json', receipt)
    try:
        config = copy.deepcopy(panel['config'])
        config['data']['input_jsonl'] = case['group']['input_jsonl']
        qwen, identity = load_policy(InferConfig.model_validate(config), device=torch.device('cuda:0'))
        model = qwen.model.eval()
        requests, _ = build_bound_native_requests(qwen, config, case['group']['cases'])
        batch = prepare_native_inputs(qwen.processor, requests, device='cuda:0', record_media_identity=True)
        saved_receipt = json.loads(Path(case['saved_receipt']['path']).read_text())
        assert _input_identity(batch) == saved_receipt['input_identity']
        saved = json.loads(Path(case['saved_raw']['path']).read_text())['rows']
        target = case['target_position']
        head_inputs = []
        def capture(module, args):
            head_inputs.append(args[0][target].detach().cpu().clone())
        hook = model.get_output_embeddings().register_forward_pre_hook(capture)
        versions = {n: p._version for n, p in model.named_parameters()}
        results = {}
        with torch.inference_mode():
            for condition, history in case['score_histories'].items():
                candidates = case['score_candidates']
                rows = {}
                tensors = {}
                for name, candidate in candidates.items():
                    tokens = candidate['token_ids']
                    actions = history + tokens
                    histories = []
                    for j, prompt in enumerate(batch.prompt_token_ids):
                        suffix = actions if j == target else saved[j]['token_ids'][:len(actions)]
                        # Native finished companions receive pad at later decoding positions.
                        suffix = suffix + [qwen.tokenizer.pad_token_id] * (len(actions)-len(suffix))
                        histories.append(list(prompt)+suffix)
                    inputs = exact_history_inputs(model, batch.inputs, histories,
                                                  pad_token_id=qwen.tokenizer.pad_token_id,
                                                  logits_to_keep=len(tokens)+1)
                    head_inputs.clear()
                    value = model(**inputs).logits[target].float().detach()
                    receipt['model_forwards'] += 1
                    assert receipt['model_forwards'] <= panel['score_forward_bound_per_case']
                    assert value.shape[0] == len(tokens)+1
                    logits = value[:-1]
                    h = head_inputs[-1][:-1]
                    assert h.shape[0] == len(tokens)
                    lp = torch.log_softmax(logits, dim=-1)
                    selected = lp[torch.arange(len(tokens), device=lp.device), torch.tensor(tokens,device=lp.device)]
                    top = torch.topk(logits, 2, dim=-1)
                    rows[name] = dict(token_ids=tokens, sum_logprob=float(selected.sum()),
                        token_logprobs=selected.tolist(), winners=top.indices[:,0].tolist(),
                        top2_margins=(top.values[:,0]-top.values[:,1]).tolist(),
                        eos_logprob=float(lp[0,151645]), first_token_logprob=float(selected[0]),
                        positions_sha256=_tensor_hash(inputs['position_ids']),
                        prefix_sha256=_json_hash(history))
                    tensors[name] = dict(logits=logits.cpu(), head_input=h,
                                         positions=inputs['position_ids'][:,target].cpu())
                # Earliest competing token conditions on an identical literal history.
                forks = []
                names = list(candidates)
                for i, left in enumerate(names):
                    for right in names[i+1:]:
                        a,b = candidates[left]['token_ids'], candidates[right]['token_ids']
                        fork = next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), None)
                        if fork is None:
                            continue
                        v,w = tensors[left]['logits'][fork], tensors[right]['logits'][fork]
                        error = float((v-w).abs().max())
                        margin = float(v[a[fork]]-v[b[fork]])
                        forks.append(dict(left=left,right=right,row_offset=fork,
                            identical_prefix_tokens=a[:fork], left_minus_right_margin=margin,
                            future_suffix_numerical_error=error,
                            left_rank=1+int((v>v[a[fork]]).sum()),
                            right_rank=1+int((v>v[b[fork]]).sum()),
                            material_numerical_ambiguity=abs(margin)<=2*error))
                tensor_path=out/f'{condition}.pt'
                torch.save(tensors,tensor_path)
                results[condition]=dict(history_token_ids=history,prefix_sha256=_json_hash(history),
                                        candidates=rows,forks=forks,tensors=_binding(tensor_path))
                _write(out/'scores.json',results)
        hook.remove()
        assert versions == {n:p._version for n,p in model.named_parameters()}
        torch.cuda.synchronize()
        receipt.update(status='candidate_complete', loaded_identity=identity,
                       input_identity=_input_identity(batch), scores=_binding(out/'scores.json'),
                       elapsed_seconds=time.monotonic()-start,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       parameters_unchanged=True)
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-start)
        _write(out/'receipt.json',receipt)
        raise
    _write(out/'receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','image_id','model_forwards','elapsed_seconds')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--panel',type=Path,required=True)
    ap.add_argument('--image',type=int,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();execute(args.panel,args.image,args.out)
