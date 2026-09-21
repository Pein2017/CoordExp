"""One fixed-row prefill diagnostic; never runs or admits a qualification."""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

REPO = Path('/data/CoordExp/.worktrees/coordexp-infras')
sys.path.insert(0, str(REPO))

import torch
from src.inference import execution_model_composition as composition_owner
from src.inference.vllm_qualification_runtime import _run_production_composition

TARGET = 'model.language_model.layers.0.self_attn.q_proj'


class DiagnosticComplete(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(left, right):
    a, b = left.detach().float(), right.detach().to(left.device).float()
    return {'shape': list(a.shape), 'left_dtype': str(left.dtype),
            'right_dtype': str(right.dtype), 'exact_equal': bool(torch.equal(left, right.to(left.device))),
            'max_abs_diff': float((a-b).abs().max().item()),
            'different_elements': int((a != b).sum().item())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    reference = json.loads(args.reference.read_text())
    config = Path(reference['resolved_config_identity']['entry_config_path'])
    result = {'status': 'error', 'scope': 'One original fixed-row prefill and one actual adapted linear; no global gate relaxation or admission.',
              'reference': str(args.reference.resolve()), 'reference_sha256': sha(args.reference),
              'reference_behavior_checks': reference['comparison']['behavior_checks'],
              'reference_full_vocab': reference['comparison']['full_vocab'],
              'reference_selected_vocab': reference['comparison']['selected_vocab'],
              'fixture_identity': reference['fixture_identity'], 'config': str(config),
              'script_sha256': sha(__file__), 'target': TARGET,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    with args.receipt.open('x') as output:
        original = composition_owner.compare_execution_models
        try:
            assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
            assert torch.cuda.device_count() == 1
            torch.set_num_threads(1)
            for source in reference['resolved_config_identity']['sources']:
                if sha(source['path']) != source['sha256']:
                    raise RuntimeError('frozen_config_source_changed')

            def observe(**kwargs):
                dynamic = kwargs['dynamic_model']
                dense = kwargs['materialized_model']
                inputs = kwargs['dynamic_native_inputs']
                from src.config.fingerprint import sha256_json
                ids = inputs['input_ids'][0].detach().cpu().tolist()
                if sha256_json(ids) != reference['fixture_identity']['dynamic_executed_prompt_ids_sha256']:
                    raise RuntimeError('frozen_prompt_changed')
                if not torch.equal(inputs['input_ids'], kwargs['materialized_native_inputs']['input_ids']):
                    raise RuntimeError('processor_prompt_mismatch')
                layer, dense_layer = dynamic.get_submodule(TARGET), dense.get_submodule(TARGET)
                captured = {}
                def capture(module, positional, value):
                    captured['input'] = positional[0].detach().clone()
                    captured['output'] = value.detach().clone()
                handle = layer.register_forward_hook(capture)
                try:
                    call = dict(inputs, return_dict=True, use_cache=False)
                    if 'logits_to_keep' in inspect.signature(dynamic.forward).parameters:
                        call['logits_to_keep'] = 1
                    with torch.inference_mode():
                        model_result = dynamic(**call)
                        del model_result
                finally:
                    handle.remove()
                x, live_output = captured['input'], captured['output']
                result['live_dtypes'] = {
                    'input': str(x.dtype), 'output': str(live_output.dtype),
                    'base': str(layer.base_layer.weight.dtype),
                    'A': str(layer.lora_A['default'].weight.dtype),
                    'B': str(layer.lora_B['default'].weight.dtype),
                    'magnitude': str(layer.lora_magnitude_vector['default'].weight.dtype),
                    'dense_weight': str(dense_layer.weight.dtype),
                }
                with torch.inference_mode():
                    result['actual_dynamic_vs_published_dense'] = metrics(live_output, dense_layer(x))
                    cpu_merge = copy.deepcopy(layer).cpu()
                    cpu_merge.merge(safe_merge=True)
                    result['cpu_merge_vs_published_weight'] = metrics(cpu_merge.base_layer.weight, dense_layer.weight.cpu())
                    del cpu_merge
                    gpu_merge = copy.deepcopy(layer)
                    gpu_merge.merge(safe_merge=True)
                    result['gpu_merge_vs_published_weight'] = metrics(gpu_merge.base_layer.weight, dense_layer.weight)
                    result['actual_dynamic_vs_gpu_merged'] = metrics(live_output, gpu_merge(x))
                    del gpu_merge
                    # Diagnostic reference only: identical BF16-representable weights
                    # and captured input, with arithmetic promoted explicitly to FP32.
                    fp32 = copy.deepcopy(layer).float()
                    fp32_x = x.float()
                    fp32_dynamic = fp32(fp32_x)
                    fp32.merge(safe_merge=True)
                    result['fp32_dynamic_vs_fp32_merged'] = metrics(fp32_dynamic, fp32(fp32_x))
                    del fp32, fp32_x, fp32_dynamic
                result['status'] = 'diagnostic_completed'
                raise DiagnosticComplete()

            composition_owner.compare_execution_models = observe
            try:
                _run_production_composition(config, args.receipt.parent)
            except DiagnosticComplete:
                pass
            else:
                raise RuntimeError('diagnostic_hook_not_executed')
        except Exception as exc:
            result['status'] = 'error'
            result['error'] = {'type': type(exc).__name__, 'code': getattr(exc, 'code', None), 'message': str(exc)[:400]}
        finally:
            composition_owner.compare_execution_models = original
            result['elapsed_seconds'] = time.time() - started
            result['exit_code'] = 0 if result['status'] == 'diagnostic_completed' else 1
            output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
