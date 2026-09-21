"""Fixed index-12 same-prefix diagnostic; no qualification or admission writes."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

REPO = Path('/data/CoordExp/.worktrees/coordexp-infras')
sys.path.insert(0, str(REPO))
import torch
from safetensors.torch import save_file
from transformers import LogitsProcessor, LogitsProcessorList
from src.config.fingerprint import sha256_json
from src.inference import execution_model as execution_owner
from src.inference import execution_model_composition as composition_owner
from src.inference.vllm_qualification_runtime import _run_production_composition

ROOT = Path(__file__).resolve().parent
REFERENCE = Path('/data/CoordExp/outputs/infra_base/optimization-20260912/vllm-qualification-final-01/evidence/composition/composition-receipt.json')
BINDING = REFERENCE.with_name('bound-execution-model.json')
INDEX = 12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DiagnosticComplete(Exception):
    pass


class FixedPrefix(LogitsProcessor):
    def __init__(self, prefix, prompt_width):
        self.prefix = prefix
        self.prompt_width = prompt_width
        self.target_scores = None

    def __call__(self, input_ids, scores):
        offset = input_ids.shape[-1] - self.prompt_width
        if offset < len(self.prefix):
            forced = torch.full_like(scores, -torch.inf)
            forced[:, self.prefix[offset]] = 0
            return forced
        if offset != INDEX:
            raise RuntimeError('unexpected_generation_offset')
        self.target_scores = scores.detach().cpu().clone()[0]
        return scores


def main():
    started = time.time()
    result = {'status': 'error', 'generated_index': INDEX,
              'scope': 'One fixed row, cached greedy decoding, 2 models x 2 fixed prefixes. Only diagnostic prefix intervention; no free-generation bound or qualification claim.',
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'reference': str(REFERENCE), 'reference_sha256': sha(REFERENCE),
              'binding': str(BINDING), 'binding_sha256': sha(BINDING),
              'script_sha256': sha(__file__)}
    old_compare = composition_owner.compare_execution_models
    old_resolve = execution_owner.resolve_execution_model
    with (ROOT / 'receipt.json').open('x') as receipt:
        try:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '7' or torch.cuda.device_count() != 1:
                raise RuntimeError('requires_only_gpu7')
            reference = json.loads(REFERENCE.read_text())
            bound = json.loads(BINDING.read_text())
            config = Path(reference['resolved_config_identity']['entry_config_path'])
            for source in reference['resolved_config_identity']['sources']:
                if sha(source['path']) != source['sha256']:
                    raise RuntimeError('frozen_config_source_changed')
            result['fixture_identity'] = reference['fixture_identity']
            result['execution_identity'] = {key: bound[key] for key in ('composition_key', 'snapshot_fingerprint', 'source_identity', 'target_dtype')}
            result['source_sha256'] = {str(path.relative_to(REPO)): sha(path) for path in [REPO/'src/inference/execution_model_composition.py', REPO/'src/inference/vllm_qualification_runtime.py', REPO/'src/inference/vllm_qualification_producer.py']}
            identity = composition_owner._snapshot_token_identity(bound['snapshot_manifest'], expected_fingerprint=bound['snapshot_fingerprint'])
            coordinate_values = {token: value for value, token in enumerate(identity.coordinate_token_ids)}
            def token_info(token):
                return {'token_id': token, 'coordinate_value': coordinate_values.get(token)}
            def top(vector):
                values, indices = vector.float().topk(8)
                return [dict(token_info(int(token)), value=float(value)) for token, value in zip(indices, values)]
            def checked_resolve(**kwargs):
                loaded = old_resolve(**kwargs)
                for field in ('composition_key', 'snapshot_fingerprint', 'source_identity', 'target_dtype', 'materialization'):
                    if loaded.get(field) != bound.get(field):
                        raise RuntimeError('execution_identity_changed:' + field)
                return loaded
            def observe(**kwargs):
                inputs = kwargs['dynamic_native_inputs']
                other = kwargs['materialized_native_inputs']
                if set(inputs) != set(other):
                    raise RuntimeError('processor_input_keys_changed')
                for key in inputs:
                    if not isinstance(inputs[key], torch.Tensor) or not torch.equal(inputs[key], other[key]):
                        raise RuntimeError('processor_native_inputs_differ:' + key)
                prompt_ids = inputs['input_ids'][0].detach().cpu().tolist()
                if sha256_json(prompt_ids) != reference['fixture_identity']['dynamic_executed_prompt_ids_sha256']:
                    raise RuntimeError('frozen_prompt_changed')
                result['native_inputs'] = {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in inputs.items()}
                result['generation_kwargs'] = kwargs['generation_kwargs']
                result['model_dtypes'] = {name: sorted({str(p.dtype) for p in kwargs[name + '_model'].parameters()}) for name in ('dynamic', 'materialized')}
                result['runs'] = {}
                vectors = {}
                for history in ('dynamic', 'materialized'):
                    prefix = reference['comparison'][history + '_generated_ids'][:INDEX]
                    for name in ('dynamic', 'materialized'):
                        key = history + '_prefix__' + name + '_model'
                        forced = FixedPrefix(prefix, len(prompt_ids))
                        generation = dict(kwargs['generation_kwargs'])
                        generation['max_new_tokens'] = INDEX + 1
                        with torch.inference_mode():
                            generated = kwargs[name + '_model'].generate(
                                **dict(inputs), **generation,
                                logits_processor=LogitsProcessorList([forced]),
                                return_dict_in_generate=True, output_logits=True,
                            )
                        observed = generated.sequences[0, len(prompt_ids):].detach().cpu().tolist()
                        if observed[:INDEX] != prefix or len(observed) != INDEX + 1:
                            raise RuntimeError('fixed_prefix_not_reproduced')
                        raw = generated.logits[-1][0].detach().cpu().clone()
                        scores = forced.target_scores
                        if scores is None or not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(scores).all()):
                            raise RuntimeError('missing_or_nonfinite_logits')
                        vectors[key + '__raw'] = raw
                        vectors[key + '__processed'] = scores
                        result['runs'][key] = {
                            'prefix_ids': prefix, 'next_token': token_info(observed[-1]),
                            'raw_dtype': str(raw.dtype), 'processed_dtype': str(scores.dtype),
                            'raw_top8': top(raw), 'processed_top8': top(scores),
                            'saved_free_generation_token': token_info(reference['comparison'][name + '_generated_ids'][INDEX]),
                        }
                        del generated
                result['same_prefix_comparisons'] = {}
                for history in ('dynamic', 'materialized'):
                    left = vectors[history + '_prefix__dynamic_model__raw'].float()
                    right = vectors[history + '_prefix__materialized_model__raw'].float()
                    a = result['runs'][history + '_prefix__dynamic_model']['next_token']
                    b = result['runs'][history + '_prefix__materialized_model']['next_token']
                    result['same_prefix_comparisons'][history] = {
                        'raw_logits_exact_equal': bool(torch.equal(left, right)),
                        'raw_logits_max_abs_diff': float((left-right).abs().max()),
                        'processed_argmax_ids_equal': a['token_id'] == b['token_id'],
                        'coordinate_value_delta': None if a['coordinate_value'] is None or b['coordinate_value'] is None else abs(a['coordinate_value']-b['coordinate_value']),
                    }
                save_file(vectors, ROOT/'target-logits.safetensors')
                result['raw_logits_artifact'] = {'path': str(ROOT/'target-logits.safetensors'), 'sha256': sha(ROOT/'target-logits.safetensors')}
                result['status'] = 'diagnostic_completed'
                raise DiagnosticComplete()
            execution_owner.resolve_execution_model = checked_resolve
            composition_owner.compare_execution_models = observe
            try:
                _run_production_composition(config, ROOT)
            except DiagnosticComplete:
                pass
            else:
                raise RuntimeError('diagnostic_hook_not_executed')
        except Exception as exc:
            result['error'] = {'type': type(exc).__name__, 'code': getattr(exc, 'code', None), 'message': str(exc)[:500]}
        finally:
            composition_owner.compare_execution_models = old_compare
            execution_owner.resolve_execution_model = old_resolve
            result['elapsed_seconds'] = time.time() - started
            result['exit_code'] = 0 if result['status'] == 'diagnostic_completed' else 1
            receipt.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
