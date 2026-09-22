"""Two fixed layer20 component patches after the admitted S/F residual contrast."""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList
from probes.training_set_completion import repetition_history_runtime as native
from probes.training_set_completion import successful_row_state as state


def run(panel_path, output, component=None, donor_path=None):
    panel = json.loads(panel_path.read_text())
    native.fresh._check_sources(panel)
    case = panel['cases'][0]
    target = case['target_position']
    assert target == 1 and case['capture_offsets'] == [63, 67, 68, 76, 103]
    assert component in (None, 'attention', 'mlp')
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', mode='component_patch' if component else 'prefix_extraction',
                   component=component, panel=native._binding(panel_path),
                   producer=native._binding(Path(__file__)), pid=os.getpid())
    native._write(output/'receipt.json', receipt)
    start = time.monotonic()
    count = 0
    tensors = {'states': {}, 'cache': {}, 'logits': {}, 'head': {}, 'positions': {}}
    patch = []
    handles = []
    try:
        infer = native.fresh.InferConfig.model_validate(panel['config'])
        assert infer.model.dtype == 'fp32' and infer.backend.hf.attn_implementation == 'sdpa'
        qwen, identity = native.fresh.load_policy(infer, device=torch.device('cuda:0'))
        model = qwen.model.eval()
        language = model.model.language_model
        assert len(language.layers) == 28
        versions = {n:p._version for n,p in model.named_parameters()}
        donor = torch.load(donor_path, map_location='cpu', weights_only=True) if donor_path else None
        if component:
            assert donor is not None
            receipt['donor'] = native._binding(donor_path)

        def counter(module, args, kwargs):
            nonlocal count
            count += 1
            assert count <= (3084 if component else 68)

        def component_hook(kind):
            def hook(module, args, result):
                if count-1 != 67:
                    return result
                value = result[0] if kind == 'attention' else result
                assert value.shape[:2] == (4, 1)
                tensors['states'][kind] = value[:, -1].detach().cpu().clone()
                if kind != component:
                    return result
                replacement = donor['states'][kind][target]
                changed = state._replace_last_target(value, target, replacement)
                tensors['states'][kind+'_after'] = changed[:, -1].detach().cpu().clone()
                patch.append(dict(kind=kind, layer=20, action_offset=67,
                                  before=native._tensor_hash(value[target,-1]),
                                  after=native._tensor_hash(changed[target,-1]),
                                  donor=native._tensor_hash(replacement)))
                return (changed, *result[1:]) if kind == 'attention' else changed
            return hook

        def head_hook(module, args):
            off = count-1
            if off in [63,67,68,76,103]:
                tensors['head'][off] = args[0][:,-1].detach().cpu().clone()

        def positions_hook(module, args, kwargs):
            off = count-1
            if off in [63,67,68,76,103]:
                tensors['positions'][off] = kwargs['position_ids'].detach().cpu().clone()

        def output_hook(module, args, result):
            off = count-1
            if off in [63,67,68,76,103]:
                tensors['logits'][off] = result.logits[:,-1].detach().cpu().clone()
            if off in [67,68]:
                meta, values = state._cache_snapshot(result.past_key_values, offset=off, target_position=target)
                tensors['cache'][str(off)] = values

        handles = [model.register_forward_pre_hook(counter, with_kwargs=True),
                   language.layers[20].self_attn.register_forward_hook(component_hook('attention')),
                   language.layers[20].mlp.register_forward_hook(component_hook('mlp')),
                   model.get_output_embeddings().register_forward_pre_hook(head_hook),
                   language.register_forward_pre_hook(positions_hook, with_kwargs=True),
                   model.register_forward_hook(output_hook)]
        if component:
            coordinate_ids, factors, readout = native.fresh._load_coefficients(
                panel=panel, model=model, tokenizer=qwen.tokenizer, device=torch.device('cuda:0'))
            readout['loaded_model_identity'] = identity
            result = native._run_norm_cell(panel_path=panel_path, panel=panel, cell=case, mode='prefix',
                output=output/'native', qwen=qwen, infer=infer, loaded_identity=identity,
                coordinate_ids=coordinate_ids, factors=factors, readout=readout)
            assert len(patch)==1 and count==result['model_forwards']
            receipt.update(raw=result['raw'], native_receipt=native._binding(output/'native/309264/prefix/receipt.json'),
                           input_identity=result['input_identity'])
        else:
            config = copy.deepcopy(panel['config'])
            config['data']['input_jsonl'] = case['group']['input_jsonl']
            requests, _ = native.fresh.build_bound_native_requests(qwen, config, case['group']['cases'])
            batch = native.fresh.prepare_native_inputs(qwen.processor, requests, device='cuda:0', record_media_identity=True)
            receipt['input_identity'] = native.fresh._input_identity(batch)
            previous = json.loads(Path(case['saved_receipt']['path']).read_text())
            assert receipt['input_identity'] == previous['input_identity']
            width = batch.inputs['input_ids'].shape[1]
            prefix = case['target_prefix_token_ids']

            class Force(LogitsProcessor):
                def __call__(self, ids, scores):
                    off = ids.shape[1]-width
                    assert off==count-1
                    if off<54:
                        assert int(scores[target].argmax())==prefix[off]
                    return native._force(scores, target, prefix[off]) if off<len(prefix) else scores

            original = model.generate
            def generate(**kwargs):
                assert kwargs['max_new_tokens']==68 and not kwargs.get('do_sample')
                return original(**kwargs, logits_processor=LogitsProcessorList([Force()]))
            model.generate = generate
            try:
                with torch.no_grad():
                    values = native.fresh.generate_continuations(model, batch, extensions=[[]]*4, budgets=[68]*4,
                        eos_token_id=151645, pad_token_id=qwen.tokenizer.pad_token_id,
                        policy=native.fresh.NativeGenerationPolicy(temperature=0, top_p=1, top_k=0,
                            repetition_penalty=1, use_model_defaults=False), trace='none', seed=None)
            finally:
                model.generate = original
            saved = json.loads(Path(case['source_sampled_raw']['path']).read_text())['rows']
            assert count==68
            for value, row in zip(values,saved):
                assert list(value.token_ids)==row['token_ids'][:68]
            baseline = torch.load(output.parent/'runtime/native-S/logits.pt', map_location='cpu', weights_only=False)['raw_logits'][67]
            assert torch.equal(tensors['logits'][67], baseline)
            native._write(output/'prefix-tokens.json',dict(scope='fixed-prefix extraction only, no free-outcome estimate',
                rows=[dict(image_id=row['image_id'],token_ids=list(value.token_ids),stop=value.stop_reason)
                      for value,row in zip(values,saved)]))
            receipt.update(prefix_tokens=native._binding(output/'prefix-tokens.json'),
                           all_native_prefixes_exact=True, fork_logits_exact=True)
        assert versions == {n:p._version for n,p in model.named_parameters()}
        assert set(tensors['states']) >= {'attention','mlp'}
        torch.save(tensors, output/'components.pt')
        receipt.update(status='candidate_complete', model_forwards=count, patch=patch,
                       tensors=native._binding(output/'components.pt'), loaded_identity=identity,
                       parameter_versions_unchanged=True, elapsed_seconds=time.monotonic()-start,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       surface='layer20 attention projection or MLP output before residual addition, target last token only')
    except BaseException as error:
        receipt.update(status='technical_invalid',error=repr(error),model_forwards=count,elapsed_seconds=time.monotonic()-start)
        raise
    finally:
        native._write(output/'receipt.json',receipt)
        for handle in handles:handle.remove()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--component',choices=['attention','mlp'])
    parser.add_argument('--donor',type=Path)
    args=parser.parse_args()
    run(args.panel,args.output,args.component,args.donor)
