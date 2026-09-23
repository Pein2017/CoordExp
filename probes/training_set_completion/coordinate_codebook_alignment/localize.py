"""First newly expanded branch mismatch on the frozen short qualification case."""
import argparse
import json
import time
from pathlib import Path

import torch
from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.coordinate_codebook_alignment import parity
from src.config.loader import load_train_config
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs, prepare_replay


def run(args):
    config = load_train_config(args.config).config
    result = {'status': 'running', 'image_id': 13004, 'model_forwards': 0, 'vision_forwards': 0,
              'bindings': [binding(args.config), binding(args.dataset), binding(__file__)], 'checked': []}
    handles = []
    started = time.time()
    try:
        source, _ = parity._load_source(config, torch.device('cuda:0'))
        expanded, receipt = parity._load_expanded(config, torch.device('cuda:0'))
        names = [name.removesuffix('.lora_B.default.weight')
                 for name in receipt['warm_start_surface']['initialized_zero_lora_B_tensors']]
        reference = {}
        def capture(name):
            def hook(module, inputs, output):
                reference[name] = (inputs[0].detach().clone(), output.detach().clone())
            return hook
        for name in names:
            handles.append(source.model.get_submodule(name).register_forward_hook(capture(name)))
        row = next(row for row in parity._read_rows(args.dataset) if int(row['image_id']) == 13004)
        inference = parity._infer_config(config, args.dataset)
        case = parity._plan_case(source, row, inference, args.dataset)
        requests, _ = build_bound_native_requests(source, inference.model_dump(mode='json'), [case])
        batch = prepare_native_inputs(source.processor, requests, device='cuda:0', record_media_identity=True)
        targets = parity._target_ids(source, row, inference, args.dataset)
        replay = prepare_replay(source.model, batch.inputs, prompt_token_ids=batch.prompt_token_ids[0],
                                continuation_token_ids=targets, compact_logits=True)
        with torch.inference_mode():
            result['model_forwards'] += 1; result['vision_forwards'] += 1
            source_logits = source.model(**replay.inputs).logits
        for handle in handles: handle.remove()
        handles.clear()
        class Found(Exception): pass
        def inspect(name):
            def hook(layer, inputs, output):
                x = inputs[0]
                ref_x, ref_y = reference.pop(name)
                base = layer.get_base_layer()(x)
                input_max = float((x.float()-ref_x.float()).abs().max())
                local_max = float((output.float()-base.float()).abs().max())
                result['checked'].append({'name':name, 'input_reference_max':input_max,
                                           'base_reference_max':float((base.float()-ref_y.float()).abs().max()),
                                           'local_post_cast_max':local_max})
                if local_max or input_max:
                    a, b = layer.lora_A['default'], layer.lora_B['default']
                    runtime_x = layer._cast_input_dtype(x, a.weight.dtype)
                    eye = torch.eye(a.weight.shape[1],device=x.device,dtype=runtime_x.dtype)
                    lora_weight = b(a(eye)).T
                    magnitude = layer.lora_magnitude_vector['default']
                    norm = magnitude.get_weight_norm(layer.get_base_layer().weight.to(runtime_x.dtype),
                                                     lora_weight, layer.scaling['default'])
                    scale = magnitude.weight / norm - 1
                    correction = magnitude(runtime_x, lora_A=a,lora_B=b,scaling=layer.scaling['default'],
                                           base_layer=layer.get_base_layer(),base_result=base)
                    index = int(scale.abs().argmax())
                    result['first_mismatch'] = {'module':name,'shape':list(x.shape),'input_dtype':str(x.dtype),
                        'runtime_input_dtype':str(runtime_x.dtype),'base_dtype':str(base.dtype),'output_dtype':str(output.dtype),
                        'device':str(x.device),'autocast_enabled':torch.is_autocast_enabled('cuda'),
                        'b_max':float(b.weight.abs().max()),'scale_minus_one_max':float(scale.abs().max()),
                        'worst_channel':index,'magnitude':float(magnitude.weight[index]),'runtime_norm':float(norm[index]),
                        'correction_before_cast_max':float(correction.abs().max()),
                        'reconstructed_post_cast_max':float(((base+correction).to(base.dtype).float()-base.float()).abs().max()),
                        'actual_reconstruction_max':float(((base+correction).to(base.dtype).float()-output.float()).abs().max()),
                        'bias_present':layer.get_base_layer().bias is not None,
                        'upstream_inputs_exact':input_max==0 and all(item['input_reference_max']==0 for item in result['checked'])}
                    raise Found()
            return hook
        for name in names:
            handles.append(expanded.model.get_submodule(name).register_forward_hook(inspect(name)))
        with torch.inference_mode():
            result['model_forwards'] += 1; result['vision_forwards'] += 1
            try:
                other = expanded.model(**replay.inputs).logits
                result['full_logits_max'] = float((source_logits.float()-other.float()).abs().max())
            except Found:
                pass
        result['status'] = 'localized' if 'first_mismatch' in result else 'no_new_branch_mismatch'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc)); raise
    finally:
        for handle in handles: handle.remove()
        result['wall_seconds'] = time.time()-started
        with args.output.open('x') as stream:
            json.dump(result,stream,indent=2); stream.write('\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ('config','dataset','output'): parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
