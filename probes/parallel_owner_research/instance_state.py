"""Owner-grounding cache intervention; scoped native-generation experiment."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

import torch

from src.artifacts.source_provenance import preserve_source

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/instance-state')
SOURCE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-small-owner-repeat-origin/packet-repair-01.json')
EOS, OPENER, BOX_END = 151645, 151646, 151649


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def region_indices(prompt, grid, image_token, box, count=None):
    """Merged visual tokens retain raster order in native Qwen3VL scatter."""
    t, h, w = grid
    require(t == 1 and h % 2 == 0 and w % 2 == 0, 'single still-image merged grid required')
    positions = [i for i, token in enumerate(prompt) if token == image_token]
    h, w = h // 2, w // 2
    require(len(positions) == h * w, 'image-token/grid count mismatch')
    x1, y1, x2, y2 = box
    require(0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999, 'invalid region')
    eligible = []
    for ordinal, position in enumerate(positions):
        y, x = divmod(ordinal, w)
        x, y = (x + .5) * 999 / w, (y + .5) * 999 / h
        if x1 <= x < x2 and y1 <= y < y2:
            eligible.append(((x - (x1 + x2) / 2) ** 2 + (y - (y1 + y2) / 2) ** 2, position))
    require(bool(eligible), 'region contains no merged image centers')
    if count is None:
        count = len(eligible)
    require(len(eligible) >= count > 0, 'control region cannot match target key count')
    return sorted(position for _, position in sorted(eligible)[:count])


def block_grounding(mask, queries, keys, *, length, device, dtype):
    require(queries and keys and min(queries) > max(keys), 'grounding keys must precede row queries')
    require(max(queries) < length, 'query outside prefill')
    if mask is None:
        allowed = torch.ones((length, length), dtype=torch.bool, device=device).tril()
        result = torch.zeros((1, 1, length, length), device=device, dtype=dtype)
        result.masked_fill_(~allowed, torch.finfo(dtype).min)
    else:
        require(mask.ndim == 4 and mask.shape[-2:] == (length, length), 'unsupported prefill mask')
        result = mask.clone()
    q = torch.tensor(queries, device=device)
    k = torch.tensor(keys, device=device)
    result[..., q[:, None], k[None, :]] = False if result.dtype == torch.bool else torch.finfo(result.dtype).min
    return result


@contextmanager
def grounding_mask(model, queries, keys, length, receipt):
    modules = [(name, module) for name, module in model.named_modules()
               if type(module).__name__ == 'Qwen3VLTextAttention']
    require(bool(modules), 'native Qwen text attention modules absent')
    receipt.update(expected_layers=len(modules), calls=[], query_count=len(queries), key_count=len(keys))
    handles = []

    def hook(module, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        require(hidden.shape[1] == length, 'mask attempted outside exact full prefill')
        altered = dict(kwargs)
        altered['attention_mask'] = block_grounding(
            kwargs.get('attention_mask'), queries, keys, length=length,
            device=hidden.device, dtype=hidden.dtype)
        receipt['calls'].append(int(module.layer_idx))
        return args, altered

    try:
        for _, module in modules:
            handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        yield
        require(sorted(receipt['calls']) == list(range(len(modules))), 'mask layer consumption mismatch')
    finally:
        for handle in handles:
            handle.remove()


def cache_slices(cache, positions):
    return [(layer.keys[..., positions, :].clone(), layer.values[..., positions, :].clone())
            for layer in cache.layers]


def transplant(cache, slices, positions):
    require(len(cache.layers) == len(slices), 'donor/recipient layer mismatch')
    selected = torch.tensor(positions, device=cache.layers[0].keys.device)
    stats = {'changed_scalars': 0, 'max_abs_delta': 0.0, 'layers': len(slices), 'positions': list(positions)}
    for layer, (keys, values) in zip(cache.layers, slices, strict=True):
        for target, donor in ((layer.keys, keys), (layer.values, values)):
            old = target.index_select(-2, selected)
            require(old.shape == donor.shape and old.dtype == donor.dtype, 'cache slice shape/dtype mismatch')
            difference = old - donor
            stats['changed_scalars'] += int(torch.count_nonzero(difference))
            stats['max_abs_delta'] = max(stats['max_abs_delta'], float(difference.abs().max()))
            target.index_copy_(-2, selected, donor)
    return stats


def prepare():
    from tokenizers import Tokenizer
    source = json.loads(SOURCE.read_text())
    case = next(c for c in source['cases'] if c['case_id'] == '9813')
    tokenizer = Tokenizer.from_file(source['config']['model']['base_model'] + '/tokenizer.json')
    # First emitted driver-person row and later naturally predicted front woman.
    # They were visually reviewed on original image; no target row is injected.
    history = case['baseline_action_ids'][:9]
    require(history[-1] == BOX_END and case['baseline_action_ids'][9] == OPENER, 'not a natural closed-row/opener boundary')
    prompt = case['prompt_token_ids']
    grid = case['source_case']['image_plan']['observed_image_grid_thw']
    image_token = tokenizer.token_to_id('<|image_pad|>')
    boxes = {'owner_a': [233,269,391,448], 'owner_b': [596,305,791,717], 'background': [50,800,350,990]}
    regions = {'owner_a': region_indices(prompt, grid, image_token, boxes['owner_a'])}
    for key in ('owner_b', 'background'):
        regions[key] = region_indices(prompt, grid, image_token, boxes[key], len(regions['owner_a']))
    require(not (set(regions['owner_a']) & set(regions['owner_b'])), 'owner regions overlap')
    config = copy.deepcopy(source['config'])
    config['adapter']['path'] = source['anchor_adapter']
    packet = {'schema': 'owner_grounding_cache.v1', 'status': 'real_slice_candidate',
              'source_packet': str(SOURCE), 'source_sha256': file_hash(SOURCE),
              'config': config, 'anchor_adapter': source['anchor_adapter'],
              'cases': [{'case_id':'9813', 'role':'clean_control', 'source_case':case['source_case'], 'golden':case['golden'],
                         'prompt_ids':prompt, 'history_ids':history, 'opener':OPENER,
                         'expected_suffix':case['baseline_action_ids'][10:], 'boxes':boxes,
                         'regions':regions, 'queries':list(range(len(prompt), len(prompt)+9)),
                         'carriers': {'closing':[len(prompt)+8], 'coordinates':list(range(len(prompt)+4,len(prompt)+8))},
                         'visual_admission':'A driver seated left; B foreground woman right; background lower sand. Both boxes are saved model predictions, not GT.'}],
              'suffix_budget':512, 'smoke_budget':32,
              'source_files': {str(Path(__file__).resolve()):file_hash(__file__)}}
    write(ROOT / 'packet.json', packet)
    return {'packet':str(ROOT/'packet.json'), 'cases':1, 'region_key_counts':{k:len(v) for k,v in regions.items()}}


def prepare_panel():
    from tokenizers import Tokenizer
    packet = json.loads((ROOT/'packet.json').read_text())
    source = json.loads(SOURCE.read_text())
    control_source = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-checkpoint-history-cross/preparation-v2/packet.json')
    crossed = json.loads(control_source.read_text())
    tok = Tokenizer.from_file(packet['config']['model']['base_model']+'/tokenizer.json')
    specs = [('417044',19,[0,318,57,376],[243,388,303,440],[600,800,900,950]),
             ('477415',18,[0,795,117,999],[256,841,440,999],[750,40,950,200])]
    for iid,end,a,b,bg in specs:
        source_case = next(c for c in source['cases'] if c['case_id']==iid)
        cross = next(c for c in crossed['images'] if str(c['image_id'])==iid)
        b_tokens = [tok.token_to_id(f'<|coord_{n}|>') for n in b]
        recorded = cross['histories']['positive32']['full_action_ids']
        require(any(recorded[i:i+4] == b_tokens for i in range(len(recorded)-3)), 'B spatial control not found in saved A32 output')
        history = source_case['baseline_action_ids'][:end]
        require(history[-1] == BOX_END and source_case['baseline_action_ids'][end] == OPENER, 'panel boundary not native closed row/opener')
        start = max(i for i,t in enumerate(history) if t == OPENER)
        prompt = source_case['prompt_token_ids']
        n = len(prompt)
        boxes = {'owner_a':a,'owner_b':b,'background':bg}
        args = (prompt,source_case['source_case']['image_plan']['observed_image_grid_thw'],tok.token_to_id('<|image_pad|>'))
        regions = {'owner_a':region_indices(*args,a)}
        for arm in ('owner_b','background'):
            regions[arm] = region_indices(*args,boxes[arm],len(regions['owner_a']))
        require(not(set(regions['owner_a']) & set(regions['owner_b'])),'A/B keys overlap')
        packet['cases'].append({'case_id':iid,'role':'old_loop_with_real_initial_owner',
          'source_case':source_case['source_case'],'golden':source_case['golden'],
          'prompt_ids':prompt,'history_ids':history,'opener':OPENER,
          'expected_suffix':source_case['baseline_action_ids'][end+1:],
          'boxes':boxes,'regions':regions,'queries':list(range(n+start,n+end)),
          'carriers':{'closing':[n+end-1],'coordinates':list(range(n+end-5,n+end-1))},
          'visual_admission': ('A is the real partially clipped left-edge initial donut; later strip rows are not equated with A. B is a separate visible interior donut.' if iid=='417044' else 'A is the real lower-left chair back; B is a separate foreground chair back. Degenerate later chair rows are not assigned A identity.'),
          'b_spatial_source':{'path':str(control_source),'sha256':file_hash(control_source),'checkpoint':'positive32','tokens_used_as_input':False}})
    packet.update(status='panel_candidate',scientific_horizon='remaining total3084 primary; first512 free tokens secondary',
                  exclusions={'39654':'bad-history banana region lies on purple fruit, no true bananaA',
                              '351017':'tiny upper-left bottle region lacks convincing physical bottleA'},
                  source_files={str(Path(__file__).resolve()):file_hash(__file__)})
    write(ROOT/'panel.json',packet)
    return {'packet':str(ROOT/'panel.json'),'cases':[c['case_id'] for c in packet['cases']],
            'keys':{c['case_id']:{k:len(v) for k,v in c['regions'].items()} for c in packet['cases']},
            'maximum_forward_calls':sum(9*(3084-len(c['history_ids'])-1)+4 for c in packet['cases'])}


def generation_inputs(batch, ids):
    excluded = {'input_ids','attention_mask','position_ids','cache_position','past_key_values','rope_deltas','labels','use_cache','return_dict','logits_to_keep'}
    inputs = {k:v for k,v in batch.inputs.items() if k not in excluded}
    inputs.update(input_ids=ids, attention_mask=torch.ones_like(ids))
    return inputs


def reduce_output(packet_path, out_dir):
    from tokenizers import Tokenizer
    from src.eval.native_rows import native_detection_record as native_record
    from probes.dora_owner_learning.candidate_opportunity import score
    packet = json.loads(Path(packet_path).read_text())
    out = Path(out_dir)
    receipt = json.loads((out/'receipt.json').read_text())
    require(receipt['status'] == 'complete', 'cannot interpret incomplete execution')
    require(receipt['packet_sha256'] == file_hash(packet_path), 'consumer packet mismatch')
    tok = Tokenizer.from_file(packet['config']['model']['base_model']+'/tokenizer.json')
    summaries = []
    for case in packet['cases']:
        prefix = case['history_ids']+[case['opener']]
        prefix_text = tok.decode(prefix,skip_special_tokens=False)
        prefix_parsed = native_record(prefix_text,case['source_case'],case['golden'],'length')
        prefix_score = score(prefix_parsed,seed=None,length=len(prefix),stop='length')
        native = json.loads((out/(case['case_id']+'-native.json')).read_text())
        native_text = tok.decode(prefix+native['suffix_ids'],skip_special_tokens=False)
        native_parsed = native_record(native_text,case['source_case'],case['golden'],native['stop'])
        native_score = score(native_parsed,seed=None,length=len(prefix)+len(native['suffix_ids']),stop=native['stop'])
        for arm in ('self','owner_a','owner_b','background'):
            for carrier in case['carriers']:
                cell = json.loads((out/f"{case['case_id']}-{arm}-{carrier}.json").read_text())
                require(cell['prefix_ids'] == prefix,'consumer history mismatch')
                action = prefix+cell['suffix_ids']
                text = tok.decode(action,skip_special_tokens=False)
                require(text == cell['text'],'consumer token/text mismatch')
                parsed = native_record(text,case['source_case'],case['golden'],cell['stop'])
                scored = score(parsed,seed=None,length=len(action),stop=cell['stop'])
                secondary_action = prefix+cell['suffix_ids'][:512]
                secondary_stop = 'eos' if secondary_action[-1] == EOS else 'length'
                secondary_parsed = native_record(tok.decode(secondary_action,skip_special_tokens=False),
                    case['source_case'],case['golden'],secondary_stop)
                secondary_score = score(secondary_parsed,seed=None,length=len(secondary_action),stop=secondary_stop)
                gains = {str(t):sorted(set(scored[str(t)]['owners'])-set(native_score[str(t)]['owners'])) for t in (50,60,80)}
                losses = {str(t):sorted(set(native_score[str(t)]['owners'])-set(scored[str(t)]['owners'])) for t in (50,60,80)}
                record = {'case_id':case['case_id'],'arm':arm,'carrier':carrier,'score':scored,
                          'native_score':native_score,'prefix_score':prefix_score,'gained':gains,'lost':losses,
                          'raw_row_starts':action.count(OPENER), 'suffix_row_starts':cell['suffix_ids'].count(OPENER),
                          'first512_score':secondary_score,
                          'parsed':parsed,'suffix_equal_native':cell['suffix_ids']==native['suffix_ids']}
                summaries.append(record)
    result = {'status':'consumer_verified','receipt_sha256':file_hash(out/'receipt.json'),
              'scope':'conditional free suffix; common prefix never credited as free owner recovery',
              'records':summaries}
    write(out/'consumer.json',result)
    return {'consumer':str(out/'consumer.json'),'records':len(summaries),
            'equal_native':sum(x['suffix_equal_native'] for x in summaries)}


def execute(packet_path, out_dir, smoke):
    from src.config.inference import InferConfig
    from probes.dora_owner_learning.runtime import load_policy
    from src.inference.bound_requests import build_bound_native_requests as build_requests
    from src.qwen.native import prepare_native_inputs
    from src.qwen.generation import generate_continuations, NativeGenerationPolicy
    packet = json.loads(Path(packet_path).read_text())
    require(file_hash(packet['source_packet']) == packet['source_sha256'], 'historical packet changed')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '5' and torch.cuda.device_count() == 1, 'exclusive GPU5 required')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    runner_source = preserve_source(Path(__file__), run_root=out, relative_name='runner.py')
    shutil.copyfile(packet_path,out/'packet.json')
    start = time.monotonic()
    receipt = {'status':'running','packet':str(packet_path),'packet_sha256':file_hash(packet_path),
               'runner_sha256':file_hash(__file__),'runner_source':str(runner_source),'smoke':smoke,'model_forwards':0,'image_forwards':0,'cells':[]}
    write(out/'receipt.json',receipt)
    try:
        qwen, identity = load_policy(InferConfig.model_validate(packet['config']), device=torch.device('cuda:0'))
        model = qwen.model
        require(identity['model_identity']['adapter']['adapter_path'] == packet['anchor_adapter'], 'loaded adapter is not frozen Stable50')
        require(identity['effective_settings']['observed_attn_implementation'] == 'sdpa', 'loaded attention is not SDPA')
        write(out/'model.json',identity)
        def model_counter(*_):
            receipt['model_forwards'] += 1
        def image_counter(*_):
            receipt['image_forwards'] += 1
        handles = [model.register_forward_pre_hook(model_counter)]
        vision = [m for m in model.modules() if type(m).__name__ == 'Qwen3VLVisionModel']
        require(len(vision) == 1, 'vision model identity mismatch')
        handles.append(vision[0].register_forward_pre_hook(image_counter))
        for case in packet['cases']:
            requests, _ = build_requests(qwen, packet['config'], [case['source_case']])
            batch = prepare_native_inputs(qwen.processor, requests, device=torch.device('cuda:0'), record_media_identity=True)
            require(list(batch.prompt_token_ids[0]) == case['prompt_ids'], 'native prompt mismatch')
            history = case['history_ids']
            budget = packet['smoke_budget'] if smoke else 3084-len(history)-1
            with torch.inference_mode():
                native = generate_continuations(model,batch,extensions=[history+[case['opener']]],budgets=[budget],
                     eos_token_id=EOS,pad_token_id=qwen.tokenizer.pad_token_id,policy=NativeGenerationPolicy())[0]
                require(list(native.token_ids) == case['expected_suffix'][:budget], 'saved native history suffix parity failed')
                ids = torch.tensor([case['prompt_ids']+history],device='cuda:0')
                inputs = generation_inputs(batch,ids)
                original = model(**inputs, use_cache=True, return_dict=True, logits_to_keep=1)
                baseline_cache = original.past_key_values
                original_delta = model.model.rope_deltas.clone()
                del original
                require(baseline_cache.get_seq_length() == ids.shape[1], 'prefill cache length mismatch')
                donors = {'self': {name:cache_slices(baseline_cache,pos) for name,pos in case['carriers'].items()}}
                mask_receipts = {}
                for arm, keys in case['regions'].items():
                    mask_receipts[arm] = {}
                    with grounding_mask(model,case['queries'],keys,ids.shape[1],mask_receipts[arm]):
                        donor = model(**inputs,use_cache=True,return_dict=True,logits_to_keep=1)
                    require(torch.equal(model.model.rope_deltas,original_delta), 'masked donor MRoPE delta changed')
                    donors[arm] = {name:cache_slices(donor.past_key_values,pos) for name,pos in case['carriers'].items()}
                    del donor
                write(out/(case['case_id']+'-masks.json'),mask_receipts)
                full_ids = torch.cat([ids,torch.tensor([[case['opener']]],device='cuda:0')],dim=1)
                for arm in ('self','owner_a','owner_b','background'):
                    for carrier, positions in case['carriers'].items():
                        cache = copy.deepcopy(baseline_cache)
                        stats = transplant(cache,donors[arm][carrier],positions)
                        if arm == 'self':
                            require(stats['changed_scalars'] == 0,'self cache changed')
                        consumed = []
                        def consume_hook(_module,_args,kwargs):
                            consumed.append({'input_length':kwargs['input_ids'].shape[1],
                                             'cache_length':kwargs['past_key_values'].get_seq_length(),
                                             'cache_position':kwargs['cache_position'].tolist()})
                        consumption_handle = model.register_forward_pre_hook(consume_hook,with_kwargs=True)
                        model.model.rope_deltas = original_delta.clone()
                        before_image = receipt['image_forwards']
                        try:
                            values = model.generate(**generation_inputs(batch,full_ids),past_key_values=cache,
                                cache_position=torch.tensor([ids.shape[1]],device='cuda:0'),
                                max_new_tokens=budget,do_sample=False,repetition_penalty=1.0,
                                eos_token_id=EOS,pad_token_id=qwen.tokenizer.pad_token_id,
                                return_dict_in_generate=False,output_scores=False,output_logits=False,
                                output_hidden_states=False,output_attentions=False)
                        finally:
                            consumption_handle.remove()
                        require(receipt['image_forwards'] == before_image,'cached continuation re-encoded image')
                        require(consumed[0] == {'input_length':1,'cache_length':ids.shape[1],'cache_position':[ids.shape[1]]},'cache not consumed at common opener')
                        suffix = values[0,full_ids.shape[1]:].tolist()
                        if arm == 'self':
                            require(suffix == list(native.token_ids),'native/self-transplant generation parity failed')
                        cell = {'case_id':case['case_id'],'arm':arm,'carrier':carrier,'suffix_ids':suffix,
                                'prefix_ids':history+[case['opener']], 'text':qwen.tokenizer.decode(history+[case['opener']]+suffix,skip_special_tokens=False),
                                'stop':'eos' if suffix[-1] == EOS else 'length','cache_delta':stats,
                                'first_consumption':consumed[0],'generation_forwards':len(consumed)}
                        receipt['cells'].append(cell)
                        write(out/f"{case['case_id']}-{arm}-{carrier}.json",cell)
                        write(out/'receipt.json',receipt)
                        del cache,values
                write(out/(case['case_id']+'-native.json'),{'suffix_ids':list(native.token_ids),'stop':native.stop_reason})
                del baseline_cache,donors,batch
        for handle in handles:
            handle.remove()
        require(len(receipt['cells']) == len(packet['cases'])*8,'cell coverage mismatch')
        receipt['status'] = 'complete'
    except BaseException:
        receipt['status'] = 'technical_invalid'
        receipt['traceback'] = traceback.format_exc()
        raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved())
        write(out/'receipt.json',receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','prepare-panel','smoke','run','reduce'])
    parser.add_argument('--packet',default=str(ROOT/'packet.json'))
    parser.add_argument('--out-dir')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(),indent=2))
    elif args.command == 'prepare-panel':
        print(json.dumps(prepare_panel(),indent=2))
    elif args.command == 'reduce':
        require(args.out_dir,'explicit output required')
        print(json.dumps(reduce_output(args.packet,args.out_dir),indent=2))
    else:
        require(args.out_dir,'explicit fresh output required')
        execute(args.packet,args.out_dir,args.command=='smoke')


if __name__ == '__main__':
    main()
