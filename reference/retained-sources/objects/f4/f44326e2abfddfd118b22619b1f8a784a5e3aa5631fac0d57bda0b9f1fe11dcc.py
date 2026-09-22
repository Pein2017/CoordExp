"""Exactly two immutable Source/old-CE23 boat-history crossed continuations."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
REPO = Path('/data/CoordExp/.worktrees/research-probes')
sys.path.insert(0, str(REPO))
from probes.dora_owner_learning.candidate_opportunity import digest, file_hash, require, rows, score
from probes.dora_owner_learning.entrance_ce_eval import validate_receipt, owner_change
from probes.dora_owner_learning.route_access import CONFIG, checked_ids, checkpoint_config, publish
from probes.dora_owner_learning.reward_rows import _pred_objects
from probes.source_rweak_row_cross.owner_row_robustness import incidence, native_rows
from probes.source_rweak_row_cross.run import build_requests, native_record

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
OLD = ROOT / '2026-09-10-native-entrance-ce-feasibility'
ROBUST = ROOT / '2026-09-10-owner-row-continuation-robustness'
UNIT = REPO / 'research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-10-selective-owner-learning-autonomous/unit.md'
EID = 'coco2017_train_000000007116'
CID = EID + ':181378'
CAP = 3084
THRESHOLDS = ('50', '60', '80')


def read(path):
    return json.loads(Path(path).read_text())


def compact_identity(identity):
    return {k: identity[k] for k in ('root', 'files', 'fingerprint')}


def assert_files(files):
    for path, expected in files.items():
        require(file_hash(path) == expected, f'frozen bytes changed: {path}')


def tokenizer_only(base):
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
    return load_qwen_components_from_options(QwenLoadOptions(
        base_model=base, dtype='fp32', attn_implementation='sdpa',
        patch_embed_linearization='enabled', load_model=False))


def canonical_row(raw, history, model, prefix, provenance):
    ids = checked_ids(raw['action_ids'], raw['stop_reason'])
    require(ids[:len(prefix)] == prefix, 'retained diagonal prefix mismatch')
    return dict(history=history, checkpoint=model, provenance=provenance,
                action_ids=ids, prefix_ids=prefix, suffix_ids=ids[len(prefix):],
                remaining_budget=CAP-len(prefix), text=raw['text'],
                stop_reason=raw['stop_reason'], parsed=raw['parsed'])


def consume(record, packet, tokenizer):
    history = packet['histories'][record['history']]
    prefix = history['prefix_ids']
    require(record['prefix_ids'] == prefix and record['action_ids'] == prefix+record['suffix_ids'],
            'raw prefix/suffix concatenation')
    require(record['remaining_budget'] == CAP-len(prefix), 'remaining budget changed')
    ids = checked_ids(record['action_ids'], record['stop_reason'])
    require(len(record['suffix_ids']) <= record['remaining_budget'], 'suffix cap')
    text = tokenizer.decode(ids, skip_special_tokens=False)
    require(text == record['text'], 'raw token/text identity')
    frozen = packet['frozen_case']
    parsed = native_record(text, frozen['case'], frozen['baseline'], record['stop_reason'])
    require(parsed == record['parsed'], 'raw native consumer identity')
    before = history['prefix_parsed']
    count = len(before['pred'])
    require(parsed['pred'][:count] == before['pred'], 'given prefix predictions changed')
    full = score(parsed, seed=-1, length=len(ids), stop=record['stop_reason'])
    old = history['prefix_score']
    suffix_pred = parsed['pred'][count:]
    projection = [j for j, obj in enumerate(parsed['pred']) if _pred_objects(dict(parsed, pred=[obj]))[0]]
    all_owners = {str(g['object_id']) for g in parsed['gt']}
    result = dict(history=record['history'], checkpoint=record['checkpoint'], provenance=record['provenance'],
                  full_score=full, prefix_score=old,
                  suffix=dict(tokens=len(record['suffix_ids']), valid_predictions=len(suffix_pred),
                              parser_drops=parsed['dropped_prediction_count']-before['dropped_prediction_count'],
                              new_strict_repeats=full['strict_repeats']-old['strict_repeats'],
                              given_prefix_strict_repeats=old['strict_repeats'],
                              row_coordinates=[dict(generated_order=p['generated_order'], description=p['description'],
                                                    bins=p['coord_bins']) for p in suffix_pred]),
                  thresholds={})
    for t in THRESHOLDS:
        prefix_owners = set(old[t]['owners'])
        complete_owners = set(full[t]['owners'])
        remaining = all_owners-prefix_owners
        direct = incidence(parsed, start=count, threshold=int(t)/100)
        assigned = sorted(m['owner'] for m in full[t]['matches'] if projection[m['pred_index']] >= count)
        gained = complete_owners-prefix_owners
        result['thresholds'][t] = dict(
            prefix_covered=sorted(prefix_owners), remaining_obligations=sorted(remaining),
            complete_owner_change=owner_change(complete_owners, prefix_owners),
            suffix_globally_assigned_owners=assigned,
            suffix_direct_incidence=direct,
            remaining_recovered_and_suffix_supported=sorted(o for o in remaining & complete_owners if direct[o]),
            gained_without_suffix_incidence=sorted(o for o in gained if not direct[o]),
            prefix_direct_incidence=history['prefix_direct_incidence'][t])
    require(result['suffix']['new_strict_repeats'] >= 0 and result['suffix']['parser_drops'] >= 0,
            'prefix contribution subtraction')
    return result


def prepare():
    require(not (OUT/'packet.json').exists(), 'packet already frozen')
    evaluation = read(OLD/'evaluation/manifest.json')
    training = read(OLD/'training/receipt.json')
    require(training['updates'] == 23, 'wrong old CE checkpoint')
    validate_receipt(training, evaluation)
    frozen = next(r for r in evaluation['records'] if r['example_id'] == EID)
    base = evaluation['source_model']['base_model_path']
    qwen = tokenizer_only(base)
    require(not qwen.load_model and qwen.model is None, 'CPU preparation loaded model')
    good = next(r for r in rows(ROBUST/'execution/rows.jsonl') if r['case_id'] == CID and r['arm'] == 'A')
    new = next(r for r in rows(OLD/'evaluation/execution/rows.jsonl') if r['example_id'] == EID)
    robust_manifest = read(ROBUST/'manifest-v2.json')
    selected = next(r for r in robust_manifest['selected'] if r['case_id'] == CID)
    require(good['manifest_sha256'] == digest(robust_manifest) and new['manifest_sha256'] == digest(evaluation),
            'retained manifest identity')
    require(good['prefix_ids'] == selected['prefix_ids'] and good['forced_ids'] == selected['A']['ids'],
            'retained complete A identity')
    require(selected['group']['prompt_token_ids'] == frozen['prompt_token_ids'], 'history prompt mismatch')
    hgood = good['prefix_ids']+good['forced_ids']
    close = qwen.tokenizer.convert_tokens_to_ids('<|box_end|>')
    ends = [j+1 for j, token in enumerate(new['action_ids']) if token == close]
    require(len(ends) >= 5, 'old CE target complete row unavailable')
    hnew = new['action_ids'][:ends[4]]
    target = new['parsed']['pred'][4]
    require(target['generated_order'] == 4 and target['description'] == 'boat' and
            target['coord_bins'] == [298,460,371,556], 'frozen native boat identity')
    require(qwen.tokenizer.decode(hnew, skip_special_tokens=False) == new['text'][:target['char_end']],
            'exact complete row token/character boundary')
    histories = {}
    for name, prefix in [('H_good',hgood), ('H_new',hnew)]:
        token_rows = native_rows(prefix, qwen.tokenizer.backend_tokenizer)
        require(token_rows[-1]['stop'] == len(prefix) and token_rows[-1]['coords'] ==
                (selected['A']['coords'] if name == 'H_good' else [298,460,371,556]), 'prefix row endpoint')
        text = qwen.tokenizer.decode(prefix, skip_special_tokens=False)
        parsed = native_record(text, frozen['case'], frozen['baseline'], 'conditional')
        require(parsed['dropped_prediction_count'] == 0 and len(parsed['pred']) == len(token_rows), 'prefix parse')
        pscore = score(parsed, seed=-1, length=len(prefix), stop='conditional')
        histories[name] = dict(prefix_ids=prefix, prefix_token_sha256=digest(prefix), prefix_text=text,
                               complete_rows=len(token_rows), prefix_parsed=parsed, prefix_score=pscore,
                               prefix_direct_incidence={t:incidence(parsed, threshold=int(t)/100) for t in THRESHOLDS})
        require('181378' in pscore['50']['owners'], 'target boat not completed before release')
    require(len(hgood) == 27 and len(hnew) == 45, 'unexpected fixed complete-prefix lengths')
    request, _ = build_requests(qwen, evaluation['configs']['train'], [frozen['case']])
    require(list(request[0].expected_token_ids) == frozen['prompt_token_ids'], 'CPU native prompt identity')
    from src.qwen.native import prepare_native_inputs
    batch = prepare_native_inputs(qwen.processor, request, device='cpu', record_media_identity=True)
    require(list(batch.prompt_token_ids[0]) == frozen['prompt_token_ids'] and
            batch.media_sha256[0] == frozen['case']['image_plan']['executed_media_sha256'], 'CPU native media identity')
    paths = [OLD/'training/receipt.json', OLD/'evaluation/manifest.json', OLD/'evaluation/execution/rows.jsonl',
             OLD/'evaluation/execution/consumer.json', ROBUST/'manifest-v2.json', ROBUST/'execution/rows.jsonl',
             ROBUST/'execution/consumer.json', Path(__file__), CONFIG, Path(frozen['case']['image_path'])]
    modules = ['probes/dora_owner_learning/candidate_opportunity.py', 'probes/dora_owner_learning/reward_rows.py',
               'probes/dora_owner_learning/runtime.py', 'probes/dora_owner_learning/train.py',
               'probes/dora_owner_learning/route_access.py', 'probes/dora_owner_learning/entrance_ce_eval.py',
               'probes/source_rweak_row_cross/run.py', 'probes/source_rweak_row_cross/owner_row_robustness.py',
               'src/adapters/dora.py', 'src/inference/parsing.py']
    paths += [REPO/p for p in modules] + list((REPO/'src/qwen').glob('*.py'))
    paths += [p for p in Path(base).iterdir() if p.is_file()]
    for identity in (training['adapter'], training['source_adapter'], training['source_embedding']):
        for f in identity['files']:
            path = Path(identity['root'])/f['relative_path']
            require(file_hash(path) == f['sha256'], 'original checkpoint bytes changed')
            paths.append(path)
    packet = dict(schema='boat_cross.packet.v1', frozen_case=frozen, config=evaluation['configs']['train'],
                  histories=histories, base_model=base,
                  checkpoints={'Source':compact_identity(training['source_adapter']),
                               'old_CE23':compact_identity(training['adapter'])},
                  embedding=compact_identity(training['source_embedding']),
                  input_files={str(p.resolve()):file_hash(p) for p in paths},
                  authorizing_unit=dict(path=str(UNIT), sha256=file_hash(UNIT)),
                  jobs=[dict(checkpoint='Source',history='H_new'),dict(checkpoint='old_CE23',history='H_good')],
                  limits=dict(model_loads=2, continuations=2, new_tokens=6168, seconds=1800, visible_gpu='2'),
                  conditioning_scope='Different histories have different covered owners, geometry and workloads; only within-history checkpoint contrasts.',
                  cpu_preflight=dict(native_prompt=True,native_media=True,model_loads=0,model_forwards=0))
    diags = [canonical_row(good,'H_good','Source',hgood,'retained_Source_full_A'),
             canonical_row(new,'H_new','old_CE23',hnew,'retained_old_CE23_native')]
    reduced = [consume(r,packet,qwen.tokenizer) for r in diags]
    old_scores = [next(r for r in read(ROBUST/'execution/consumer.json') if r['case_id'] == CID and r['arm'] == 'A')['score'],
                  next(r for r in read(OLD/'evaluation/execution/consumer.json') if r['example_id'] == EID)['score']]
    require([r['full_score'] for r in reduced] == old_scores, 'retained native consumer score changed')
    require(reduced[0]['suffix']['new_strict_repeats'] == 0 and reduced[1]['suffix']['new_strict_repeats'] == 21,
            'retained new-suffix repeat accounting')
    publish(OUT/'packet.json',packet)
    with (OUT/'retained-diagonals.jsonl').open('x') as stream:
        for row in diags:stream.write(json.dumps(row)+'\n')
    publish(OUT/'diagonal-consumer.json',reduced)
    publish(OUT/'preflight.json',dict(status='ready',packet_sha256=digest(packet),packet_file_sha256=file_hash(OUT/'packet.json'),
                                    diagonal_rows=2,cpu_only=True,histories={k:dict(tokens=len(v['prefix_ids']),rows=v['complete_rows'],
                                    owners50=v['prefix_score']['50']['owners']) for k,v in histories.items()}))
    print(json.dumps(read(OUT/'preflight.json')))


def generate_one(packet, job, terminal):
    import torch
    from src.config.inference import load_research_infer_config
    from src.adapters.dora import select_dora_parameters
    from src.qwen.native import prepare_native_inputs
    from src.qwen.generation import generate_continuations, NativeGenerationPolicy
    from probes.dora_owner_learning.runtime import load_policy
    from probes.dora_owner_learning.train import _tensor_state_hash, _parameter_layout
    checkpoint, history = job['checkpoint'], job['history']
    label = checkpoint+'__'+history
    config = checkpoint_config(load_research_infer_config(CONFIG).config,packet['checkpoints'][checkpoint]['root'])
    require(str(config.model.base_model) == packet['base_model'] and str(config.embedding_delta.path) == packet['embedding']['root'],
            'base/embedding config identity')
    require(config.backend.hf.patch_embed_linearization == 'enabled', 'patch linearization disabled')
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    terminal['model_load_attempts'] += 1
    qwen, identity = load_policy(config,device=torch.device('cuda:0'))
    terminal['model_loads'] += 1
    require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names'] == ['torch.float32'] and
            identity['effective_settings']['observed_attn_implementation'] == 'sdpa', 'cold FP32/SDPA')
    require(identity['model_identity']['adapter']['adapter_path'] == packet['checkpoints'][checkpoint]['root'] and
            not identity['model_identity']['adapter']['merged_adapters'], 'cold unmerged adapter identity')
    from src.qwen.patches import LinearizedQwen3VLPatchEmbed
    visual = [m for n,m in qwen.model.named_modules() if n.endswith('visual')]
    require(len(visual) == 1 and isinstance(visual[0].patch_embed, LinearizedQwen3VLPatchEmbed), 'cold patch activation')
    training = read(OLD/'training/receipt.json')
    named = select_dora_parameters(qwen.model,towers=('language',),adapter_name='default')
    require(_parameter_layout(named) == training['trainable_layout'], 'cold DoRA tensor layout')
    selected_ids = {id(p) for _,p in named}
    frozen = [(n,p) for n,p in qwen.model.named_parameters() if id(p) not in selected_ids]
    hashes = dict(adapter=_tensor_state_hash(named),frozen=_tensor_state_hash(frozen))
    require(hashes['adapter'] == training['adapter_tensor_hash_before' if checkpoint == 'Source' else 'adapter_tensor_hash_after'] and
            hashes['frozen'] == training['frozen_tensor_hash_before'], 'cold adapter/base/embedding tensor bytes')
    counters = dict(model_forwards=0,image_forwards=0)
    def model_count(*_):counters['model_forwards'] += 1
    def image_count(*_):counters['image_forwards'] += 1
    qwen.model.register_forward_pre_hook(model_count)
    visual[0].register_forward_pre_hook(image_count)
    qwen.model.eval()
    for p in qwen.model.parameters():p.requires_grad_(False)
    frozen_case = packet['frozen_case']
    requests, metadata = build_requests(qwen,packet['config'],[frozen_case['case']])
    require(list(requests[0].expected_token_ids) == frozen_case['prompt_token_ids'], 'cold exact prompt')
    batch = prepare_native_inputs(qwen.processor,requests,device=torch.device('cuda:0'),record_media_identity=True)
    require(list(batch.prompt_token_ids[0]) == frozen_case['prompt_token_ids'] and
            batch.media_sha256[0] == frozen_case['case']['image_plan']['executed_media_sha256'], 'cold exact media')
    publish(OUT/(label+'-cold.json'),dict(identity=identity,components=qwen.to_artifact_dict(),tensor_hashes=hashes,
            config=config.model_dump(mode='json'),prompt_metadata=metadata,model_forwards_before_generation=counters['model_forwards']))
    prefix = packet['histories'][history]['prefix_ids']
    remaining = CAP-len(prefix)
    require(terminal['continuation_attempts'] < 2 and terminal['new_tokens']+remaining <= 6168,'finite invocation')
    terminal['continuation_attempts'] += 1
    result = generate_continuations(qwen.model,batch,extensions=[prefix],budgets=[remaining],eos_token_id=151645,
            pad_token_id=qwen.tokenizer.pad_token_id,policy=NativeGenerationPolicy(temperature=0.,top_p=1.,repetition_penalty=1.),trace='none')[0]
    torch.cuda.synchronize()
    terminal['continuations'] += 1
    terminal['new_tokens'] += len(result.token_ids)
    terminal['model_forwards'] += counters['model_forwards']
    terminal['image_forwards'] += counters['image_forwards']
    require(result.request_id == EID,'generation request association')
    ids = prefix+list(result.token_ids)
    text = qwen.tokenizer.decode(ids,skip_special_tokens=False)
    row = dict(history=history,checkpoint=checkpoint,provenance='new_crossed_continuation',
               packet_sha256=digest(packet),action_ids=ids,prefix_ids=prefix,suffix_ids=list(result.token_ids),
               remaining_budget=remaining,text=text,stop_reason=result.stop_reason,
               parsed=native_record(text,frozen_case['case'],frozen_case['baseline'],result.stop_reason))
    consumption = consume(row,packet,qwen.tokenizer)
    require(counters['image_forwards'] == 1 and counters['model_forwards'] == len(result.token_ids),'forward/token accounting')
    resources = dict(**counters,seconds=time.monotonic()-started,new_tokens=len(result.token_ids),
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                     rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    publish(OUT/(label+'-resources.json'),resources)
    return row, consumption, resources


def execute():
    import torch
    packet = read(OUT/'packet.json')
    require(read(OUT/'preflight.json')['packet_sha256'] == digest(packet), 'packet preflight mismatch')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '2' and torch.cuda.is_available() and torch.cuda.device_count() == 1,'GPU2 only')
    require(not (OUT/'launch.json').exists(), 'execution already attempted; no automatic retry')
    assert_files(packet['input_files'])
    started = time.monotonic()
    terminal = dict(status='running',pid=os.getpid(),packet_sha256=digest(packet),model_load_attempts=0,model_loads=0,
                    continuation_attempts=0,continuations=0,new_tokens=0,model_forwards=0,image_forwards=0,per_cell=[])
    publish(OUT/'launch.json',dict(**terminal,visible_devices='2',time=time.time(),
            gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)))
    def expired(*_):raise TimeoutError('1800 second two-cross cumulative budget')
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(1800)
    try:
        with (OUT/'raw.jsonl').open('x') as stream:
            for job in packet['jobs']:
                row, consumed, resources = generate_one(packet,job,terminal)
                stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                terminal['per_cell'].append(dict(**job,**resources))
                publish(OUT/(job['checkpoint']+'__'+job['history']+'-consumer.json'),consumed)
                del row, consumed
                gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
                require(torch.cuda.memory_allocated() == 0,'previous model remains live before next load')
        require(terminal['model_loads'] == terminal['continuations'] == 2,'exact two-cell completion')
        assert_files(packet['input_files'])
        terminal['status'] = 'completed'
    except BaseException as exc:
        terminal.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        terminal.update(elapsed_seconds=time.monotonic()-started,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        artifact_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()))
        publish(OUT/'terminal.json',terminal)


def verify():
    packet = read(OUT/'packet.json')
    assert_files(packet['input_files'])
    terminal = read(OUT/'terminal.json')
    require(terminal['status'] == 'completed' and terminal['model_loads'] == terminal['continuations'] == 2,'execution incomplete')
    require(terminal['new_tokens'] <= 6168 and terminal['elapsed_seconds'] <= 1800,'invocation budget violation')
    tokenizer = tokenizer_only(packet['base_model']).tokenizer
    diags, fresh = rows(OUT/'retained-diagonals.jsonl'),rows(OUT/'raw.jsonl')
    require([(r['checkpoint'],r['history']) for r in fresh] == [('Source','H_new'),('old_CE23','H_good')], 'cross coverage/order')
    require(all(r['packet_sha256'] == digest(packet) for r in fresh),'cross packet identity')
    consumed = [consume(r,packet,tokenizer) for r in diags+fresh]
    require(consumed[:2] == read(OUT/'diagonal-consumer.json'),'retained diagonal readback')
    require(sum(len(r['suffix_ids']) for r in fresh) == terminal['new_tokens'] == terminal['model_forwards'],'exact raw forward/token count')
    contrast = {}
    for history in ('H_good','H_new'):
        cells = {r['checkpoint']:r for r in consumed if r['history'] == history}
        source,candidate = cells['Source'],cells['old_CE23']
        contrast[history] = dict(source=source,candidate=candidate,
            owner_changes={t:owner_change(candidate['full_score'][t]['owners'],source['full_score'][t]['owners']) for t in THRESHOLDS},
            metric_deltas={t:{k:candidate['full_score'][t][k]-source['full_score'][t][k] for k in ('tp','fp','fn','f1')} for t in THRESHOLDS},
            suffix_burden_deltas={k:candidate['suffix'][k]-source['suffix'][k] for k in ('valid_predictions','parser_drops','new_strict_repeats','tokens')})
    payload = dict(schema='boat_cross.consumer.v1',cells=consumed,contrasts=contrast,
                   limitation=packet['conditioning_scope'],suffix_accounting='Given prefixes are identical within each row; only newly generated repeat rows are attributed to continuation.')
    if (OUT/'consumer.json').exists():require(read(OUT/'consumer.json') == payload,'consumer readback changed')
    else:publish(OUT/'consumer.json',payload)
    receipt = dict(status='candidate',technical='CPU-verified after two completed native crossed continuations',
                   packet_sha256=digest(packet),packet_file_sha256=file_hash(OUT/'packet.json'),raw_sha256=file_hash(OUT/'raw.jsonl'),
                   consumer_sha256=file_hash(OUT/'consumer.json'),terminal_sha256=file_hash(OUT/'terminal.json'),
                   source_diagonal_reruns=0,old_CE_diagonal_reruns=0,new_generations=2,model_loads=2,
                   new_tokens=terminal['new_tokens'],model_forwards=terminal['model_forwards'],image_forwards=terminal['image_forwards'],
                   elapsed_seconds=terminal['elapsed_seconds'],scientific_acceptance='lead-owned; no learning-arm or architecture promotion')
    if (OUT/'receipt.json').exists():require(read(OUT/'receipt.json') == receipt,'receipt changed')
    else:publish(OUT/'receipt.json',receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','execute','verify'))
    globals()[parser.parse_args().command]()
