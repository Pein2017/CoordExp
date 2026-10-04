"""Frozen two-cue, two-region diagnostic at the native first-bottle prefix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from probes.coordinate_diagnostics import cued_visual as reused
from src.qwen.images import rgb_image_sha256

a, ROOT = reused.a, reused.ROOT
OUTPUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-04/cue-region-specificity'
UNIT = ROOT / 'research/experiments/2026-10-04-cue-region-specificity'
SCHEMA = 'cue-region-specificity-v1'
SOURCE_PATHS = ['probes/coordinate_diagnostics/cue_specificity.py',
                'tests/probes/coordinate_diagnostics/test_cue_specificity.py']
BOUNDS = dict(reused.BOUNDS, requests=6, actions=114, maximum_actions_per_request=19,
              maximum_context_tokens=1381, wall_seconds=300, retained_bytes=64 * 1024**2)
CUES = {'A': ('target', [186, 30, 207, 106]), 'B': ('control', [495, 396, 543, 614])}
RECT, FILL = [618, 329, 678, 511], [138, 115, 108]
binding, revision = reused.binding, reused.revision
NativeSession = reused.NativeSession


def producer_identity():
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *SOURCE_PATHS], cwd=ROOT)
    return dict(commit=revision(), diff_sha256=hashlib.sha256(diff).hexdigest(),
                files={p: a.digest(ROOT / p) for p in SOURCE_PATHS})


def definitions(anchor):
    result = []
    for region in ('clean', 'mask-A', 'mask-B'):
        for cue, (owner, box) in CUES.items():
            cell = dict(condition=cue+'-'+region, image_id=351017, cue=cue, input_region=region,
                mode='cued', owner=owner, target_box=box, position=14, budget=19, row_start=9,
                observations=[14, 15, 16, 17], population='supplied_x1',
                forced_actions={str(i): t for i, t in enumerate(anchor['token_ids'][:14])},
                expected_ids=anchor['token_ids'][:14])
            cell['forced_actions']['14'] = reused.COORD_START + box[0]
            if cell['condition'] == 'A-clean':
                cell.update(expected_ids=anchor['token_ids'],
                    reference_median_winners=[s['argmax'] for s in anchor['median_steps']],
                    reference_median_logprobs=[s['pre_force_logprob'] for s in anchor['median_steps']],
                    reference_raw_logprobs=anchor['raw_logprobs'])
            result.append(cell)
    return result


def regenerated_B(clean):
    import numpy as np
    site = dict(target=RECT, background=RECT, fill=FILL,
                ring_counts=[4128, 4128], pixel_L1=[1442926, 1442926])
    statistics = reused.visual.mask_statistics(clean, site)
    mask = reused.visual.rectangle_mask(clean.shape, RECT)
    rgb = clean.copy(); rgb[mask] = FILL
    if int(mask.sum()) != 10920 or int(np.any(rgb != clean, axis=-1).sum()) != 10920:
        raise ValueError('B frozen mask/changed-pixel count differs')
    return rgb, mask, statistics


def verify_inputs(packet):
    import numpy as np
    from PIL import Image
    old = a.load(packet['bindings']['prior_packet']['path'])
    preflight = a.load(packet['bindings']['mask_preflight']['path'])
    review = a.load(packet['bindings']['visual_review']['path'])
    if (packet['schema'] != SCHEMA or packet['bounds'] != BOUNDS or packet['model_loaded'] is not False or
            packet['coordinate_ids'] != reused.COORD_IDS or
            packet['summary_convention'] != old['summary_convention'] or
            packet['images'] != {'351017': old['images']['351017']} or
            packet['original_media'] != {'351017': old['original_media']['351017']}):
        raise ValueError('frozen matrix/image/runtime observation contract differs')
    anchor = a.load(packet['bindings']['reference-351017-cued']['path'])
    if packet['conditions'] != definitions(anchor):
        raise ValueError('native history/cue boundary/order differs')
    native = a.load(reused.readout.saved_path(16, 351017))
    if anchor['token_ids'][:14] != native['token_ids'][:14]:
        raise ValueError('anchor is not the literal native first-bottle prefix')
    for region, previous in [('clean', 'clean'), ('mask-A', 'target')]:
        if (packet['pixels'][region] != old['pixels']['351017']['variants'][previous] or
                packet['requests']['351017'][region] != old['requests']['351017'][previous] or
                packet['processor']['351017'][region] != old['processor']['351017'][previous]):
            raise ValueError('clean/A reused input identity differs')
    if (preflight['rectangle'] != RECT or preflight['fill'] != FILL or
            preflight['mask_area'] != 10920 or preflight['pixel_L1'] != 1442926 or
            review['status'] != 'lead-accepted-region-mask' or
            review['mask_identity'] != preflight['array'] or review['gallery'] != preflight['gallery']):
        raise ValueError('frozen B preflight/visual review differs')
    clean = np.load(packet['pixels']['clean']['array']['path'], allow_pickle=False)
    rgb_B, mask_B, statistics = regenerated_B(clean)
    if (not np.array_equal(rgb_B, np.load(preflight['array']['path'], allow_pickle=False)) or
            not np.array_equal(mask_B, np.load(preflight['mask']['path'], allow_pickle=False)) or
            packet['B_statistics'] != statistics):
        raise ValueError('regenerated B pixels/mask/preflight differ')
    for region, variant in packet['pixels'].items():
        for key in ('image', 'array', 'mask'):
            if a.digest(variant[key]['path']) != variant[key]['sha256']:
                raise ValueError('bound input array/image/mask changed')
        rgb = np.load(variant['array']['path'], allow_pickle=False)
        mask = np.load(variant['mask']['path'], allow_pickle=False)
        with Image.open(variant['image']['path']) as im:
            decoded = np.array(im.convert('RGB'))
            pixel_sha256 = rgb_image_sha256(im)
        if not np.array_equal(rgb, decoded) or not np.array_equal(rgb[~mask], clean[~mask]):
            raise ValueError('lossless RGB/complement changed')
        if region == 'mask-B' and (not np.array_equal(rgb, rgb_B) or not np.array_equal(mask, mask_B)):
            raise ValueError('executed B differs from deterministic source pixels')
        if pixel_sha256 != variant['pixel_sha256']:
            raise ValueError('decoded media identity differs')
        request = packet['requests']['351017'][region]; identity = packet['processor']['351017'][region]
        if (request['image_path'] != variant['image']['path'] or request['image_sha256'] != variant['image']['sha256'] or
                request['media_sha256'] != variant['pixel_sha256'] or identity['media_sha256'] != variant['pixel_sha256'] or
                identity['prompt_token_ids'] != old['processor']['351017']['clean']['prompt_token_ids'] or
                identity['image_grid_thw'] != old['processor']['351017']['clean']['image_grid_thw']):
            raise ValueError('image/request/prompt/grid association differs')
    for b in packet['bindings'].values():
        if a.digest(b['path']) != b['sha256']:
            raise ValueError('bound input/preflight/anchor changed')


def prepare(directory):
    import numpy as np
    from PIL import Image
    from probes import iterative_positive as p, rollout_row_credit as retained
    oldpath = reused.OUTPUT/'prepared-01/input-packet.json'
    _, old = reused.load_packet(reused.OUTPUT/'native-release-01.json', cpu=True)
    q = retained.frontend()
    if q.model is not None: raise ValueError('CPU preparation loaded a model')
    directory = Path(directory).resolve()
    if not directory.is_relative_to(OUTPUT): raise ValueError('preparation outside task owner')
    directory.mkdir(parents=True, exist_ok=False)
    pixels = {r: old['pixels']['351017']['variants'][k] for r,k in [('clean','clean'),('mask-A','target')]}
    clean = np.load(pixels['clean']['array']['path'], allow_pickle=False)
    rgb, mask, statistics = regenerated_B(clean)
    np.save(directory/'region-B.npy', rgb, allow_pickle=False)
    np.save(directory/'region-B-mask.npy', mask, allow_pickle=False)
    Image.fromarray(rgb).save(directory/'region-B.png')
    pixels['mask-B'] = dict(image=binding(directory/'region-B.png'), array=binding(directory/'region-B.npy'),
        mask=binding(directory/'region-B-mask.npy'), pixel_sha256=rgb_image_sha256(Image.fromarray(rgb)),
        changed_pixel_count=10920)
    requests = {r: old['requests']['351017'][k] for r,k in [('clean','clean'),('mask-A','target')]}
    requests['mask-B'] = dict(requests['clean'], image_path=pixels['mask-B']['image']['path'],
        image_sha256=pixels['mask-B']['image']['sha256'], media_sha256=pixels['mask-B']['pixel_sha256'],
        request_id='cue-region-specificity:351017:mask-B')
    processor = {r: reused.visual.batch_identity(p.native_request(v, p.load(p.POLICY), q.processor)) for r,v in requests.items()}
    anchorpath = reused.OUTPUT/'native-01/bottle-cued-clean.json'; anchor = a.load(anchorpath)
    preflight = OUTPUT/'lead-mask-inspection-01/mask.json'; pre = a.load(preflight)
    bindings = dict(old['bindings'], protocol=binding(UNIT/'unit.md'), prior_packet=binding(oldpath),
        mask_preflight=binding(preflight), visual_review=binding(preflight.parent/'lead-visual-review-01.json'),
        **{'reference-351017-cued': binding(anchorpath), 'anchor_scores': anchor['score_tensors']})
    bindings.update({'B-'+k:pre[k] for k in ('array','mask','image','gallery')})
    packet = dict(schema=SCHEMA, bounds=BOUNDS, conditions=definitions(anchor), bindings=bindings,
        runtime=old['runtime'], payloads=old['payloads'], cached_pipeline_files=dict(old['cached_pipeline_files'],
            **{'probes/coordinate_diagnostics/cued_visual.py':a.digest(ROOT/'probes/coordinate_diagnostics/cued_visual.py')}),
        producer=producer_identity(), pixels=pixels, B_statistics=statistics, images={'351017':old['images']['351017']},
        requests={'351017':requests}, processor={'351017':processor}, original_media={'351017':old['original_media']['351017']},
        model_loaded=False, **{k:old[k] for k in ('coordinate_ids','eos_id','object_start_id','vocabulary_size','summary_convention')})
    verify_inputs(packet)
    a.write(directory/'input-packet.json',packet)
    proposal = dict(schema=SCHEMA, released=False, source_revision=revision(), producer_files=packet['producer']['files'],
        input_packet=binding(directory/'input-packet.json'), runtime=packet['runtime'], bounds=BOUNDS,
        output=str(OUTPUT/'native-01'), retry='no_automatic_relaunch')
    a.write(directory/'native-proposal.json',proposal)
    return proposal


def load_packet(path, *, cpu=False):
    from probes.rule_stability.__main__ import runtime_identity
    config=a.load(path)
    if config['schema']!=SCHEMA or config['bounds']!=BOUNDS or config['retry']!='no_automatic_relaunch':
        raise ValueError('release/resource contract differs')
    if a.digest(config['input_packet']['path'])!=config['input_packet']['sha256']:
        raise ValueError('input packet changed')
    packet=a.load(config['input_packet']['path']);verify_inputs(packet)
    if config['runtime']!=runtime_identity() or config['runtime']!=packet['runtime']:
        raise ValueError('current runtime differs')
    if config['producer_files']!=producer_identity()['files'] or config['producer_files']!=packet['producer']['files']:
        raise ValueError('producer bytes differ')
    for path,sha in packet['cached_pipeline_files'].items():
        if a.digest(ROOT/path)!=sha:raise ValueError('cached helper changed')
    reused.continuity.check_payloads(packet['payloads'])
    if not cpu:
        if config['released'] is not True or config['source_revision']!=revision():
            raise ValueError('exact clean lead release required')
        if subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT):
            raise ValueError('native source dirty')
        subprocess.run(['git','ls-files','--error-unmatch',*SOURCE_PATHS],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('single GPU0 required')
    return config,packet


def validate_record(record, cell, packet):
    if record['schema']!=SCHEMA:raise ValueError('wrong task record schema')
    # The reused schema names a mechanics encoding; task provenance stays above.
    reused.validate_record(dict(record,schema=reused.SCHEMA),cell,packet)


@reused.cpu_diagnostics()
def endpoints(rows):
    indexed={r['condition']:r for r in rows if r['status']=='completed' and r['control']['qualified']}
    effects={r+','+c:None for r in ('mask-A','mask-B') for c in CUES}
    pairs={};margins={};unavailable={};divergence={}
    for region in ('mask-A','mask-B'):
        for cue in CUES:
            key=region+','+cue;names=[cue+'-clean',cue+'-'+region]
            if any(n not in indexed for n in names):
                unavailable[key]='missing_or_unqualified_pair';continue
            left,right=[indexed[n] for n in names];pair=reused.compare(left,right);pairs[key]=pair
            l,r=left['record']['token_ids'],right['record']['token_ids']
            divergence[key]=next((i for i,(x,y) in enumerate(zip(l,r)) if x!=y),
                min(len(l),len(r)) if len(l)!=len(r) else None)
            y1=[x for x in pair if x['position']==15 and x.get('role')=='y1']
            if not y1:unavailable[key]='unavailable_causal_y1';continue
            if not all(x['prefix_exact'] for x in y1):raise ValueError('first-free y1 paired prefix differs')
            effects[key]=next(x['coordinate_conditional_TV'] for x in y1 if x['channel']=='median')
            margins[key]={ch:{str(v):right['margins'][ch][1][str(v)]-left['margins'][ch][1][str(v)] for v in (30,396)} for ch in ('raw','median')}
    return dict(E=effects,D_A=effects['mask-A,A']-effects['mask-A,B'] if all(effects[k] is not None for k in ('mask-A,A','mask-A,B')) else None,
        D_B=effects['mask-B,B']-effects['mask-B,A'] if all(effects[k] is not None for k in ('mask-B,A','mask-B,B')) else None,
        paired_scores=pairs,unavailable=unavailable,first_free_divergence=divergence,
        fixed_margin_mask_minus_clean=margins,signs_are_not_acceptance_gates=True)


@reused.cpu_diagnostics()
def consume(output, packet, tokenizer):
    verify_inputs(packet);output=Path(output);entries=a.load(output/'conditions.json');rows=[];anchor=False
    if [e['condition'] for e in entries]!=[c['condition'] for c in packet['conditions']]:raise ValueError('published matrix/order differs')
    for entry,cell in zip(entries,packet['conditions'],strict=True):
        name=cell['condition']
        if entry['status']=='skipped-HOLD':
            if anchor and not entry['reason'].startswith(('resource_limit','technical_failure')):raise ValueError('unjustified HOLD')
            rows.append(entry);continue
        if entry['status']!='completed' or (name!='A-clean' and not anchor) or entry['filename']!=name+'.json':
            raise ValueError('published invalid cell/dependency')
        record=a.load(output/entry['filename']);validate_record(record,cell,packet)
        if tokenizer.decode(record['token_ids'],skip_special_tokens=False)!=record['text']:raise ValueError('full original decode differs')
        control=reused.fidelity(record,cell,packet) if name=='A-clean' else dict(qualified=True,quality_gate=False)
        if name=='A-clean':anchor=control['qualified']
        analysis=reused.owner_entry.analyze(record,cell,packet,tokenizer)
        margins={ch:[{str(v):o['coordinate_scores'][v]-o['coordinate_scores'][0] for v in (30,396)} for o in record[ch+'_observations']] for ch in ('raw','median')}
        selected=analysis['selected_row']
        if selected:
            selected['both_designated_overlaps']={cue:next(x['iou'] for x in selected['same_category_overlaps'] if x['annotation_id']==reused.owner_entry.OWNERS[owner]['ann_id']) for cue,(owner,box) in CUES.items()}
        summaries={ch:[dict(position=o['position'],role=reused.causal_roles(record,cell).get(str(o['position'])),**reused.continuity.score_summary(o,record[ch+'_steps'][o['position']]),target_metrics=o['target_metrics']) for o in record[ch+'_observations']] for ch in ('raw','median')}
        rows.append(dict(condition=name,status='completed',cue=cell['cue'],input_region=cell['input_region'],control=control,
            record=record,analysis=analysis,causal_roles=reused.causal_roles(record,cell),margins=margins,scores=summaries,artifact=binding(output/entry['filename'])))
    expected={e['filename'] for e in entries if e['status']=='completed'}
    actual={p.name for prefix in ('A-','B-') for p in output.glob(prefix+'*.json')}
    if expected!=actual:raise ValueError('extra/unconsumed cell artifact')
    return dict(schema=SCHEMA,complete=anchor and len(expected)==6,conditions=rows,endpoint=endpoints(rows),A_anchor_qualified=anchor,
        requested_denominators=dict(states=1,cues=2,image_variants=3,requested_cells=6,historical_anchors=1),
        completed_requests=len(expected),generated_actions=sum(len(r['record']['token_ids']) for r in rows if r['status']=='completed'),
        limitations='Supplied cues/native history; mixed context, unequal region doses; no physical recovery or endogenous owner claim.')


def resources(output, begin, native):
    usage=reused.readout.usage(output,native=native)
    return usage,[k for k,v in usage.items() if v>BOUNDS[k]]+(['wall_seconds'] if time.monotonic()-begin>BOUNDS['wall_seconds'] else [])


def run(config_path,output,*,cpu_factory=None):
    import torch
    from safetensors.torch import save_file
    output=Path(output).resolve()
    if not output.is_relative_to(OUTPUT):raise ValueError('invocation outside task owner')
    output.mkdir(parents=True,exist_ok=False);begin=time.monotonic();session=None;packet=None;entries=[];code=2;anchor=False
    native=cpu_factory is None
    counts=dict(checkpoint_loads=0,fixture_sessions=0,attempted_requests=0,completed_requests=0,generated_actions=0,
        training=0,optimizer=0,backward=0,replay=0,warmup=0,exports=0)
    a.write(output/'invocation.json',dict(config=binding(config_path),pid=os.getpid(),started=time.time(),
        compute='native' if native else 'CPU_FIXTURE',cpu_threads_incoming=torch.get_num_threads()))
    try:
        config,packet=load_packet(config_path,cpu=not native)
        if native and str(output)!=config['output']:raise ValueError('output differs from release')
        a.write(output/'qualification.json',dict(source_revision=config['source_revision'],producer=config['producer_files'],
            runtime=config['runtime'],bindings=packet['bindings'],helpers=packet['cached_pipeline_files'],payloads=packet['payloads']))
        if resources(output,begin,native)[1]:raise ValueError('resource excess before load')
        start=time.monotonic();counts['checkpoint_loads' if native else 'fixture_sessions']=1
        session=(cpu_factory or NativeSession)(reused.PREVIOUS/'checkpoint-16',packet)
        a.write(output/'load-B16.json',dict(seconds=time.monotonic()-start,composition=session.composition))
        for cell in packet['conditions']:
            name=cell['condition'];reason='A-anchor' if name!='A-clean' and not anchor else ''
            excess=resources(output,begin,native)[1]
            if excess:reason='resource_limit:'+','.join(excess)
            if reason:entries.append(dict(condition=name,status='skipped-HOLD',reason=reason));continue
            counts['attempted_requests']+=1
            record=session.generate(cell);record.update(schema=SCHEMA,condition=name,image_id=cell['image_id'])
            tensors=record.pop('_tensors');path=output/(name+'.safetensors');save_file(tensors,str(path))
            record.update(score_tensors=binding(path),tensor_identities={k:reused.visual.tensor_identity(v) for k,v in tensors.items()})
            a.write(output/(name+'.json'),record);counts['generated_actions']+=len(record['token_ids'])
            validate_record(record,cell,packet);counts['completed_requests']+=1
            if name=='A-clean':
                check=reused.fidelity(record,cell,packet);anchor=check['qualified'];a.write(output/'control-A-clean.json',check)
            entries.append(dict(condition=name,status='completed',filename=name+'.json'))
            a.write(output/f'resource-{counts["attempted_requests"]:02d}.json',dict(seconds=time.monotonic()-begin,counts=counts,usage=resources(output,begin,native)[0]))
        a.write(output/'conditions.json',entries);tokenizer=session.q.tokenizer;session.close();session=None
        report=consume(output,packet,tokenizer);report['compute']='native' if native else 'CPU_FIXTURE'
        a.write(output/'readback.json',report);load_packet(config_path,cpu=not native)
        if report['complete'] and not resources(output,begin,native)[1]:code=0
    except Exception as error:
        a.write(output/'error.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        if session is not None and getattr(session,'failure_evidence',None) is not None:
            a.write(output/'partial-generation.json',session.failure_evidence)
        if packet is not None and not (output/'conditions.json').exists():
            done={e['condition'] for e in entries}
            entries.extend(dict(condition=c['condition'],status='skipped-HOLD',reason='technical_failure:'+type(error).__name__) for c in packet['conditions'] if c['condition'] not in done)
            a.write(output/'conditions.json',entries)
    finally:
        if session is not None:
            try:session.close()
            except Exception as error:a.write(output/'cleanup-error.json',dict(error=str(error)));code=2
        usage,excess=resources(output,begin,native)
        a.write(output/'terminal.json',dict(exit_code=code,status='complete' if code==0 else 'technical_HOLD',
            counts=counts,seconds=time.monotonic()-begin,usage=usage,resource_excess=excess,
            readback=binding(output/'readback.json') if (output/'readback.json').exists() else None))
    return code


def main(argv=None,*,cpu_factory=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True)
    p=sub.add_parser('run');p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('readback');p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':print(json.dumps(prepare(args.output),sort_keys=True));return 0
    if args.command=='run':return run(args.config,args.output,cpu_factory=cpu_factory)
    import torch
    from probes import rollout_row_credit as retained
    print('cue_specificity_cpu_threads_incoming='+str(torch.get_num_threads()),file=sys.stderr)
    output=Path(args.output);inv=a.load(output/'invocation.json')
    if a.digest(inv['config']['path'])!=inv['config']['sha256']:raise ValueError('saved config changed')
    _,packet=load_packet(inv['config']['path'],cpu=inv['compute']=='CPU_FIXTURE')
    q=retained.frontend()
    if q.model is not None:raise ValueError('saved consumer loaded a model')
    print(json.dumps(consume(output,packet,q.tokenizer),sort_keys=True));return 0


if __name__=='__main__':
    raise SystemExit(main())
