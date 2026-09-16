"""Frozen observed-output pairs and deterministic replay for the paired16 repair."""
from __future__ import annotations
import copy
import json
import random
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
from probes.training_set_completion import training, source256_training as predecessor

BASE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
OLD = BASE/'2026-09-16-source256-fixed-prefix-completion'
NORMALIZED = BASE/'2026-09-16-source256-completion-ce-normalization'
ROOT = BASE/'2026-09-16-source256-output-ranking-repair'
PREPARATION = OLD/'preparation/source256-admitted-v1/preparation.json'
DIAGNOSIS = NORMALIZED/'lead/credit-diagnosis-v1/diagnosis.json'
ANCHOR_TERMINAL = NORMALIZED/'runtime/main-normalized-v1/B-normalized/training/terminal.json'
ANCHOR_MANIFEST = NORMALIZED/'runtime/main-normalized-v1/B-normalized/training-manifest.json'
SCHEMA = 'source256.output_ranking_repair.v1'
EOS = 151645

def read(path):
    return json.loads(Path(path).read_text())

def checked(binding):
    assert training.binding(binding['path']) == binding, binding['path']
    return read(binding['path'])

def load_data(manifest):
    data = checked(manifest['data'])
    assert data['schema'] == SCHEMA+'.data'
    assert len(data['canonical_routes']) == 256 and len(data['pairs']) == 15
    for key, pair in data['pairs'].items():
        a,b=pair['preferred'],pair['rejected']
        assert a['prompt_token_ids']==b['prompt_token_ids']
        assert a['image_identity']==b['image_identity']
        assert pair['denominator']==max(len(a['continuation_token_ids']),len(b['continuation_token_ids']))
        assert len(a['continuation_token_ids'])==pair['preferred_tokens']
        assert len(b['continuation_token_ids'])==pair['rejected_tokens']
        assert a['continuation_token_ids'][-1]==EOS
        for route in (a,b):
            assert route['ce_weights']==[1]*len(route['continuation_token_ids'])
            assert EOS not in route['continuation_token_ids'][:-1]
        assert not b['trusted_boxes']
        if int(key)==548337:
            assert (pair['preferred_tokens'],pair['rejected_tokens'])==(177,3084)
            assert b['continuation_token_ids'][-1]!=EOS
        else:
            assert b['continuation_token_ids'][-1]==EOS
    return data

def prepare(output_root=ROOT):
    root=Path(output_root);out=root/'preparation';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'prepared.json').exists(), 'immutable preparation already exists'
    raw=read(PREPARATION)
    assert training.file_hash(PREPARATION)=='9b94baeb0699e479413483cba5ee6fb4b4d98c062aebb0ba462ae61a871cf242'
    diag=read(DIAGNOSIS);images=sorted(diag['strict_route_preference_images'])
    assert len(images)==15 and len(set(images))==15
    records=predecessor.hydrate_bound_cases({'preparation':raw,'records':raw['routes']})
    canonical={str(r['image_id']):r['canonical_route'] for r in records}
    anchor=read(ANCHOR_TERMINAL);old_manifest=read(ANCHOR_MANIFEST)
    assert anchor['status']=='completed' and anchor['updates']==64
    assert anchor['manifest']==training.binding(ANCHOR_MANIFEST)
    checkpoint=next(c for c in anchor['checkpoints'] if c['step']==64)
    for f in checkpoint['adapter']['files']:
        assert training.file_hash(Path(checkpoint['adapter']['root'])/f['relative_path'])==f['sha256']
    tokenizer=Tokenizer.from_file(raw['sources']['tokenizer']['path'])
    assert training.binding(raw['sources']['tokenizer']['path'])==raw['sources']['tokenizer']
    boxstart=tokenizer.token_to_id('<|box_start|>');boxend=tokenizer.token_to_id('<|box_end|>')
    coords={token:i for i,token in enumerate(old_manifest['validity_hinge']['coordinate_token_ids'])}
    def route(saved, kind, base):
        ids=list(saved['generated_token_ids'])
        assert saved['generated_token_ids_sha256']==training.digest(ids)
        assert saved['prompt_token_ids']==base['prompt_token_ids']
        assert saved['executed_media_sha256']==base['image_identity']['executed_media_sha256']
        assert tokenizer.decode(ids,skip_special_tokens=False)==saved['raw_decode_text']
        boxes=[]
        if kind=='preferred':
            for i,tok in enumerate(ids):
                if tok!=boxstart: continue
                assert ids[i+5]==boxend and all(t in coords for t in ids[i+1:i+5])
                bins=[coords[t] for t in ids[i+1:i+5]]
                assert bins[0]<bins[2] and bins[1]<bins[3]
                boxes.append(dict(zip(('x1_position','y1_position','x2_position','y2_position'),range(i+1,i+5)),expected_bins=bins))
        return dict(route_id=base['example_id']+':'+kind,image_id=base['image_id'],example_id=base['example_id'],case=base['case'],image_identity=base['image_identity'],prompt_token_ids=base['prompt_token_ids'],continuation_token_ids=ids,ce_weights=[1]*len(ids),labels=ids,trusted_boxes=boxes,trusted_complete_support_endpoint=False,provenance=dict(kind=kind,weak_full_output=True,unknown_is_not_gt=True,observed_stop=saved['decode_stop_reason']))
    pairs={};sources={};unknown=0;advantage=0;drops=0
    for im in images:
        ds=diag['images'][str(im)];assert ds['route_comparison']['strict_known_owner_debt_dominance']
        routes={}
        for label,kind in [('A64','preferred'),('Bnormalized64','rejected')]:
            file,fragment=ds[label]['pointer'].split('#');shard=read(file)
            expected=next(b for b in diag['bindings'] if b['path']==file)
            assert training.file_hash(file)==expected['sha256']
            sources[file]=training.binding(file)
            saved=shard['generation']['rows'][int(fragment.rsplit('/',1)[-1])]
            assert saved['image_id']==im
            routes[kind]=route(saved,kind,canonical[str(im)])
        a,b=routes['preferred'],routes['rejected'];la,lb=len(a['continuation_token_ids']),len(b['continuation_token_ids'])
        pairs[str(im)]={**routes,'denominator':max(la,lb),'preferred_tokens':la,'rejected_tokens':lb}
        unknown+=ds['A64']['burden']['annotation_unmatched_prediction_count']
        advantage+=len(ds['route_comparison']['A_only']);drops+=len(ds['Bnormalized64']['dropped'])
    assert unknown==80 and advantage==26 and drops==293
    assert sum(p['rejected_tokens'] for k,p in pairs.items() if k!='548337')==2162
    rng=random.Random(19);ids=list(map(int,canonical));common=[]
    for _ in range(2):
        shuffled=ids.copy();rng.shuffle(shuffled);common.extend(shuffled)
    order=images.copy();random.Random(19).shuffle(order);pairseq=(order*35)[:512]
    schedule=[dict(step=i+1,common_image_ids=common[i*32:(i+1)*32],pair_image_ids=pairseq[i*32:(i+1)*32]) for i in range(16)]
    assert set(Counter(common).values())=={2};assert set(Counter(pairseq).values())=={34,35}
    data=dict(schema=SCHEMA+'.data',canonical_routes=canonical,pairs=pairs,schedule=schedule,sources=dict(preparation=training.binding(PREPARATION),diagnosis=training.binding(DIAGNOSIS),anchor_terminal=training.binding(ANCHOR_TERMINAL),anchor_manifest=training.binding(ANCHOR_MANIFEST),readbacks=list(sources.values())),counts=dict(canonical=512,pair=512,pair_counts=dict(Counter(pairseq)),preferred_unknown_rows=unknown,known_match_advantage=advantage,rejected_parser_drops=drops))
    training.publish(out/'prepared.json',data)
    for arm,mode,updates in [('P','main',16),('R','main',16),('R','qualification',1)]:
        config=copy.deepcopy(old_manifest['model_config']);config['adapter']['path']=checkpoint['adapter']['root'];config['run']['artifact_root']=str(root/'runtime'/mode/arm)
        runtime=dict(seed=19,updates=updates,world_size=4,microbatch_size=2,effective_image_batch=64,branch_image_count=32,gradient_clip_norm=1.,eos_token_id=EOS,activation_checkpointing=True,checkpoint_steps=[updates],fresh_optimizer=True,wall_seconds=14400,max_model_calls=updates*(32 if arm=='P' else 48),max_model_forwards=updates*(64 if arm=='P' else 96))
        manifest=dict(schema=SCHEMA+'.manifest',arm=arm,mode=mode,data=training.binding(out/'prepared.json'),preparation=training.binding(PREPARATION),source_adapter=checkpoint['adapter'],model_config=config,optimizer=old_manifest['optimizer'],scheduler=dict(type='cosine',total_updates=16,min_lr_ratio=0.,warmup_updates=0),validity_hinge=old_manifest['validity_hinge'],runtime=runtime,reference_cache_path=str(out/'reference-scores.json'),ranking=dict(lambda_value=1.,branch_weight=.5,denominator='max_recorded_action_lengths',reference='frozen_starting_Bnormalized64',rejected_geometry=False),sources=dict(data_producer=training.binding(Path(__file__)),paired_embeddings=raw['sources']['embedding_weights']))
        training.publish(out/f'{arm}-{mode}.json',manifest);load_data(manifest)
    return data['counts']

if __name__=='__main__':
    print(json.dumps(prepare(),sort_keys=True))
