"""Saved-output accounting; physical identity remains a separate bounded sidecar."""
import argparse
import importlib.util
import json
from pathlib import Path
from transformers import AutoTokenizer
from probes.training_set_completion.readout_norm_fresh import _binding, _write

SCORER=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-natural-readout-norm/reduce.py')
spec=importlib.util.spec_from_file_location('accepted_pure_score',SCORER)
scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)


def consume(root):
    master=json.loads((root/'stage-a-panel.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(master['config']['model']['base_model'],trust_remote_code=True)
    bank_panel=json.loads((root.parent/'2026-09-17-readout-norm-fresh128/panel.json').read_text())
    manifest=json.loads((root/'runtime-manifest.json').read_text())
    cells={};cost=[]
    for entry in manifest['cells']:
        folder=Path(entry['reuse_from']) if 'reuse_from' in entry else (Path(entry['output_root']) if entry['stage']=='C' else Path(entry['output_root'])/str(entry['image_id'])/entry['mode'])
        path=folder/'receipt.json'
        if not path.exists():continue
        receipt=json.loads(path.read_text())
        if receipt['status']!='candidate_complete':continue
        assert _binding(Path(entry['panel']['path']))==entry['panel']
        assert receipt['panel']==entry.get('reuse_receipt_panel',entry['panel'])
        assert receipt['raw']==_binding(folder/'raw.json')
        panel=json.loads(Path(entry['panel']['path']).read_text());case=panel['cases'][0]
        raw=json.loads((folder/'raw.json').read_text());row=raw['rows'][case['target_position']]
        ids=row['token_ids'];prefix=case['target_prefix_token_ids'];assert ids[:len(prefix)]==prefix
        start=case.get('norm_start',len(prefix)) if entry['mode']!='prefix' else len(prefix)
        pulse=raw.get('sampling_pulse',raw.get('norm_pulse'))
        last=pulse['last_active_offset'];release=(last+1) if last is not None else len(prefix)
        image=str(entry['image_id']);input_case=case['group']['cases'][case['target_position']]
        def view(tokens,stop):
            text=tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            z=scorer.score(dict(token_ids=tokens,text=text,stop=stop),input_case,bank_panel['banks'][image])
            z['semantic_malformed']=0 if tokens in ([],[151645]) else z['burden']['malformed']
            return z
        saved=json.loads(Path(case['saved_raw']['path']).read_text())['rows'][case['target_position']]
        key=f"{entry['stage']}/{image}/{entry['condition']}"
        cell=dict(entry=entry,receipt=_binding(path),raw=_binding(folder/'raw.json'),start=start,release=release,
            full=view(ids,row['stop']),prefix=view(ids[:start],'prefix'),
            intervention=view(ids[start:release],'intervention'),
            after_release=view(ids[release:],row['stop']),
            continuation_from_intervention_start=view(ids[start:],row['stop']),
            native_future=view(saved['token_ids'][start:],saved['stop']),
            first_changed_token=next((j for j,(a,b) in enumerate(zip(ids,saved['token_ids'])) if a!=b),None),
            winner_changes=sum(x['winner_changed'] for x in pulse['steps']),
            post_release_token_ids=ids[release:])
        supplied_rows=len(view(ids[:release],'history')['complete_rows'])
        seen=set();literal=[]
        for row_index,parsed in enumerate(cell['full']['complete_rows']):
            fingerprint=(parsed['description'],tuple(parsed['box']))
            if row_index>=supplied_rows and fingerprint in seen:
                literal.append(dict(full_row=row_index+1,**parsed))
            seen.add(fingerprint)
        strict=scorer.metrics._strict_repeat_rows(cell['full']['valid_predictions'])
        cell['post_release_recurrence_against_all_history']=dict(
            literal_rows=literal,
            strict_valid_rows=[x for x in strict if x['generated_order']>=supplied_rows],
            interpretation='geometric/literal recurrence, not automatic physical duplicate identity')
        # If the bounded pulse ends mid-row, its partially supplied owner is not free credit.
        opens=[j for j,t in enumerate(ids[:release]) if t==151646]
        cell['crossing_supplied_owner_ids']=[]
        if opens and opens[-1]>=start and 151649 not in ids[opens[-1]:release]:
            end=next((j+1 for j in range(release,len(ids)) if ids[j]==151649),release)
            crossing=view(ids[opens[-1]:end],'crossing_pulse_row')
            cell['crossing_supplied_owner_ids']=crossing['matches']['covered_owner_ids']
        cells[key]=cell
        if 'reuse_from' not in entry:cost.append(receipt)
    # Union of pulse/supplied known identities is symmetric within each stage/image.
    unions={}
    for key,c in cells.items():
        stage,image,condition=key.split('/');group=f'{stage}/{image}'
        union=unions.setdefault(group,set())
        if stage=='A': union.update(['367404'])
        elif stage=='C':
            union.update(c['intervention']['matches']['covered_owner_ids'])
            union.update(c['crossing_supplied_owner_ids'])
        elif condition in ('pre_entry_one','established_one','established_four'):
            union.update(c['intervention']['matches']['covered_owner_ids'])
    for key,c in cells.items():
        group='/'.join(key.split('/')[:2]);excluded=unions[group]
        free=set(c['after_release']['matches']['covered_owner_ids'])-excluded
        native=set(c['native_future']['matches']['covered_owner_ids'])-excluded
        prefix=set(c['prefix']['matches']['covered_owner_ids'])
        c['known_accounting']=dict(excluded_union=sorted(excluded),free=sorted(free),
            new_relative_prefix=sorted(free-prefix),gained_vs_native_future=sorted(free-native),
            lost_vs_native_future=sorted(native-free),retained_native_future=sorted(native&free))
        individual=set(c['intervention']['matches']['covered_owner_ids'])|set(c['crossing_supplied_owner_ids'])
        own_free=set(c['after_release']['matches']['covered_owner_ids'])-individual
        own_native=set(c['native_future']['matches']['covered_owner_ids'])-individual
        c['per_trajectory_known_accounting']=dict(supplied_owner_ids=sorted(individual),
            post_release_owner_ids=c['after_release']['matches']['covered_owner_ids'],
            autonomous_new_relative_prefix=sorted(own_free-prefix),
            gained_vs_native_future=sorted(own_free-own_native),
            lost_vs_native_future=sorted(own_native-own_free),
            retained_native_future=sorted(own_free&own_native))
        c['interpretation']='known matching only; no physical disappearance or FP inference'
        if c['entry']['mode']=='sustained':
            c['interpretation']+='; no withdrawal, after_release is not a recovery estimand'
            active=set(c['continuation_from_intervention_start']['matches']['covered_owner_ids'])
            native=set(c['native_future']['matches']['covered_owner_ids'])
            c['known_accounting']=dict(post_withdrawal='not_applicable',
                active_policy_gained=sorted(active-native),active_policy_lost=sorted(native-active),
                active_policy_retained=sorted(native&active))
    return dict(status='candidate',consumer=_binding(Path(__file__)),scorer=_binding(SCORER),cells=cells,
        cost=dict(batch_executions=len(cost),model_forwards=sum(x['model_forwards'] for x in cost),
                  measured_runtime_seconds=sum(x['elapsed_seconds'] for x in cost)),
        physical_identity='separate sidecar; pulse owner union may include credible unlabeled IDs beyond known ledger')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();result=consume(a.root);_write(a.output,result)
    print(json.dumps(dict(cells=len(result['cells']),cost=result['cost'])) )
