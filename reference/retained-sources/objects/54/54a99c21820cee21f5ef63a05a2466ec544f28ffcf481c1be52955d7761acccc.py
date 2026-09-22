"""Lead saved-output verification of fixed-history image substitution."""
import hashlib
import importlib.util
import json
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer

R=Path(__file__).resolve().parent
OLD=R.parent/'2026-09-16-corner-loop-bridge-factorial'
U=Path('/data/CoordExp/.worktrees/research-probes/research/experiments')/R.name
read=lambda p:json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
p,res,term,q=(read(R/n) for n in ['panel.json','result.json','terminal.json','qualification.json'])
for b in p['sources']+term['bindings']+term['raw_outputs']+list(term['runtime_receipts'].values())+q['sources']:
    path=Path(b['path'])
    if path.parent==U:path=R/'candidate-records'/path.name
    assert sha(path)==b['sha256'],str(path)
spec=importlib.util.spec_from_file_location('frozen_image_consumer',R/'reduce.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
tok=AutoTokenizer.from_pretrained(p['config']['model']['base_model'],local_files_only=True)
base=read(OLD/'runtime/C00/receipt.json');orig=read(OLD/'runtime/C00/raw.json')
recs=[];live=[]
for name in ['I00','D00','D10']:
    rec=read(R/f'runtime/{name}/receipt.json');raw=read(R/f'runtime/{name}/raw.json');recs.append(rec)
    assert rec['status']=='candidate_complete' and not rec['mechanical_forks']
    assert rec['producer']['sha256']==sha(R/'producer.py')
    for key in ['loaded_identity','generate_settings','prompt_positions_sha256','positions']:
        assert rec[key]==base[key],(name,key)
    assert {k:v for k,v in rec['batch_shape'].items() if k!='image_ids'}=={k:v for k,v in base['batch_shape'].items() if k!='image_ids'}
    for j in range(4):
        if name=='I00' or j!=1:assert raw['rows'][j]==orig['rows'][j]
    assert raw['rows'][1]['image_id']==(477415 if name=='I00' else 7116)
    ids=raw['rows'][1]['token_ids']
    assert ids[:1233]==p['cells'][name]['history_ids']
    assert raw['free_token_ids']==ids[1233:] and len(ids)==3084
    assert rec['history_mask']=={'batch_position':1,'start':0,'end_exclusive':1233,'free_logits_unrestricted':True}
    assert rec['counts']=={'model_forwards':3084,'vision_forwards':1,'processor_calls':3084}
    proc=Path('/proc')/str(rec['pid'])/'cmdline'
    if proc.exists() and str(R/'producer.py').encode() in proc.read_bytes().split(b'\0'):live.append(rec['pid'])
summary={};raws={}
for name,path,iid in [('O00',OLD/'runtime/C00/raw.json',477415),('O10',OLD/'runtime/C10/raw.json',477415),('D00',R/'runtime/D00/raw.json',7116),('D10',R/'runtime/D10/raw.json',7116)]:
    raw=read(path);raws[name]=raw
    assert raw['free_token_ids']==raw['rows'][1]['token_ids'][1233:]
    got=c.score(raw['free_token_ids'],raw['rows'][1]['stop'],iid,p,tok)
    assert got=={k:v for k,v in res['cells'][name].items() if k!='raw'}
    b=got['burden'];assert b['complete_rows']==205 and got['free_token_count']==1851 and got['stop']=='length'
    summary[name]={'matched':got['primary_free_only']['matched_count'],'owners':got['primary_free_only']['covered_owner_ids'],'burden':b,'endpoint_occupancy':got['endpoint_occupancy']}
for name,ref in [('D00','O00'),('D10','O10')]:
    a,b=raws[name]['free_token_ids'],raws[ref]['free_token_ids']
    first=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
    expected=res['paired_comparisons'][name]
    assert first==expected['first_changed_free_token_zero_based'] and a[first]==expected['donor_token'] and b[first]==expected['original_token']
assert [summary[x]['matched'] for x in ['O00','O10','D00','D10']]==[0,16,0,0]
baseline=c.score(p['donor']['generation']['generated_token_ids'],p['donor']['generation']['decode_stop_reason'],7116,p,tok)
assert baseline==res['baseline_donor_saved_output'] and baseline['primary_free_only']['matched_count']==4
assert baseline['stop']=='im_end' and baseline['burden']['strict_valid_repeats']==0 and not baseline['drops']
# Requalify the donor pool from the original saved outputs, not its audit booleans.
datafile=Path(q['sources'][-2]['path']);data={x['image_id']:(i,x) for i,x in enumerate(json.loads(line) for line in datafile.read_text().splitlines())}
bank={x['image_id']:x['owners'] for x in read(Path(q['sources'][-1]['path']))['bank']['records']}
eligible=[];seen=[]
for b in q['sources'][:-2]:
    for g in read(Path(b['path']))['generation']['rows']:
        iid=g['image_id'];seen.append(iid);ri,row=data[iid]
        if iid in {x['image_id'] for x in p['batch_group']}:continue
        if g['observed_image_grid_thw']!=[1,52,78] or g['prompt_token_ids']!=p['batch_group'][1]['prompt_token_ids'] or g['decode_stop_reason']!='im_end':continue
        with Image.open((datafile.parent/row['images'][0]).resolve()) as im:
            if im.size!=(1248,832):continue
        native=c.parse_compact_object_box_closed(g['raw_decode_text'],image_width=row['width'],image_height=row['height'],row_id=g['example_id'],row_index=ri).to_artifact_dict()
        valid,drops=c.match._matchable_rows_with_geometry_debt({**native,'pred':native['predictions']})
        for j,x in enumerate(valid):x['prediction_id']=f'baseline-{j}'
        targets=[c.metrics._target(image_id=iid,owner_id=o['owner_id'],description=o['description'],coord_bins=o['coord_bins']) for o in bank[iid]]
        if not drops and not c.metrics._strict_repeat_rows(valid) and c.match._ledger_image(targets,valid,threshold=.5)['matched_count']>=2:eligible.append(iid)
assert len(seen)==len(set(seen))==256
assert sorted(eligible)==q['eligible_ids'] and len(eligible)==58 and min(eligible)==p['donor']['image_id']==7116
synthetic=[c.metrics._target(image_id=477415,owner_id='credit-fixture',description='person',coord_bins=[0,0,999,999])]
text='<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_0|><|coord_0|><|coord_999|><|coord_999|><|box_end|>'
native=c.parse_compact_object_box_closed(text,**p['parse_context']).to_artifact_dict();v,_=c.match._matchable_rows_with_geometry_debt({**native,'pred':native['predictions']})
for x in v:x['prediction_id']='SUPPLIED'
assert c.match._ledger_image(synthetic,v,threshold=.5)['matched_count']==1 and c.match._ledger_image(synthetic,[],threshold=.5)['matched_count']==0
assert sum(x['counts']['model_forwards'] for x in recs)==9252 and not live
exits={x.name:int(x.read_text()) for x in R.glob('*.exit')};assert exits and set(exits.values())=={0}
out={'status':'lead-accepted','scope':'One donor x two supplied histories; visual sensitivity without donor known-owner recovery','result':bind(R/'result.json'),'panel':bind(R/'panel.json'),'terminal':bind(R/'terminal.json'),'verifier':bind(Path(__file__)),'all_free_only_reductions_equal':True,'identity_all_four_sequences_exact':True,'companions_exact_all_cells':True,'requalified_donor_candidates':256,'requalified_eligible':58,'selected_donor':7116,'credit_exclusion_falsification':[1,0],'summary':summary,'paired_comparisons':res['paired_comparisons'],'exit_codes':exits,'live_owned_jobs':live,'limits':['Long supplied history is off-policy on donor.','One donor does not establish image independence or initial-entry causality.','UNKNOWN is not confirmed false positive; valid-box repetition is not repair.']}
(R/'lead-acceptance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'matched':{k:v['matched'] for k,v in summary.items()},'donor_pool':len(eligible),'first_changes':{k:v['first_changed_free_token_zero_based'] for k,v in res['paired_comparisons'].items()},'live_owned_jobs':live}))
