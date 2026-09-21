import json,hashlib
from pathlib import Path
from transformers import AutoTokenizer
R=Path(__file__).resolve().parent;F=R.parent/'2026-09-17-readout-norm-fresh128'
read=lambda p:json.loads(p.read_text())
def bind(p):p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
p=read(F/'panel.json');z=read(F/'result.json');tok=AutoTokenizer.from_pretrained(p['config']['model']['base_model'],trust_remote_code=True)
def text(desc,b):return '<|object_ref_start|>'+desc+'<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{v}|>' for v in b)+'<|box_end|>'
cases=[]
for iid,row,same,distinct,desc,ddesc,union,physical in [
 (309264,5,[0,781,96,903],[637,552,701,624],'bird','bird',['367404'],['physical:left_lower_cage_bird','known:367404']),
 (386313,12,[554,629,566,692],[684,501,750,597],'book','clock',['1660099','340554'],['known:1660099','known:340554'])]:
 x=z['images'][str(iid)];g=next(g for g in p['groups'] if g['key']==x['group']);g=json.loads(json.dumps(g))
 for c in g['cases']:c['image_id']=c['input_record']['image_id']
 j=x['batch_index'];raw=Path(x['raw_bindings']['O']['path']);saved=read(raw)['rows'][j];original=saved['token_ids'];starts=[k for k,t in enumerate(original) if t==tok.convert_tokens_to_ids('<|object_ref_start|>')];s=starts[row];e=next(k+1 for k in range(s,len(original)) if original[k]==tok.convert_tokens_to_ids('<|box_end|>'))
 arms={}
 for name,d,b in [('native',desc,x['baseline']['complete_rows'][row]['box']),('same',desc,same),('distinct',ddesc,distinct)]:
  literal=text(d,b);ids=tok.encode(literal,add_special_tokens=False);assert len(ids)==e-s==9
  arms[name]=dict(token_ids=ids,text=literal,description=d,box=b,changed_offsets=[s+k for k,(a,b) in enumerate(zip(original[s:e],ids)) if a!=b])
 assert arms['native']['token_ids']==original[s:e]
 case=dict(image_id=iid,group=g,target_position=j,saved_raw=bind(raw),saved_receipt=bind(raw.parent/'receipt.json'),start_offset=s,end_offset=e,prefix_sha256=hashlib.sha256(json.dumps(original[:s],separators=(',',':')).encode()).hexdigest(),arms=arms,capture_offsets=sorted(set([s+4,s+5,s+6,s+7,e,e+4,e+13,e+40])),supplied_known_owner_union=union,supplied_physical_owner_union=physical,remaining_free_budget=3084-e,conditional_control=None)
 case['candidate_provenance']=dict(same='visually checked same visible lower-left bird; manual small extent adjustment, evaluation-only' if iid==309264 else 'existing normalized row14 (zero-based), visible teal/green book spine; native bbox overextends left',distinct='existing normalized row4 (zero-based), frozen known bird367404' if iid==309264 else 'existing original future row30 (zero-based), frozen known clock340554',confounds='same-owner edit small versus large spatial displacement to distinct bird; no comparable covered owner near distinct available' if iid==309264 else 'distinct clock changes category as well as position/extent; same-owner alternative is geometry-only; owner novelty cannot be isolated')
 cases.append(case)
sources=[bind(F/'panel.json'),bind(F/'result.json'),bind(R/'admission/result.json')]
for b in p['sources']:
 if any(t in b['path'] for t in ['/src/','/probes/','/adapter/','/special_token_embeddings/']):sources.append(b)
for c in cases:
 sources += [c['saved_raw'],c['saved_receipt'],bind(Path(c['group']['input_jsonl']))]
 for item in c['group']['cases']:sources.append(bind(Path(item['image_path'])))
sources=list({b['path']:b for b in sources}.values())
for b in sources:assert bind(b['path'])==b,b['path']
result=dict(schema='owner_recurrence.row_branch.v1',config=p['config'],sources=sources,cases=cases,banks={str(c['image_id']):p['banks'][str(c['image_id'])] for c in cases},selected_hold=dict(image_id=253212,reason='selected early orange/book extent cannot confidently separate neighboring spines; no execution or substitute'),bounds=dict(scientific_cases=2,held_cases=1,batch_executions=6,model_forwards=18504,allocated_gpu_seconds=7200,elapsed_seconds=7200),policy='original native greedy; no coordinate norm transform',physical_sidecar='root-admission.json',optional_control='not admitted; none preselected with comparable covered-owner displacement')
(R/'panel.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({str(c['image_id']):dict(interval=[c['start_offset'],c['end_offset']],remaining=c['remaining_free_budget'],arms=c['arms']) for c in cases},indent=2))
