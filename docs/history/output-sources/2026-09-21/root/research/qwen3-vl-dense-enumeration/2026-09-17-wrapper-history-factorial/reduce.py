"""Saved-token positive-only factorial ledgers; immutable predecessor scorer."""
import importlib.util,json,hashlib,sys
from pathlib import Path
from transformers import AutoTokenizer
R=Path(__file__).resolve().parent;P=R.parent/'2026-09-17-history-rereading-mechanism'
read=lambda p:json.loads(p.read_text())
bind=lambda p:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
scorer=R.parent/'2026-09-16-endpoint-loop-natural-readout-norm/reduce.py'
spec=importlib.util.spec_from_file_location('accepted_score',scorer);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
prior=read(P/'human-evaluation/evaluation-v1/result.json');banks=prior['banks']
panel=read(R/'panels/F.json');case=panel['cases'][0]['group']['cases'][1]
tok=AutoTokenizer.from_pretrained(panel['config']['model']['base_model'],trust_remote_code=True)
entries={n:P/f'runtime/{v}/raw.json' for n,v in [('F','native-F'),('R','residual-only'),('H','cache-only'),('RH','joint')]}
entries.update({n:R/f'runtime/{n}/raw.json' for n in ['W','RW','HW','RHW']})
def score(ids,stop,bank):
 v=mod.score(dict(token_ids=ids,text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),stop=stop),case,bank)
 return {k:v[k] for k in ['matches','burden','token_count','stop','complete_rows','first_literal_repeat','longest_exact_run']}
cells={}
for name,path in entries.items():
 raw=read(path)['rows'][1];ids=raw['token_ids'];assert raw['image_id']==309264
 assert ids[:63]==panel['cases'][0]['target_prefix_token_ids']
 assert ids[63:67]==[151646,22592,151647,151648] and ids[71]==151649
 views={'full':(ids,raw['stop']),'prefix':(ids[:54],'prefix'),'supplied':(ids[54:72],'supplied'),'free':(ids[72:],raw['stop'])}
 cells[name]=dict(raw=bind(path),versions={v:{k:score(t,s,b) for k,(t,s) in views.items()} for v,b in banks.items()})
union=set().union(*(set(c['versions']['current']['supplied']['matches']['covered_owner_ids']) for c in cells.values()))
base=set(cells['F']['versions']['current']['free']['matches']['covered_owner_ids'])-union
summary=[]
for name,c in cells.items():
 v=c['versions']['current'];free=set(v['free']['matches']['covered_owner_ids']);prefix=set(v['prefix']['matches']['covered_owner_ids']);primary=free-union
 c['primary']=dict(excluded_supplied_union=sorted(union),free=sorted(primary),new_relative_prefix=sorted(primary-prefix),retained=sorted(primary&base),gained=sorted(primary-base),lost=sorted(base-primary),retention_denominator=len(base),denominator=14-len(union),FN=14-len(union)-len(primary))
 summary.append(dict(condition=name,current_full=v['full']['matches']['covered_owner_ids'],current_free=sorted(free),primary=c['primary'],burden=v['free']['burden'],tokens=v['full']['token_count'],stop=v['full']['stop']))
for n,old in [('F','native-F'),('R','residual-only'),('H','cache-only'),('RH','joint')]:
 for version in banks:
  for view in ['full','prefix','supplied','free']:
   assert cells[n]['versions'][version][view]['matches']==prior['cells']['new/'+old]['versions'][version][view]['matches']
o=banks['current'][0];txt='<|object_ref_start|>'+o['description']+'<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{x}|>' for x in o['reference_coord_bins_1000'])+'<|box_end|>'
fixture=score(tok.encode(txt,add_special_tokens=False),'supplied',banks['current']);covered=set(fixture['matches']['covered_owner_ids']);assert o['owner_id'] in covered and len(covered-covered)==0
result=dict(status='candidate',banks=banks,annotation_version='annotation-snapshot-v1',bindings=[bind(scorer),bind(P/'human-evaluation/evaluation-v1/result.json'),bind(R/'panels/F.json'),bind(Path(__file__))],cells=cells,summary=summary,verification=dict(predecessor_ledgers_exact=True,credit_exclusion_fixture=dict(owner_id=o['owner_id'],before=1,after=0)),limits=['No explicit ignore fields; unmatched is UNKNOWN','Current positives are not exhaustive negative labels','Supplied union exclusion across all eight factorial cells'])
output=Path(sys.argv[1]) if len(sys.argv)>1 else R/'reduction.json';output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary))
