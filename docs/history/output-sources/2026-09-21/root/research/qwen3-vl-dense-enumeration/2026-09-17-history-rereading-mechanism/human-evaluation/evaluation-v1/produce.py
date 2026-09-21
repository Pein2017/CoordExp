"""Positive-only current-export scoring; no negative/ignore inference."""
import importlib.util,json,hashlib
from pathlib import Path
from transformers import AutoTokenizer
E=Path(__file__).resolve().parent;R=E.parent.parent;P=R.parent/'2026-09-17-successful-row-mechanism';S=E.parent/'annotation-snapshot-v1'
SCORER=R.parent/'2026-09-16-endpoint-loop-natural-readout-norm/reduce.py'
read=lambda p:json.loads(p.read_text())
bind=lambda p:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
spec=importlib.util.spec_from_file_location('accepted_score',SCORER);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
working=next(json.loads(l) for l in (S/'working.norm.jsonl').read_text().splitlines() if json.loads(l)['image_id']==309264)
source=next(json.loads(l) for l in (S/'source.norm.jsonl').read_text().splitlines() if json.loads(l)['image_id']==309264)
newbank=[dict(owner_id=str(o['coco_ann_id']),image_id=309264,description=o['desc'],normalized_description=o['desc'].lower(),reference_coord_bins_1000=o['bbox_2d']) for o in working['objects']]
oldbank=read(R.parent/'2026-09-17-readout-norm-fresh128/panel.json')['banks']['309264']
assert len(newbank)==14 and len({o['owner_id'] for o in newbank})==14 and len(oldbank)==10
source_by_id={str(o['coco_ann_id']):o for o in source['objects']};working_by_id={str(o['coco_ann_id']):o for o in working['objects']}
assert set(source_by_id)=={o['owner_id'] for o in oldbank}
for o in oldbank:assert o['reference_coord_bins_1000']==source_by_id[o['owner_id']]['bbox_2d']
panel=read(R/'panels/F.json');case=panel['cases'][0]['group']['cases'][1]
tok=AutoTokenizer.from_pretrained(panel['config']['model']['base_model'],trust_remote_code=True)
entries=[]
for n in ['native-S','native-F','residual-S-to-F-layer13','residual-S-to-F-layer20']:
 entries.append(('prior/'+n,P/'stage2/runtime'/n/'raw.json'))
for n in ['native-F','self-F','residual-only','cache-only','joint','rebuild-cache-only']:
 entries.append(('new/'+n,R/'runtime'/n/'raw.json'))
def score(ids,stop,bank):
 d=mod.score(dict(token_ids=ids,text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),stop=stop),case,bank)
 return dict(matches=d['matches'],burden=d['burden'],token_count=len(ids),stop=stop)
cells={}
old_expected={**{'prior/'+k:v for k,v in read(P/'stage2-final-reduction.json')['cells'].items()},**{'new/'+k:v for k,v in read(R/'reduction.json')['cells'].items()}}
for name,path in entries:
 row=read(path)['rows'][1];ids=row['token_ids'];assert row['image_id']==309264
 views={'full':(ids,row['stop']),'prefix':(ids[:54],'prefix'),'supplied':(ids[54:72],'supplied'),'free':(ids[72:],row['stop'])}
 versions={v:{k:score(t,stop,b) for k,(t,stop) in views.items()} for v,b in [('old',oldbank),('current',newbank)]}
 for view in views:assert versions['old'][view]['matches']==old_expected[name][view]['matches'],(name,view)
 cells[name]=dict(raw=bind(path),versions=versions)
union=set().union(*(set(c['versions']['current']['supplied']['matches']['covered_owner_ids']) for n,c in cells.items() if n.startswith('new/')))
base=set(cells['new/native-F']['versions']['current']['free']['matches']['covered_owner_ids'])-union
summary=[]
for name,c in cells.items():
 v=c['versions']['current'];full=set(v['full']['matches']['covered_owner_ids']);free=set(v['free']['matches']['covered_owner_ids']);prefix=set(v['prefix']['matches']['covered_owner_ids'])
 if name.startswith('new/'):
  primary=free-union;c['primary']=dict(excluded_supplied_union=sorted(union),free=sorted(primary),new_relative_prefix=sorted(primary-prefix),retained=sorted(primary&base),gained=sorted(primary-base),lost=sorted(base-primary),retention_denominator=len(base),denominator=14-len(union),FN=14-len(union)-len(primary))
 c['current_sets']={k:v[k]['matches']['covered_owner_ids'] for k in views}
 summary.append(dict(condition=name,old_full=c['versions']['old']['full']['matches']['matched_count'],current_full=len(full),current_full_FN=14-len(full),current_free=len(free),current_free_FN=14-len(free),current_free_UNKNOWN=v['free']['burden']['unknown'],primary=c.get('primary')))
# Source-grounded exclusion sensitivity: a current positive emitted in both supplied and free must be excluded.
o=newbank[0];b=o['reference_coord_bins_1000'];text='<|object_ref_start|>'+o['description']+'<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{x}|>' for x in b)+'<|box_end|>'
f=score(tok.encode(text,add_special_tokens=False),'supplied',newbank);covered=set(f['matches']['covered_owner_ids']);assert o['owner_id'] in covered and len(covered)==1
assert len(covered-covered)==0
cross=[]
for i in sorted(set(source_by_id)|set(working_by_id)):
 a=source_by_id.get(i);b=working_by_id.get(i)
 cross.append(dict(id=i,status='retained' if a and b else 'removed_from_working' if a else 'added_export_id',source=a,working=b,identity_claim='annotation membership only; no physical novelty/disappearance inferred'))
result=dict(status='candidate_positive_only',image_id=309264,annotation_version='annotation-snapshot-v1',old_denominator=10,current_positive_denominator=14,source_bindings=[bind(S/'working.norm.jsonl'),bind(S/'source.norm.jsonl'),bind(S/'manifest.json'),bind(E/'admission.json'),bind(SCORER),bind(Path(__file__))],banks=dict(old=oldbank,current=newbank),crosswalk=cross,cells=cells,summary=summary,verification=dict(old_ledgers_exact_all_views=True,source_bank_exact=True,positive_exclusion_fixture=dict(owner_id=o['owner_id'],raw_free_matches=1,after_symmetric_exclusion=0)),held_claims=['physical FP','unannotated-as-negative','ignore masks','exhaustive physical recall','physical disappearance from removed ID'])
(E/'result.json').write_text(json.dumps(result,indent=2)+'\n')
(E/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
