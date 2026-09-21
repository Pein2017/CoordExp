"""Freeze already reviewed bird anchors; no outcomes used for selection."""
import copy,hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'2026-09-17-owner-recurrence-row-branch';F=R.parent/'2026-09-17-readout-norm-fresh128'
read=lambda p:json.loads(p.read_text())
def bind(p):p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
p=read(P/'panel.json');c=copy.deepcopy(p['cases'][0]);assert c['image_id']==309264
raw=read(Path(c['saved_raw']['path']))['rows'][c['target_position']]['token_ids'];a=c['arms']['native']['token_ids'];b=c['arms']['distinct']['token_ids'];alt=c['arms']['same']['token_ids'];d=read(P/'runtime/309264/distinct/raw.json')['rows'][c['target_position']]['token_ids'][54:63]
assert all(len(x)==9 for x in (a,b,alt,d));assert raw[36:45]==a
base=raw[:36];hist={x+y:base+(a if x=='A' else b)+(a if y=='A' else b)+a+b for x in 'AB' for y in 'AB'}
hist['native_onset']=raw[:45];hist['native_length']=raw[:72]
c['score_histories']=hist;c['score_candidates']={k:dict(token_ids=v) for k,v in [('A',a),('B',b),('C',d),('A_alt',alt),('EOS',[151645])]}
c['anchor_evidence']=dict(A='physical:left_lower_cage_bird',B='known:367404',C='known:42098',A_alt='same lower-left bird accepted small extent edit',source=bind(P/'physical-review.json'),Hbase='original first four rows,36tokens; earlier same-owner exposure may be present and is fixed',native_reference='45token onset and72token length-matched natural histories; distinct conditioning from synthetic panel')
c['supplied_known_owner_union']=['367404'];c['supplied_physical_owner_union']=['physical:left_lower_cage_bird','known:367404']
f=read(F/'panel.json');sources=p['sources']+[bind(P/'panel.json'),bind(P/'physical-review.json'),bind(P/'runtime/309264/distinct/raw.json'),bind(F/'coefficients.pt'),bind(Path('probes/training_set_completion/repetition_history_scores.py').resolve())]
sources=list({s['path']:s for s in sources}.values())
deltas=[]
for j,x in enumerate(sources):
 current=bind(x['path'])
 if current!=x:
  assert x['path'].endswith('/src/qwen/native.py')
  old=subprocess.check_output(['git','show','af0cc5973^:src/qwen/native.py'])
  assert hashlib.sha256(old).hexdigest()==x['sha256']
  (R/'native-before.py').write_bytes(old)
  diff=subprocess.check_output(['git','diff','af0cc5973^','af0cc5973','--','src/qwen/native.py'])
  (R/'native-additive-helper.diff').write_bytes(diff)
  deltas.append(dict(predecessor=x,current=current,diff=bind(R/'native-additive-helper.diff'),old_snapshot=bind(R/'native-before.py'),impact='added unused singleton-combination helper; native replay required'))
  sources[j]=current
(R/'source-deltas.json').write_text(json.dumps(deltas,indent=2)+'\n')
panel=dict(schema='repetition_history.stage_a.v1',config=p['config'],cases=[c],banks={'309264':p['banks']['309264']},sources=sources,coordinate_ids=f['coordinate_ids'],coefficients=f['coefficients'],score_forward_bound_per_case=64,bounds=dict(full_rollouts=7,score_forwards=64,cap=3084),scope='four-cell earlier relative exposure, native onset and length references; A_alt candidate only; no pure A-count claim')
out=R/'stage-a-panel.json';assert not out.exists();out.write_text(json.dumps(panel,indent=2)+'\n');print(json.dumps(dict(panel=bind(out),prefix_lengths={k:len(v) for k,v in hist.items()},score_rows=len(hist)*len(c['score_candidates']))))
