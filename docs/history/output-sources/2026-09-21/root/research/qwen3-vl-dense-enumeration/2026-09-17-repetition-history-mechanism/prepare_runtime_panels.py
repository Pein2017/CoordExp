"""Freeze the declared history/pulse cells before their rollouts."""
import copy,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'2026-09-17-owner-recurrence-row-branch';F=R.parent/'2026-09-17-readout-norm-fresh128'
read=lambda p:json.loads(p.read_text())
def bind(p):p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
a=read(R/'stage-a-panel.json');p=read(P/'panel.json');f=read(F/'panel.json');z=read(F/'result.json')
producer=Path('probes/training_set_completion/repetition_history_runtime.py').resolve()
sources=[bind(x['path']) for x in a['sources'] if not x['path'].endswith('repetition_history_scores.py')]+[bind(producer),bind(R/'stage-a-panel.json')]
sources=list({x['path']:x for x in sources}.values())
base=dict(config=a['config'],sources=sources,coordinate_ids=f['coordinate_ids'],coefficients=f['coefficients'],schema='repetition_history.runtime.v1')
manifest=[]
def save(stage,iid,name,mode,case):
 out=R/'panels'/stage/f'{iid}-{name}.json';out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists(),out
 panel=copy.deepcopy(base);panel['cases']=[case];panel['condition']=name;panel['bounds']=dict(batch_executions=1,max_model_forwards=3084)
 out.write_text(json.dumps(panel,indent=2)+'\n');manifest.append(dict(stage=stage,image_id=iid,condition=name,mode=mode,panel=bind(out),output_root=str(R/'runtime'/stage/name)))
for c0 in p['cases']:
 c=copy.deepcopy(c0);iid=c['image_id'];ids=read(Path(c['saved_raw']['path']))['rows'][c['target_position']]['token_ids'];onset=c['start_offset'];established=c['end_offset'];pre=onset-9
 c['target_prefix_token_ids']=[];c['common_native_prefix_length']=3084;c['native_identity']=True;c['capture_offsets']=sorted(set([pre,onset,established]+list(range(72,77)) if iid==309264 else [pre,onset,established]))
 save('B',iid,'native','prefix',c)
 c=copy.deepcopy(c);c['native_identity']=False;c['common_native_prefix_length']=0;c['norm_start']=0;c['expected_target_raw']=z['images'][str(iid)]['raw_bindings']['N'];save('B',iid,'full_norm','sustained',c)
 for name,mode,start in [('pre_entry_one','pulse1',pre),('established_one','pulse1',established),('established_four','pulse4',established),('late_sustained','sustained',established)]:
  c=copy.deepcopy(c0);c.update(target_prefix_token_ids=ids[:start],common_native_prefix_length=start,norm_start=start,native_identity=False,capture_offsets=list(range(start,start+9)))
  save('B',iid,name,mode,c)
for iid in [309264]:
 c0=a['cases'][0]
 for name,h in c0['score_histories'].items():
  c=copy.deepcopy(c0);c.update(target_prefix_token_ids=h,common_native_prefix_length=36 if name in ['AA','AB','BA','BB'] else len(h),native_identity=False,capture_offsets=list(range(len(h),len(h)+9)))
  save('A',iid,name,'prefix',c)
(R/'runtime-manifest.json').write_text(json.dumps(dict(cells=manifest,replay_controls='three pulse arms per B case; exact observed pulse token prefix replay then unrestricted original greedy, mechanically derived only',total_initial_batches=len(manifest),max_replay_batches=6,total_forward_bound=(len(manifest)+6)*3084),indent=2)+'\n')
print(json.dumps(dict(cells=len(manifest),replays=6,root=str(R/'panels'))))
