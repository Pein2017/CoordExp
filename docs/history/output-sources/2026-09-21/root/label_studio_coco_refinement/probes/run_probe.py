import json, shutil, time, hashlib, resource, traceback
from pathlib import Path
from src.label_studio_coco_refinement.store import *
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
from src.label_studio_coco_refinement.materialize import WorkingCoordMaterializer
from src.data import iter_raw_examples

ROOT=Path('outputs/label_studio_coco_refinement/probes').resolve(); ROOT.mkdir(parents=True,exist_ok=True)
SRC=Path('public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl').resolve()
IMG=Path('public_data/coco/rescale_32_1024_bbox/images').resolve()
source_hash=sha256_file(SRC)
result={'source':str(SRC),'source_sha256_before':source_hash,'source_size':SRC.stat().st_size,'runs':[]}
lines=SRC.read_text().splitlines(True)

def spec(src, root, pid):
 return BootstrapSpec(split='train',source_path=src,runtime_root=root,image_root=IMG,expected_source_sha256=sha256_file(src),project_id=pid,storage_id='storage-'+pid,adapter_version='label-studio-coco-refinement-v1',vendor_revision='probe',registry_fingerprint=COCO80_REGISTRY.fingerprint,label_config_fingerprint='probe')

def make_req(store, cid):
 r=store.restore_draft(9)
 regs=[]
 for o in r.row['objects']:
  x=dict(o); x['region_key']=next(k for k,v in r.region_id_mapping.items() if v==o['coco_ann_id']); regs.append(x)
 regs[0]['bbox_2d']=[max(0,regs[0]['bbox_2d'][0]+1),*regs[0]['bbox_2d'][1:]]
 sh=semantic_hash(regs); ds=DraftSaveReceipt('probe-project','train:9','annotation:9','draft:'+cid,1,sh)
 return CommitRequest(cid,'train',9,'probe-project','train:9','annotation:9','draft:'+cid,1,sh,r.row_hash,r.generation,regs,ds)

for n in (1000,10000,len(lines)):
 name='full' if n==len(lines) else str(n)
 d=ROOT/f'bootstrap_{name}'; shutil.rmtree(d,ignore_errors=True); (d/'public_data/coco/rescale_32_1024_bbox_len12000').mkdir(parents=True); (d/'public_data/coco/rescale_32_1024_bbox').symlink_to(Path('public_data/coco/rescale_32_1024_bbox').resolve(), target_is_directory=True)
 src=d/'public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl'; src.write_text(''.join(lines[:n]))
 t=time.perf_counter(); rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 try:
  b=WorkingDatasetStore.bootstrap(spec(src,d/'runtime','probe-'+name)); elapsed=time.perf_counter()-t
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss0
  st=b.store; req=make_req(st,'commit-'+name); tc=time.perf_counter(); cr=st.commit(req); ct=time.perf_counter()-tc
  run={'rows':n,'bootstrap_s':elapsed,'bootstrap_rss_kb_delta':rss,'task_count':b.task_count,'task_manifest_hash':b.task_manifest_hash,'commit_s':ct,'commit_status':cr.status.value,'generation':cr.generation,'row_hash':cr.row_hash,'working_bytes':st.working_path.stat().st_size,'journal_bytes':st.journal_path.stat().st_size,'manifest':json.loads(st.manifest_path.read_text())}
 except BaseException as e:
  run={'rows':n,'error_type':type(e).__name__,'error':str(e),'trace':traceback.format_exc()}
 result['runs'].append(run)
 if n==len(lines) and run.get('commit_s',0)>5: result['full_commit_stopped']=True; break

# fault boundaries on bounded 1000-row slice, fresh store each
bounds=['prepared_journal_flushed','prepared_journal_fsynced','working_temp_flushed','working_temp_fsynced','working_replaced','working_directory_fsynced','manifest_temp_flushed','manifest_temp_fsynced','manifest_replaced','manifest_directory_fsynced','terminal_journal_flushed','terminal_journal_fsynced','before_response']
fr=[]
for boundary in bounds:
 d=ROOT/f'fault_{boundary}'; shutil.rmtree(d,ignore_errors=True); (d/'public_data/coco/rescale_32_1024_bbox_len12000').mkdir(parents=True); (d/'public_data/coco/rescale_32_1024_bbox').symlink_to(Path('public_data/coco/rescale_32_1024_bbox').resolve(), target_is_directory=True); src=d/'public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl'; src.write_text(''.join(lines[:1000]))
 seen=[]
 def inj(x):
  seen.append(x)
  if x==boundary: raise InjectedCrash(x)
 try:
  st=WorkingDatasetStore.bootstrap(spec(src,d/'runtime','fault-'+boundary),fault_injector=inj).store
  req=make_req(st,'fault-'+boundary)
  try: st.commit(req); outcome='returned'
  except BaseException as e: outcome=type(e).__name__+':'+str(e)
  rec=WorkingDatasetStore(st.split_dir)
  fr.append({'boundary':boundary,'seen':seen,'commit_call':outcome,'recovered_status':rec.status(req.commit_id).value,'generation':rec.restore_draft(9).generation,'journal_bytes':rec.journal_path.stat().st_size})
 except BaseException as e: fr.append({'boundary':boundary,'error':type(e).__name__+':'+str(e),'trace':traceback.format_exc()})
result['faults']=fr

# materializer with negative object ID, then actual loader
md=ROOT/'materializer'; shutil.rmtree(md,ignore_errors=True); md.mkdir(); (md/'images').symlink_to(IMG, target_is_directory=True); wn=md/'working.norm.jsonl';
# Construct a valid working-row locator under the disposable probe root.
row=json.loads(lines[0]); row['images']=['images/train2017/000000000009.jpg']; row['file_name']='images/train2017/000000000009.jpg'; row['objects'][0]['coco_ann_id']=-123456; wn.write_text(json.dumps(row,separators=(',',':'))+'\n')
out=md/'working.coord.jsonl'; mr=WorkingCoordMaterializer().materialize(wn,out)
loaded=list(iter_raw_examples(out)); result['materializer']={'receipt':mr.to_artifact_dict(),'loader_rows':len(loaded),'loader_first_type':type(loaded[0]).__name__,'loader_first_repr':repr(loaded[0])[:1000],'coord_text':out.read_text()[:1000]}
result['source_sha256_after']=sha256_file(SRC); result['source_unchanged']=result['source_sha256_before']==result['source_sha256_after']
(ROOT/'receipt.json').write_text(json.dumps(result,indent=2,default=str))
print(json.dumps(result,indent=2,default=str))
