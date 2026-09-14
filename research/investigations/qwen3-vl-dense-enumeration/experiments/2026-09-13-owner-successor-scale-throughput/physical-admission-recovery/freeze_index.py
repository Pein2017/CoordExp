import json, hashlib, pathlib, re
O=pathlib.Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput')
D=O/'physical-admission-recovery'
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def bind(p): return dict(path=str(p),sha256=sha(p),size_bytes=pathlib.Path(p).stat().st_size)
def read(p): return json.loads(pathlib.Path(p).read_text())
completion=O/'supply-recovery/completion.json'
assert sha(completion)=='d2db76808104049317fda9d7f60de082a98ac2f562df2c4d149ac4f833f5014f'
C=read(completion); pool=read(O/'supply/pool.json'); order={int(i):n for n,i in enumerate(pool['image_ids'])}
sources={}; jobs={}; rows={}; images={}
for entry in C['source_bindings']:
 b=entry['binding']; p=pathlib.Path(b['path'])
 if not ('/supply-recovery/full/' in str(p) or '/supply-recovery/slice/' in str(p)): continue
 assert sha(p)==b['sha256']
 sid='recovery-'+('full-' if '/full/' in str(p) else 'slice-')+p.parent.name
 sources.setdefault(sid,{})[p.stem]=b
 target={'jobs':jobs,'rows':rows,'images':images}[p.stem]
 for i,line in enumerate(p.read_text().splitlines()):
  x=json.loads(line); key=x.get('job_id',x.get('image_id')); ident=dict(binding=b,line_1based=i+1,line_sha256=hashlib.sha256(line.encode()).hexdigest(),source_shard_id=sid)
  if p.stem in ['images','jobs']: target.setdefault(key,[]).append((x,ident))
  else:
   assert key not in target; target[key]=(x,ident)
canonical_jobs={x['job_id']:x for x in read(O/'supply-recovery/ordered-jobs.json')['jobs']}
canonical_rows={x['job_id']:x for x in read(O/'supply-recovery/ordered-results.json')['rows']}
canonical_images={x['image_id']:x for x in read(O/'supply-recovery/image-records.json')['records']}
cards={x['job_id']:x for x in C['cards']}
selected=[]; groups=[]; bykey={}
for jid,(r,ri) in sorted(rows.items(),key=lambda x:(order[int(x[1][0]['image_id'])],int(x[0].split(':h')[1].split(':')[0]),int(x[0].split(':c')[1]))):
 if r['local_w']['status']!='candidate_local_w': continue
 j,ji=next((x,i) for x,i in jobs[jid] if i['source_shard_id']==ri['source_shard_id']); assert j==canonical_jobs[jid] and r==canonical_rows[jid]
 image_id=int(j['image_id']); im=canonical_images[image_id]; case=im['frozen']['case']; w=r['local_w']
 key=(image_id,j['c_description'],tuple(j['c_bbox']),w['w_description'],tuple(w['w_bbox_xyxy_pixels']))
 if key not in bykey:
  if len(groups)>=80: continue
  gid=f'PAR-{len(groups)+1:04d}'; bykey[key]=gid
  card=cards[jid]['card']; assert sha(card['path'])==card['sha256']
  groups.append(dict(visual_group_id=gid,image_id=image_id,job_ids=[],representative_card=dict(path=card['path'],binding=card,representative_job_id=jid),review_status='pending_view_image',training_target=False))
 gid=bykey[key]; group=next(x for x in groups if x['visual_group_id']==gid); group['job_ids'].append(jid)
 image_sources=[ident for _,ident in images.get(image_id,[])]
 def axis(desc,box,text,tids):return dict(description=desc,bbox_native_pixels_xyxy=box,literal_text=text,token_ids_sha256=hashlib.sha256(json.dumps(tids,separators=(',',':')).encode()).hexdigest(),physical_review=dict(status='pending_view_image'))
 selected.append(dict(job_id=jid,image_id=image_id,example_id=j['example_id'],visual_group_id=gid,shard=ri['source_shard_id'],source_identity=dict(candidate_index=j['candidate_index'],history_index=j['history_index'],later_row_ordinal=j['later_row_ordinal'],row_id=jid,output_shard=ri['source_shard_id'],output_status=w['status'],packet=r['packet']),source_job=ji,source_row=ri,source_image_records=image_sources,source_job_ordinal=ji['line_1based']-1,source_row_ordinal=ri['line_1based']-1,source_job_sha256=ji['line_sha256'],source_row_sha256=ri['line_sha256'],image=dict(path=case['image_path'],sha256=sha(case['image_path']),dimensions_native_pixels=[case['image_width'],case['image_height']],image_id=image_id,example_id=j['example_id']),c=axis(j['c_description'],j['c_bbox'],j['c_text'],j['c_token_ids']),w=axis(w['w_description'],w['w_bbox_xyxy_pixels'],w['w_text'],w['w_token_ids']),exact_history=dict(literal_text=j['h_text'],token_ids_sha256=hashlib.sha256(json.dumps(j['h_token_ids'],separators=(',',':')).encode()).hexdigest(),history_boundary=j['history_boundary']),first_owner_axis=j['first_owner'],repeat_bbox=j['repeat_bbox'],training_target=False))
out=dict(schema='owner_successor_scale.physical_review_index.v1',status='frozen_pending_individual_review',scope='First 80 unique c/w visual groups in frozen pool image then history/candidate order; completed recovery slice/full ONLY',admission_boundary='candidate_accept/HOLD only; no root admission or training export',source_bindings=dict(completion=bind(completion),pool=bind(O/'supply/pool.json'),canonical_jobs=bind(O/'supply-recovery/ordered-jobs.json'),canonical_rows=bind(O/'supply-recovery/ordered-results.json'),canonical_images=bind(O/'supply-recovery/image-records.json'),shards=sources),counts=dict(exact_visual_groups=len(groups),candidate_local_w_rows=len(selected),images=len({x['image_id'] for x in groups})),groups=groups,rows=selected)
D.mkdir(exist_ok=True); p=D/'review-index-v1.json'; assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(path=str(p),sha256=sha(p),counts=out['counts'],boundary=out['scope'])))
