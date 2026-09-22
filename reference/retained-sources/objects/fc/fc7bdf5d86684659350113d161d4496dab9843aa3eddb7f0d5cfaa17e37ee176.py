"""Deterministic token-row/episode/replacement selection; CPU only, before outcomes."""
import json,hashlib,argparse
from pathlib import Path
R=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback')
ROLES=['x1','y1','x2','y2']
def token_hash(t):return hashlib.sha256(json.dumps(t,separators=(',',':')).encode()).hexdigest()
def rows(tokens):
 out=[]
 for start,v in enumerate(tokens):
  if v!=151646:continue
  try:end=tokens.index(151647,start+1)
  except ValueError:continue
  if 151646 in tokens[start+1:end] or end+6>=len(tokens):continue
  if tokens[end+1]!=151648 or tokens[end+6]!=151649:continue
  coords=tokens[end+2:end+6]
  if not all(151670<=t<=152669 for t in coords):continue
  values=[t-151670 for t in coords]
  out.append(dict(index=len(out),start=start,end=end+7,description_tokens=tokens[start+1:end],coordinate_offsets=list(range(end+2,end+6)),values=values,valid=values[0]<values[2] and values[1]<values[3]))
 return out

def same(a,b,eps=0):return a['description_tokens']==b['description_tokens'] and max(abs(x-y) for x,y in zip(a['values'],b['values']))<=eps

def choose_episode(rr):
 for eps,label in [(0,'literal'),(8,'near')]:
  for i in range(len(rr)-2):
   if all(same(rr[j],rr[k],eps) for j,k in [(i,i+1),(i,i+2),(i+1,i+2)]):return i,label
 return None,None

def select(panel):
 boundaries=[];exclusions=[];source=[]
 for iid in panel['case_order']:
  g=next(g for g in panel['groups'] if iid in g['focus_ids']);bi=next(j for j,c in enumerate(g['cases']) if c['input_record']['image_id']==iid)
  for model in ['untied','tied']:
   saved=next((s for s in panel['saved_sources'] if s['condition']==model+'-original' and s['group']==g['key']),None)
   base=R/'runtime'/f'{model}-original'/g['key'];rawpath=Path(saved['raw']['path']) if saved else base/'raw.json';tracepath=Path(saved['trace']['path']) if saved else base/'trace.json';receiptpath=Path(saved['receipt']['path']) if saved else base/'receipt.json'
   if not rawpath.exists():exclusions.append(dict(image_id=iid,model=model,status='pending_natural'));continue
   raw=json.loads(rawpath.read_text())['rows'][bi];t=raw['token_ids'];rr=rows(t);idx,kind=choose_episode(rr);picked=[]
   if idx is not None:picked.append(('failure',idx,kind))
   else:exclusions.append(dict(image_id=iid,model=model,status='no_qualifying_episode',complete_rows=len(rr)))
   recurrent={j for i in range(len(rr)-2) if all(same(rr[k],rr[l],8) for k,l in [(i,i+1),(i,i+2),(i+1,i+2)]) for j in range(i,i+3)}
   healthy=[i for i in range(len(rr)-1) if i not in recurrent and not any(same(rr[i],r,8) for r in rr[:i])]
   if healthy:
    if idx is not None:
     ref=rr[idx];h=min(healthy,key=lambda j:(rr[j]['valid']!=ref['valid'],sum(abs(a-b) for a,b in zip(rr[j]['values'],ref['values'])),abs(rr[j]['end']-ref['end']),j))
    else:h=healthy[-1]
    picked.append(('healthy',h,'nonrecurrent_proxy'))
   else:exclusions.append(dict(image_id=iid,model=model,status='healthy_boundary_absent'))
   for typ,i,label in picked:
    targets=[dict(row_delay=delay-1,row_index=i+delay,role=role,offset=rr[i+delay]['coordinate_offsets'][k]) for delay in [1,2,4] if i+delay<len(rr) for k,role in enumerate(ROLES)]
    boundaries.append(dict(id=f'{model}-{iid}-{typ}',model=model,image_id=iid,group=g['key'],batch_index=bi,kind=typ,episode_stratum=label,source_row=rr[i],previous_row=rr[i-1] if i else None,next_row=rr[i+1] if i+1<len(rr) else None,target_slots=targets,fixed_suffix_end=rr[max(x['row_index'] for x in targets)]['end'] if targets else rr[i]['end'],native_tokens=t,native_token_hash=token_hash(t),prefix_hash=token_hash(t[:rr[i]['end']]),raw_path=str(rawpath),trace_path=str(tracepath),receipt_path=str(receiptpath),healthy_matching='same image/model; validity, coordinate L1 distance, prefix length priority; no exact-match claim',physical_status='not_required_UNKNOWN'))
 assert len(boundaries)<=28
 return dict(status='frozen' if not any(e['status']=='pending_natural' for e in exclusions) else 'partial',boundaries=boundaries,exclusions=exclusions,definition='failure source is first completed row of earliest qualifying triple; not necessarily already third repeat. Near triple pairwise maximum <=8. Delays0,1,3 mean next,+2,+4 native rows.',panel_sha256=hashlib.sha256((R/'panel.json').read_bytes()).hexdigest())

def order_stratum(row,previous,nextrow):
 anchor=tuple(row[:2]);sign=lambda x:(x>0)-(x<0)
 def cmp(a,b):return (a>b)-(a<b)
 return [cmp(anchor,tuple(previous['values'][:2])) if previous else None,cmp(tuple(nextrow['values'][:2]),anchor) if nextrow else None]

def replacements(boundary,role,logits):
 r=boundary['source_row'];k=ROLES.index(role);a=r['values'][k];prev=boundary['previous_row'];nxt=boundary['next_row'];native_order=order_stratum(r['values'],prev,nxt)
 def info(b):
  box=r['values'].copy();box[k]=b;valid=box[0]<box[2] and box[1]<box[3];order=order_stratum(box,prev,nxt)
  return dict(value=b,validity_preserved=valid==r['valid'],ordering_preserved=order==native_order,valid=valid,order_stratum=order)
 def pool(values):
  xs=[info(b) for b in values];best=min((not x['validity_preserved'],not x['ordering_preserved']) for x in xs);return [x for x in xs if (not x['validity_preserved'],not x['ordering_preserved'])==best]
 adj=pool([b for b in [a-1,a+1] if 0<=b<1000]);adj=min(adj,key=lambda x:(-float(logits[x['value']]),x['value']));far=pool([b for b in range(1000) if abs(b-a)>=16]);far=min(far,key=lambda x:(-float(logits[x['value']]),x['value']));oppvals=[b for b in range(1000) if (b-a)*(far['value']-a)<0];out=[dict(kind='adjacent',**adj),dict(kind='far',**far)]
 if oppvals:
  opp=pool(oppvals);opp=min(opp,key=lambda x:(abs(abs(x['value']-a)-abs(far['value']-a)),-float(logits[x['value']]),x['value']));out.append(dict(kind='opposite',**opp))
 for x in out:x.update(original_value=a,role=role,source_offset=r['coordinate_offsets'][k],bin_distance=abs(x['value']-a),native_logit=float(logits[x['value']]),native_rank=1+sum(float(v)>float(logits[x['value']]) for v in logits))
 return out

def selfcheck():
 def row(v):return [151646,100,151647,151648]+[151670+x for x in v]+[151649]
 t=row([0,1,2,3])*3+[151645];rr=rows(t);assert len(rr)==3 and rr[0]['end']==9 and choose_episode(rr)==(0,'literal')
 assert choose_episode(rows(row([0,1,2,3])+row([7,1,2,3])+row([14,1,2,3])))==(None,None)
 b=dict(source_row=rr[0],previous_row=None,next_row=rr[1]);r=replacements(b,'x1',list(range(1000)));assert r[0]['value']==1 and all(x['value']!=0 for x in r);assert all(x['source_offset']==4 for x in r)
 print('PASS token boundary, invalid retention, near nontransitivity, replacement role')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--selfcheck',action='store_true');a=p.parse_args()
 if a.selfcheck:selfcheck()
 else:
  s=select(json.loads((R/'panel.json').read_text()));(R/'selection.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(dict(status=s['status'],boundaries=len(s['boundaries']),exclusions=s['exclusions'])))
