"""Literal numerical recurrence, never physical-owner classification.

The frozen compact token vocabulary and pairwise eight-bin comparisons are
explicit. Adjacent near-equality is not treated as a transitive relation.
"""
ROLES = ['x1', 'y1', 'x2', 'y2']


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


def longest_run(rr,eps):
 best=0
 for i,a in enumerate(rr):
  run=[]
  for b in rr[i:]:
   if b['description_tokens']!=a['description_tokens'] or any(not same(b,c,eps) for c in run):break
   run.append(b);best=max(best,len(run))
 return best


def release_metrics(tokens,source,role=None,b=None):
 rr=rows(tokens);native=[i+1 for i,r in enumerate(rr) if same(r,source,0)];near=[i+1 for i,r in enumerate(rr) if same(r,source,8)];edited=dict(source,values=source['values'].copy())
 if role is not None:edited['values'][ROLES.index(role)]=b
 alternate=[]
 for i in range(len(rr)-2):
  if same(rr[i],rr[i+1],8) and same(rr[i],rr[i+2],8) and same(rr[i+1],rr[i+2],8) and not same(rr[i],source,8):alternate.append(i+1)
 out=dict(complete_rows=len(rr),invalid_rows=sum(not r['valid'] for r in rr),malformed_openers=max(0,tokens.count(151646)-len(rr)),unparsed_tokens=len(tokens)-sum(r['end']-r['start'] for r in rr)-int(151645 in tokens),eos=151645 in tokens,first_row=rr[0] if rr else None,native_exact_return_rows=native,native_near_return_rows=near,substituted_exact_return_rows=[i+1 for i,r in enumerate(rr) if same(r,edited,0)],substituted_near_return_rows=[i+1 for i,r in enumerate(rr) if same(r,edited,8)],longest_exact_run=longest_run(rr,0),longest_near_run=longest_run(rr,8),alternate_repeat_starts=alternate,physical_recovery='not_established_by_numerical_metrics')
 if role is not None:
  k=ROLES.index(role);a=source['values'][k];out['same_role_copy']=dict(a=a,b=b,first_equals_a=bool(rr and rr[0]['values'][k]==a),first_equals_b=bool(rr and rr[0]['values'][k]==b),rows_equal_a=sum(r['values'][k]==a for r in rr),rows_equal_b=sum(r['values'][k]==b for r in rr),first_distance_to_a=abs(rr[0]['values'][k]-a) if rr else None,first_distance_to_b=abs(rr[0]['values'][k]-b) if rr else None)
 return out


def crossed_margin(original,modified,a,b):
 return float((original[a]-original[b])-(modified[a]-modified[b]))
