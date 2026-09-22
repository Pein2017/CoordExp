"""CPU numerical recurrence accounting; physical identity is deliberately separate."""
from probes.training_set_completion.numerical_feedback.select import rows,same,ROLES

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

def selfcheck():
 def tokenrow(v):return [151646,100,151647,151648]+[151670+x for x in v]+[151649]
 native=rows(tokenrow([0,0,0,0]))[0];m=release_metrics(tokenrow([0,0,0,0])*3+[151645],native,'x1',1);assert m['invalid_rows']==3 and m['longest_exact_run']==3 and m['same_role_copy']['rows_equal_a']==3 and m['native_exact_return_rows']==[1,2,3]
 assert crossed_margin([3.,1.],[0.,2.],0,1)==4.
 r=rows(tokenrow([0,1,2,3])+tokenrow([7,1,2,3])+tokenrow([14,1,2,3]));assert longest_run(r,8)==2
 assert release_metrics([151646,100,151649],native)['malformed_openers']==1
 print('PASS positive K sign, invalid literal return, nontransitive near-run, malformed accounting')
if __name__=='__main__':selfcheck()
