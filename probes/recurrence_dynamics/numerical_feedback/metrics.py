"""CPU numerical recurrence accounting; physical identity is deliberately separate."""
from src.eval.numerical_recurrence import rows, same, ROLES

from src.eval.numerical_recurrence import longest_run, release_metrics, crossed_margin







def selfcheck():
 def tokenrow(v):return [151646,100,151647,151648]+[151670+x for x in v]+[151649]
 native=rows(tokenrow([0,0,0,0]))[0];m=release_metrics(tokenrow([0,0,0,0])*3+[151645],native,'x1',1);assert m['invalid_rows']==3 and m['longest_exact_run']==3 and m['same_role_copy']['rows_equal_a']==3 and m['native_exact_return_rows']==[1,2,3]
 assert crossed_margin([3.,1.],[0.,2.],0,1)==4.
 r=rows(tokenrow([0,1,2,3])+tokenrow([7,1,2,3])+tokenrow([14,1,2,3]));assert longest_run(r,8)==2
 assert release_metrics([151646,100,151649],native)['malformed_openers']==1
 print('PASS positive K sign, invalid literal return, nontransitive near-run, malformed accounting')
if __name__=='__main__':selfcheck()
