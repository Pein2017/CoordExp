"""Descriptive paired effects from independently replayed admitted spatial states."""
import json,statistics
from pathlib import Path
base=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1/reduced')
states=[json.loads(p.read_text()) for p in base.glob('*.json')]
result={}
for kind in ['failure','proxy']:
 ss=[d for d in states if d['admission']['mode']==kind and d['admission']['admitted']]
 r={'admitted_states':len(ss),'cells':{}}
 for key in ['00','10-','10+','01-','01+','11-','11+']:
  vals=[d['cells'][key] for d in ss]
  r['cells'][key]={'recurrence_states':sum(c['parse']['failure_predicate'] for c in vals),'complete_rows':sum(c['parse']['complete_rows'] for c in vals),'invalid_rows':sum(c['parse']['invalid_rows'] for c in vals),'row_cap':sum(c['row_limit']['injected_eos'] for c in vals)}
 r['conditional_window_logratio_effects']={}
 for key in ['10','01','11']:
  v=[]
  for d in ss:
   for sign in ['-','+']:
    def ratio(cell):return cell['boundary']['windows_by_sign'][sign]['moved_minus_old_log_mass']
    v.append(ratio(d['cells'][key+sign])-ratio(d['cells']['00']))
  r['conditional_window_logratio_effects'][key]={'paired_state_signs':len(v),'median':statistics.median(v),'positive':sum(x>0 for x in v),'min':min(v),'max':max(v)}
 result[kind]=r
out=Path(__file__).with_name('spatial-effect-summary.json')
if out.exists(): assert json.loads(out.read_text())==result
else: out.write_text(json.dumps(result,indent=2)+'\n')
print('PASS spatial effect summary')
