"""Apply predeclared specificity gate and deduplicate native-rebuild prefixes."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'2026-09-17-history-rereading-mechanism';Q=R.parent/'2026-09-17-successful-row-mechanism'
read=lambda p:json.loads(p.read_text())
bind=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
paths={n:P/f'runtime/{v}' for n,v in [('F','native-F'),('R','residual-only'),('H','cache-only'),('RH','joint')]};paths.update({n:R/f'runtime/{n}' for n in ['W','RW','HW','RHW']})
reduction=read(R/'reduction.json')['cells'];pairs=dict(W='F',RW='R',HW='H',RHW='RH');summary={};eligible=[]
for n,p in paths.items():
 rec=read(p/'receipt.json');raw=read(p/'raw.json')['rows'][1];v=reduction[n];summary[n]=dict(fork=rec['fork'],primary=v['primary'],burden=v['versions']['current']['free']['burden'],stop=raw['stop'],tokens=len(raw['token_ids']),receipt=bind(p/'receipt.json'))
for n,b in pairs.items():
 a=summary[n];z=summary[b];new=set(a['primary']['free']);old=set(z['primary']['free'])
 gate=dict(winner_changed=a['fork']['argmax']!=z['fork']['argmax'],positive_gain=len(new-old)>=1,retention=not(old-new),debt_not_worse=all(a['burden'][k]<=z['burden'][k] for k in ['invalid','malformed','strict_valid_repeats','literal_repeats']))
 a['specificity_gate']=gate
 if n!='RHW' and all(gate.values()):eligible.append(n)
selected=next((n for n in ['W','RW','HW'] if n in eligible),None)
# Mechanical reuse requires equal first68 in every batch member plus native runtime/inputs.
controls={'coord0':P/'runtime/rebuild-cache-only','coord131':Q/'stage2/runtime/rebuild-S-to-F'}
rebuilds={}
for n in ['W','RW','HW','RHW']:
 rows=read(paths[n]/'raw.json')['rows'];nr=read(paths[n]/'receipt.json');reuse=[]
 for key,c in controls.items():
  cr=read(c/'receipt.json');old=read(c/'raw.json')['rows']
  identity=cr.get('loaded_identity',cr.get('loaded_model_identity'))
  if all(a['token_ids'][:68]==b['token_ids'][:68] for a,b in zip(rows,old)) and nr['input_identity']==cr['input_identity'] and nr['loaded_identity']==identity:
   reuse.append(dict(control=key,raw=bind(c/'raw.json'),receipt=bind(c/'receipt.json'),target_full_equal=rows[1]['token_ids']==old[1]['token_ids'],all_full_equal=all(a['token_ids']==b['token_ids'] and a['stop']==b['stop'] for a,b in zip(rows,old)),first_target_difference=next((i for i,(a,b) in enumerate(zip(rows[1]['token_ids'],old[1]['token_ids'])) if a!=b),None)))
 rebuilds[n]=reuse
out=dict(status='candidate',factorial=summary,selected_specificity=selected,priority=['W','RW','HW'],rebuild_reuse=rebuilds,rule='No donor search or additional controls for a nonqualifying partial effect')
(R/'factorial-decision.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(selected=selected,cells={n:dict(winner=v['fork']['argmax'],margin=v['fork']['coord131_minus_coord0'],free=v['primary']['free'],stop=v['stop'],tokens=v['tokens'],gate=v.get('specificity_gate')) for n,v in summary.items()},rebuilds=rebuilds)))
