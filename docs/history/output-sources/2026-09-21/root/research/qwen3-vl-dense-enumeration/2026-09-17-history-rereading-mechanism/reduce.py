import json
from pathlib import Path
from probes.training_set_completion.successful_row_reduce import reduce
R=Path(__file__).resolve().parent;P=R.parent/'2026-09-17-successful-row-mechanism'
read=lambda p:json.loads(p.read_text())
cells=[]
for p in sorted((R/'runtime').glob('*/receipt.json')):
 rec=read(p)
 if rec['mode']!='full':continue
 assert rec['status']=='candidate_complete',p
 name=p.parent.name
 cells.append(dict(id=name,panel=rec['panel']['path'],raw=str(p.parent/'raw.json'),receipt=str(p),
                   baseline='native-F',comparison_group='F',intervention_offset=67))
# Same conservative first-row exclusion on reference S; separate group avoids altering the F comparison union.
cells.append(dict(id='reference-S',panel=str(P/'stage2/panels/S.json'),raw=str(P/'stage2/runtime/native-S/raw.json'),
                  receipt=str(P/'stage2/runtime/native-S/receipt.json'),baseline='reference-S',comparison_group='S-reference',intervention_offset=67))
manifest=R/'reduction-manifest.json';manifest.write_text(json.dumps(dict(cells=cells),indent=2)+'\n')
result=reduce(manifest);out=R/'reduction.json';out.write_text(json.dumps(result,indent=2)+'\n')
summary=[]
for name,c in result['cells'].items():
 row=read(Path(c['raw']['path']))['rows'][1]
 b=c['free']['burden'];summary.append(dict(condition=name,free=c['symmetric_known_accounting'],burden=b,
     tokens=len(row['token_ids']),stop=row['stop'],release=c['release']))
(R/'reduction-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps([dict(condition=x['condition'],free=x['free']['free'],tokens=x['tokens'],stop=x['stop'],invalid=x['burden']['invalid']) for x in summary]))
