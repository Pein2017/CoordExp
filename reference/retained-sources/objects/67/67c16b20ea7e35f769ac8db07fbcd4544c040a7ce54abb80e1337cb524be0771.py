import json, hashlib, csv, html
from pathlib import Path
from src.vis import render_prediction_comparison
R=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128'); O=R/'user-comparison-v1'
p=json.loads((R/'panel.json').read_text()); result=json.loads((R/'result.json').read_text())
cases={c['row_id']:c for g in p['groups'] for c in g['cases']}
summary={}; ledger=[]; exclusions=[]; bindings={}
for cohort in ['fresh','diagnostic']:
 ids=sorted([i for i,x in result['images'].items() if x['cohort']==cohort],key=int)
 for arm in ['baseline','treated']:
  d=O/cohort/arm;d.mkdir(parents=True,exist_ok=True); rows=[]
  for iid in ids:
   x=result['images'][iid][arm]; rid=x['native_parse']['row_id']; c=cases[rid]
   row={k:c[k] for k in ['row_id','image_path','image_width','image_height']}
   row['gt']=[dict(description=b['description'],bbox=b['reference_coord_bins_1000'],owner_id=b['owner_id']) for b in p['banks'][iid]]
   row['pred']=[dict(description=v['description'],bbox=v['bbox_pixel_xyxy'],coord_bins=v['coord_bins_1000'],prediction_id=v['prediction_id']) for v in x['valid_predictions']]
   assert len(row['pred'])==len(x['native_parse']['predictions'])
   rows.append(row)
   exclusions.append(dict(image_id=iid,cohort=cohort,arm=arm,drops=x['drops'],burden=x['burden'],stop=x['stop']))
  for name in ['gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl']:
   (d/name).write_text(''.join(json.dumps(v)+'\n' for v in rows))
 rendered=render_prediction_comparison(O/cohort/'baseline',O/cohort/'treated',O/cohort/'png',left_label='Original greedy',right_label='Coordinate norm equalized')
 m=json.loads(rendered.manifest_path.read_text()); assert len(m['items'])==len(ids)==len(rendered.image_paths)
 assert all(v.exists() for v in rendered.image_paths)
 summary[cohort]={}
 for arm,side in [('baseline','left'),('treated','right')]:
  summary[cohort][arm]={'renderer_class_aware':{k:sum(v[side]['match'][k] for v in m['items']) for k in ['tp','fp','fn']},'study_class_agnostic':{'tp':sum(result['images'][i][arm]['matches']['matched_count'] for i in ids),'fp_annotation_unmatched':sum(len(result['images'][i][arm]['matches']['annotation_unmatched_prediction_ids']) for i in ids),'fn':sum(result['images'][i][arm]['matches']['fn_count'] for i in ids)}}
 for iid,item in zip(ids,m['items'],strict=True):
  entry={'image_id':iid,'cohort':cohort,'row_id':item['row_id'],'png':item['output_png']}
  for arm,side in [('baseline','left'),('treated','right')]:
   entry.update({arm+'_'+k:item[side]['match'][k] for k in ['tp','fp','fn']})
  ledger.append(entry)
links=[]
for v in ledger:
 rel=Path(v['png']).relative_to(O)
 links.append(f'<tr><td>{v["cohort"]}</td><td><a href="{rel}">{v["image_id"]}</a></td><td>{v["baseline_tp"]}/{v["baseline_fp"]}/{v["baseline_fn"]}</td><td>{v["treated_tp"]}/{v["treated_fp"]}/{v["treated_fn"]}</td></tr>')
(O/'index.html').write_text('<meta charset="utf-8"><title>Paired detection review</title><h1>Original vs coordinate norm equalization</h1><p>Class-aware IoU 0.5. Green: matched predictions; red: annotation-unmatched predictions (not established physical FP); yellow: missing GT; purple dashed: duplicate hints. Matched GT hidden. Invalid/malformed outputs excluded by native parser; see excluded-outputs.json. No GT or labels changed.</p><table border="1"><tr><th>Cohort</th><th>Image / comparison</th><th>Original TP/FP/FN</th><th>Normalized TP/FP/FN</th></tr>'+''.join(links)+'</table>')
with (O/'per-image.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(O/'excluded-outputs.json').write_text(json.dumps(exclusions,indent=2)+'\n')
for file in [R/'panel.json',R/'result.json',Path(__file__),*Path('src/vis').glob('*.py')]: bindings[str(file.resolve())]=hashlib.sha256(file.read_bytes()).hexdigest()
(O/'receipt.json').write_text(json.dumps({'sources':bindings,'counts':{'fresh':128,'diagnostic':4},'ordered_pngs':[v['png'] for v in ledger],'selected_row_ids':[v['row_id'] for v in ledger],'matching':'shared renderer class-aware IoU .5; separate original study class-agnostic ledger','prediction_surface':'all native parsed valid predictions, duplicates preserved; rejected raw outputs in excluded-outputs.json','no_model_calls':True},indent=2)+'\n')
print(json.dumps(summary,indent=2))
