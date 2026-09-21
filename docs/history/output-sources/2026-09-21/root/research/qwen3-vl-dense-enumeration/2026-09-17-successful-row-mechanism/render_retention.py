import json
from pathlib import Path
from src.vis import render_prediction_comparison
R=Path(__file__).resolve().parent
d=json.loads((R/'stage1-reduction.json').read_text());base=json.loads((R/'review/SS/gt_vs_pred.jsonl').read_text())
for name in ['FF','FX_SY']:
 out=R/'review'/f'{name}-first';out.mkdir(parents=True,exist_ok=True)
 x=d['cells'][name]['free']['valid_predictions'][0];row=dict(base,pred=[dict(description=x['description'],bbox=x['bbox_pixel_xyxy'],coord_bins=x['coord_bins_1000'],score=1.)])
 for file in ['gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl']:(out/file).write_text(json.dumps(row)+'\n')
z=render_prediction_comparison(R/'review/FF-first',R/'review/FX_SY-first',R/'review/retention-context',left_label='FF first free row only',right_label='Fx/Sy first free row only',limit=1)
print(*z.image_paths,sep='\n')
