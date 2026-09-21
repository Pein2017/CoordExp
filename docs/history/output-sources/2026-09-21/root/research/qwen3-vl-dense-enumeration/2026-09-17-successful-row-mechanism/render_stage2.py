import json
from pathlib import Path
from src.vis import render_prediction_comparison
R=Path(__file__).resolve().parent
d=json.loads((R/'stage2-reduction.json').read_text());base=json.loads((R/'review/SS/gt_vs_pred.jsonl').read_text())
for name in ['native-S','head-S-to-F']:
 out=R/'review'/name;out.mkdir(parents=True,exist_ok=True)
 pred=[dict(description=x['description'],bbox=x['bbox_pixel_xyxy'],coord_bins=x['coord_bins_1000'],score=1.) for x in d['cells'][name]['free']['valid_predictions']]
 row=dict(base,pred=pred)
 for file in ['gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl']:(out/file).write_text(json.dumps(row)+'\n')
z=render_prediction_comparison(R/'review/native-S',R/'review/head-S-to-F',R/'review/stage2-comparison',left_label='Native S, after row8',right_label='Head S-to-F rescue, after row8',limit=1)
print(*z.image_paths,sep='\n')
