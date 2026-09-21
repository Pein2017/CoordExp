import json,re
from pathlib import Path
from src.vis import render_prediction_comparison
R=Path(__file__).resolve().parent
d=json.loads((R/'stage1-reduction.json').read_text());p=json.loads((R/'stage1/panels/309264-SS.json').read_text());case=p['cases'][0]['group']['cases'][1]
gt=[dict(description=o['desc'],bbox=[int(re.fullmatch(r'<\|coord_(\d+)\|>',s)[1]) for s in o['bbox_2d']],coco_ann_id=o['coco_ann_id']) for o in case['input_record']['objects']]
for name in ['SS','SX_FY']:
 out=R/'review'/name;out.mkdir(parents=True,exist_ok=True)
 preds=[dict(description=x['description'],bbox=x['bbox_pixel_xyxy'],coord_bins=x['coord_bins_1000'],score=1.) for x in d['cells'][name]['free']['valid_predictions']]
 row=dict(row_id=case['row_id'],row_index=case['row_index'],image_path=case['image_path'],image_width=case['image_width'],image_height=case['image_height'],gt=gt,pred=preds)
 for file in ['gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl']:(out/file).write_text(json.dumps(row)+'\n')
z=render_prediction_comparison(R/'review/SS',R/'review/SX_FY',R/'review/stage1-comparison',left_label='SS free suffix',right_label='Sx/Fy free suffix',limit=1)
print(z.manifest_path);print(*z.image_paths,sep='\n')
