import json,hashlib
from pathlib import Path
from PIL import Image
root=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/corrected-pilot-v2/inputs/manifests')
seen=set();count=0;ys=0
for p in root.glob('*.json'):
 d=json.loads(p.read_text());g=d['geometry'];s=d['source']['source_image'];path=Path(s['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==s['sha256'];im=Image.open(path).convert('RGB');assert im.size==(g['source_width'],g['source_height'])
 for k,c in d['cells'].items():
  count+=1;ip=Path(c['image_path']); key=(str(ip),c['visual_offset_px'])
  if key not in seen:
   canvas=Image.open(ip).convert('RGB');assert canvas.size==(g['canvas_width'],g['source_height']);expected=Image.new('RGB',canvas.size,tuple(g['fill_rgb']));expected.paste(im,(c['visual_offset_px'],0));assert canvas.tobytes()==expected.tobytes();seen.add(key)
  for box in c['history_boxes']:
   source=box['source_bins'];mapped=box['mapped_bins'];assert mapped[1]==source[1] and mapped[3]==source[3];ys+=2
   for i in [0,2]:
    expected=round((source[i]*(g['source_width']-1)/999+c['history_offset_px'])*999/(g['canvas_width']-1));assert expected==mapped[i]
assert count==315
out={'status':'pass','states':45,'cells':count,'exact_pixel_images_checked':len(seen),'y_roles_checked':ys,'independent_horizontal_formula':True,'gpu_calls':0}
(Path(__file__).parent/'spatial-input-check.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
