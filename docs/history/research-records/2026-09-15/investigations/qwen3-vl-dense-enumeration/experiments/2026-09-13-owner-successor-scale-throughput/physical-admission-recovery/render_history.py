import json,pathlib,re,hashlib
from PIL import Image,ImageDraw,ImageFont
O=pathlib.Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/physical-admission-recovery')
d=json.load(open(O/'review-index-v1.json'));(O/'cards').mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
manifest=[]
for g in d['groups']:
 r=next(r for r in d['rows'] if r['job_id']==g['representative_card']['representative_job_id']);im=Image.open(r['image']['path']).convert('RGB');W,H=im.size
 out=Image.new('RGB',(W*3,H+80),'white');draw=ImageDraw.Draw(out)
 for col in range(3):out.paste(im,(col*W,50))
 draw.text((4,4),g['visual_group_id']+' '+r['job_id']+' original',font=font,fill='black')
 draw.text((W+4,4),'c cyan / immediate w lime / first owner orange',font=font,fill='black')
 draw.text((2*W+4,4),'Exact h proposals (not GT); labels = history ordinal',font=font,fill='black')
 for k,color in [('c','cyan'),('w','lime')]:
  b=r[k]['bbox_native_pixels_xyxy'];box=(b[0]+W,b[1]+50,b[2]+W,b[3]+50);draw.rectangle(box,outline=color,width=3);draw.text((box[0]+2,box[1]+2),k+': '+r[k]['description'],font=font,fill=color,stroke_width=1,stroke_fill='black')
 b=r['first_owner_axis']['bbox'];draw.rectangle((b[0]+W,b[1]+50,b[2]+W,b[3]+50),outline='orange',width=2)
 hist=[]
 pat=r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>((?:<\|coord_\d+\|>){4})<\|box_end\|>'
 for i,m in enumerate(re.finditer(pat,r['exact_history']['literal_text'])):
  z=list(map(int,re.findall(r'coord_(\d+)',m.group(2))));b=[int(z[0]*W/999),int(z[1]*H/999),int(z[2]*W/999),int(z[3]*H/999)];hist.append(dict(ordinal=i,description=m.group(1),bbox=b));box=(b[0]+2*W,b[1]+50,b[2]+2*W,b[3]+50)
  if b[2]>=b[0] and b[3]>=b[1]:
   draw.rectangle(box,outline='yellow',width=1);draw.text((box[0],box[1]),str(i),font=font,fill='yellow',stroke_width=1,stroke_fill='black')
 p=O/'cards'/f"{g['visual_group_id']}.png";out.save(p);manifest.append(dict(visual_group_id=g['visual_group_id'],job_id=r['job_id'],path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),history_projection=hist))
(O/'card-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(len(manifest))
