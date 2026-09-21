from pathlib import Path
import json
from PIL import Image,ImageDraw
R=Path(__file__).parent;d=json.loads((R/'reduction.json').read_text());a=json.loads((R/'annotation-identity.json').read_text());im=Image.open(a['image']['resolved_path']).convert('RGB');out=R/'physical';out.mkdir(exist_ok=True)
unique={}
for p in d['paths']:
 if p['box'] is not None:unique.setdefault(tuple(p['box']),[]).append(p['id'])
assert len(unique)<=15
index=[];thumbs=[]
for i,(box,ids) in enumerate(unique.items()):
 canvas=im.copy();draw=ImageDraw.Draw(canvas);x1,y1,x2,y2=box;px=[x1*im.width/999,y1*im.height/999,x2*im.width/999,y2*im.height/999];draw.rectangle(px,outline='red',width=3);draw.text((8,8),f'{ids} {box}',fill='red',stroke_width=1);path=out/f'row-{i}.png';canvas.save(path)
 crop=canvas.crop((0,170,300,360)).resize((900,570));crop.save(out/f'row-{i}-context.png');thumbs.append(crop);index.append(dict(index=i,box=box,paths=ids,full=str(path),context=str(out/f'row-{i}-context.png')))
 grid=Image.new('RGB',(900*3,600*((len(thumbs)+2)//3)),'white')
for i,t in enumerate(thumbs):grid.paste(t,((i%3)*900,(i//3)*600));ImageDraw.Draw(grid).text(((i%3)*900,(i//3)*600+575),f'row-{i}: {index[i]["box"]}',fill='black')
grid.save(out/'contact.png');(out/'index.json').write_text(json.dumps(index,indent=2));print(len(unique))
