"""Internal GT-free scene-and-crop preparation for the small judge profile."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def prepare_requests(candidates, output_dir, profile):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/'images').mkdir()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    seen=set();requests=[]
    def panel(image,box):
        scale=min(640/image.width,640/image.height)
        scaled=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))))
        draw=ImageDraw.Draw(scaled)
        draw.rectangle(tuple(v*scale for v in box),outline=(255,20,20),width=3)
        return ImageOps.pad(scaled,(640,640),color='white')
    for i,candidate in enumerate(candidates):
        key=candidate['case_id']
        if key in seen: raise ValueError(f'Duplicate candidate: {key}')
        seen.add(key)
        source=Path(candidate['image_path'])
        with Image.open(source) as image: image=image.convert('RGB')
        box=tuple(map(float,candidate['bbox']))
        x1,y1,x2,y2=box
        if not(0<=x1<x2<=image.width and 0<=y1<y2<=image.height):
            raise ValueError(f'Invalid pixel box: {key}')
        cx,cy=(x1+x2)/2,(y1+y2)/2
        hx,hy=max(80,(x2-x1)*.85),max(80,(y2-y1)*.85)
        region=(max(0,int(cx-hx)),max(0,int(cy-hy)),min(image.width,int(cx+hx)+1),min(image.height,int(cy+hy)+1))
        crop=image.crop(region)
        local=(x1-region[0],y1-region[1],x2-region[0],y2-region[1])
        canvas=Image.new('RGB',(1280,704),'white')
        draw=ImageDraw.Draw(canvas)
        draw.text((12,8),'SCENE CONTEXT | red = candidate',font=font,fill='black')
        draw.text((652,8),'ZOOM OF SAME REGION | same candidate',font=font,fill='black')
        draw.text((12,36),f"Predicted category: {candidate['category']}",font=font,fill='black')
        canvas.paste(panel(image,box),(0,64))
        canvas.paste(panel(crop,local),(640,64))
        path=output_dir/'images'/f'{i:03d}.png'
        canvas.save(path)
        requests.append({'case_id':key,'image_path':str(path.resolve()),
            'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'system_prompt':profile['system_prompt'],
            'user_prompt':profile['user_prompt_template'].replace('{category}',candidate['category'])})
    with (output_dir/'requests.jsonl').open('x') as stream:
        for row in requests: stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    return output_dir/'requests.jsonl'
