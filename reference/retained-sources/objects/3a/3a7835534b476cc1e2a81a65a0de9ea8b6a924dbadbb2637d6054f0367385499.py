"""Cards for cold-validated interrupted-shard outcomes, no physical judgments."""
from pathlib import Path
from PIL import Image,ImageDraw
from probes.owner_successor_scale import data as d
from src.vis.rendering import _font,_load_image,_save

root=d.ROOT.parent/'supply-recovery'
snapshot=d.e.read(root/'snapshot.json')
images={r['image_id']:r['frozen'] for r in snapshot['records']}
jobs={j['job_id']:j for j in snapshot['jobs']}
cards=[]
for shard in (0,1):
    rows_path=d.ROOT/f'remainder-v1/shard-{shard}/rows.jsonl'
    for result in d.e.read_jsonl(rows_path):
        job=jobs[result['job_id']];frozen=images[result['image_id']]
        source=_load_image(Path(frozen['case']['image_path']))
        canvas=Image.new('RGB',(source.width*2+20,source.height+65),'white')
        canvas.paste(source,(0,45));canvas.paste(source,(source.width+20,45))
        draw=ImageDraw.Draw(canvas);font=_font('regular',16)
        draw.text((5,8),job['job_id']+' / original',fill='black',font=font)
        draw.text((source.width+25,8),'cyan c; green w; orange first; red repeat',fill='black',font=font)
        boxes=[(job['c_bbox'],'cyan','c'),(job['first_owner']['bbox'],'orange','first'),(job['repeat_bbox'],'red','repeat')]
        if 'w_bbox_xyxy_pixels' in result['local_w']:boxes.append((result['local_w']['w_bbox_xyxy_pixels'],'green','w'))
        for box,color,label in boxes:
            a,b,c,z=box;xy=(a+source.width+20,b+45,c+source.width+20,z+45)
            draw.rectangle(xy,outline=color,width=4);draw.text(xy[:2],label,fill=color,stroke_width=1,stroke_fill='black',font=font)
        path=root/'preserved-cards'/f"{job['job_id'].replace(':','-')}.png"
        assert not path.exists();_save(canvas,path)
        cards.append({'job_id':job['job_id'],'image_id':job['image_id'],'card':d.e.binding(path),'source_rows':d.e.binding(rows_path),'local_w':result['local_w'],'c_bbox':job['c_bbox'],'first_owner':job['first_owner'],'repeat_bbox':job['repeat_bbox'],'physical_review_status':'pending'})
d.e.publish(root/'preserved-cards-manifest.json',{'status':'rendered_not_physically_reviewed','snapshot':d.e.binding(root/'snapshot.json'),'shards':[0,1],'cards':cards,'boundary':'Only durable cold-validated outcomes; no-w and ambiguous/group c remain inadmissible without independent physical support. Sealed2/3 reviewer-owned projections untouched.'})
print('preserved cards',len(cards))
