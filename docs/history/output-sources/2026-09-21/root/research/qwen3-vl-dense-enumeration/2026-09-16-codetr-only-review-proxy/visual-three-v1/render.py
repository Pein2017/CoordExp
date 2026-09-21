"""Display exact retained crops and source-pixel boxes; no inferred GT or model run."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageStat

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator')
OUT = Path(__file__).resolve().parent
IDS = ['237954:p11','307814:p0','134520:p0']
ORANGE = '#ff6600'
CYAN = '#00dce8'
YELLOW = '#ffe45c'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(p):
    return [json.loads(l) for l in p.read_text().splitlines()]


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def panel(canvas,im,x,y,w,h,boxes):
    scale=min(w/im.width,h/im.height)
    size=(round(im.width*scale),round(im.height*scale))
    px=x+(w-size[0])//2;py=y+(h-size[1])//2
    canvas.paste(im.resize(size,Image.Resampling.LANCZOS),(px,py))
    draw=ImageDraw.Draw(canvas)
    for box,color in boxes:
        mapped=[px+box[0]*size[0]/im.width,py+box[1]*size[1]/im.height,
                px+box[2]*size[0]/im.width,py+box[3]*size[1]/im.height]
        draw.rectangle(mapped,outline=color,width=4)


def main():
    candidate_path=ROOT/'holdout-v1/candidates.jsonl'
    decision_path=ROOT/'codetr-context-holdout-v1/decisions.json'
    prediction_path=ROOT/'codetr-context-holdout-v1/image-predictions.jsonl'
    config_path=ROOT/'codetr-context-holdout-v1/effective-config.json'
    candidates={r['case_id']:r for r in rows(candidate_path)}
    decisions={r['case_id']:r for r in json.loads(decision_path.read_text())}
    views={Path(r['image_path']).stem:r for r in rows(prediction_path)}
    manifest={'purpose':'User-requested three historical errors, source and exact retained detector context',
              'box_semantics':{'orange':'rollout prediction','cyan':'Co-DETR selected prediction','yellow':'detector context window'},
              'not_ground_truth':'No corrected GT box is drawn; historical visual labels are reference judgments.',
              'sources':[{'path':str(p),'sha256':sha(p)} for p in [candidate_path,decision_path,prediction_path,config_path]],'cases':[]}
    historical={'237954:p11':'too_tight','307814:p0':'too_loose','134520:p0':'too_tight'}
    for case in IDS:
        c=candidates[case];d=decisions[case];v=views[case.replace(':','_')]
        source=Image.open(c['image_path']).convert('RGB');crop=Image.open(v['image_path']).convert('RGB')
        assert sha(c['image_path'])==c['image_sha256']==v['source_image_sha256']
        assert sha(v['image_path'])==v['image_sha256']
        win=v['window']
        assert crop.size==(win[2]-win[0],win[3]-win[1])
        diff=ImageChops.difference(source.crop(win),crop)
        equality=source.crop(win).tobytes()==crop.tobytes()
        mean_difference=sum(ImageStat.Stat(diff).mean)/3
        def local(box):return [box[0]-win[0],box[1]-win[1],box[2]-win[0],box[3]-win[1]]
        canvas=Image.new('RGB',(1650,760),'#18202b');draw=ImageDraw.Draw(canvas)
        panel(canvas,source,20,100,410,570,[(win,YELLOW),(c['bbox'],ORANGE)])
        panel(canvas,crop,450,100,580,570,[(local(c['bbox']),ORANGE)])
        panel(canvas,crop,1050,100,580,570,[(local(d['selected']['bbox_xyxy']),CYAN)])
        draw=ImageDraw.Draw(canvas)
        draw.text((22,16),f"{case} | {c['category']} | agreement IoU {d['candidate_iou']:.4f} | detector score {d['selected']['score']:.4f}",font=font(25),fill='white')
        draw.text((22,57),'FULL SOURCE: yellow = detector crop',font=font(18),fill=YELLOW)
        draw.text((455,57),'EXACT CONTEXT + ROLLOUT BOX',font=font(20),fill=ORANGE)
        draw.text((1055,57),'SAME CONTEXT + Co-DETR BOX',font=font(20),fill=CYAN)
        draw.text((22,688),f"Retained crop {crop.width}x{crop.height}px | source window {win} | native Resize: keep_ratio, scale (2048,1280)",font=font(18),fill='white')
        draw.text((22,718),f"Historical reference: {historical[case]}. Neither colored box is GT. Display resize only; no regeneration.",font=font(18),fill='#c8d1df')
        dest=OUT/(case.replace(':','_')+'.png');canvas.save(dest)
        manifest['cases'].append({'case_id':case,'image_path':c['image_path'],'crop_path':v['image_path'],
                                  'crop_window':win,'crop_size':list(crop.size),'rollout_bbox':c['bbox'],
                                  'codetr_bbox':d['selected']['bbox_xyxy'],'category':c['category'],
                                  'agreement_iou':d['candidate_iou'],'historical_geometry_label':historical[case],
                                  'current_decode_pixels_equal_saved_crop':equality,
                                  'mean_channel_difference_current_decode_vs_saved_crop':mean_difference,
                                  'displayed_detector_input':'exact hash-verified saved PNG, never recropped',
                                  'png':str(dest),'png_sha256':sha(dest)})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'count':len(manifest['cases']),'pngs':[r['png'] for r in manifest['cases']]}))


if __name__=='__main__':main()
