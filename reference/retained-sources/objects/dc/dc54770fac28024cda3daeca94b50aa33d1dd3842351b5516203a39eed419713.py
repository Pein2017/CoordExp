"""One-time, image-disjoint reference cohort. No model scores enter selection."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '2026-09-09-fp-visual-distribution'
OUT = ROOT / 'holdout-v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    (OUT / 'reference-crops').mkdir()
    dev = json.loads((SOURCE / 'final-v2/reviewed-cases.json').read_text())
    excluded = {r['source_sample']['image_id'] for r in dev}
    inventory = [json.loads(line) for line in (SOURCE / 'inventory.jsonl').read_text().splitlines()]
    eligible = [r for r in inventory if r['image_id'] not in excluded and r['stratum'] != 'strict_repeat']
    # Uniform image sample, then a uniform real unmatched candidate within image.
    # Avoids outcome-selected quotas and correlated same-image reference labels.
    by_image = {}
    for r in eligible:
        by_image.setdefault(r['image_id'], []).append(r)
    rng = random.Random(2026091001)
    ids = rng.sample(sorted(by_image), 64)
    selected = [rng.choice(sorted(by_image[i], key=lambda r: r['case_id'])) for i in ids]
    assert len({r['image_id'] for r in selected}) == 64
    assert not ({r['image_id'] for r in selected} & excluded)
    public = []
    for i, r in enumerate(selected):
        im = Image.open(r['image_path']).convert('RGB')
        assert im.size == (r['width'], r['height'])
        x1, y1, x2, y2 = r['bbox']
        pad = max(32, .35 * max(x2-x1, y2-y1))
        win = [max(0, int(x1-pad)), max(0, int(y1-pad)), min(im.width, int(x2+pad)+1), min(im.height, int(y2+pad)+1)]
        crop = im.crop(win)
        scale = min(4, 640 / max(crop.size))
        crop = crop.resize((round(crop.width*scale), round(crop.height*scale)))
        canvas = Image.new('RGB', (crop.width*2, crop.height+38), 'white')
        canvas.paste(crop, (0, 38)); canvas.paste(crop, (crop.width, 38))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 8), f"{r['case_id']} | {r['category']} | raw left; target right", fill='black')
        draw.rectangle([crop.width+(x1-win[0])*scale, 38+(y1-win[1])*scale, crop.width+(x2-win[0])*scale, 38+(y2-win[1])*scale], outline='red', width=2)
        path = OUT / 'reference-crops' / (r['case_id'].replace(':', '_')+'.png')
        canvas.save(path)
        public.append({k:r[k] for k in ['case_id','image_id','image_path','bbox','category','width','height']} | {'image_sha256':sha(Path(r['image_path'])), 'crop_path':str(path)})
    for filename, rows in [('candidates.jsonl', public), ('reference-a.jsonl', public[:32]), ('reference-b.jsonl', public[32:]), ('selection-private.jsonl', selected)]:
        (OUT / filename).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    receipt = {'seed':2026091001, 'population_candidates':len(eligible), 'population_images':len(by_image), 'sample_images':64, 'sample_candidates':64, 'development_excluded_images':len(excluded), 'strata':dict(Counter(r['stratum'] for r in selected)), 'estimand':'image-balanced random real non-strict-repeat unmatched candidate, conditioned on images outside development set', 'inventory_sha256':sha(SOURCE/'inventory.jsonl'), 'development_sha256':sha(SOURCE/'final-v2/reviewed-cases.json'), 'candidates_sha256':sha(OUT/'candidates.jsonl'), 'code_sha256':sha(Path(__file__)), 'reference_status':'unlabeled; reviewers blinded to model predictions and GT'}
    (OUT/'selection-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
