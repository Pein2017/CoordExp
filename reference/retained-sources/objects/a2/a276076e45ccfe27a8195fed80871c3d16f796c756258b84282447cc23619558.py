"""Literal-pixel review cards, consuming only the completed frozen evaluation."""
from pathlib import Path
import math
from PIL import Image, ImageDraw
from probes.native_owner_scale import evaluation as e
from src.vis.rendering import _font, _load_image, _save

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/evaluation')
CONSUMER = ROOT / 'candidate-natural-v11-consumer'
OUT = ROOT / 'candidate-review-cards-v11'


def panel(source, proposals, title, targets=()):
    image = source.copy()
    draw = ImageDraw.Draw(image)
    font = _font('regular', 14)
    for index, proposal in enumerate(proposals):
        box = tuple(float(v) for v in proposal['bbox'])
        assert len(box) == 4 and all(math.isfinite(v) for v in box)
        assert box[0] <= box[2] and box[1] <= box[3]
        # Color and ordinal contain no arm or source prediction chronology.
        color = ((index * 79 + 60) % 200, (index * 131 + 40) % 200, (index * 47 + 70) % 200)
        draw.rectangle(box, outline=color, width=3)
        label = str(index + 1)
        x, y = box[:2]
        draw.rectangle((x, y, x + 28, y + 19), fill='white')
        draw.text((x + 1, y), label, fill=color, font=font)
    for target in targets:
        box = target['target']['target_bbox_pixels']
        draw.rectangle(box, outline=(0, 255, 255), width=5)
        draw.text((box[0], max(0, box[1] - 20)), f"T{target['ordinal']}", fill='cyan', stroke_width=1, stroke_fill='black', font=font)
    return image


def card(path, source, left, right, labels, legend):
    width = source.width * 2 + 20
    font = _font('regular', 15)
    cols = max(1, width // 450)
    rows = math.ceil(len(legend) / cols)
    image = Image.new('RGB', (width, source.height + 65 + rows * 21), 'white')
    image.paste(left, (0, 45)); image.paste(right, (source.width + 20, 45))
    draw = ImageDraw.Draw(image)
    draw.text((5, 10), labels[0], fill='black', font=font)
    draw.text((source.width + 25, 10), labels[1], fill='black', font=font)
    for i, line in enumerate(legend):
        draw.text(((i // max(1, rows)) * (width // cols) + 5, source.height + 55 + (i % max(1, rows)) * 21), line, fill='black', font=font)
    _save(image, path)
    with Image.open(path) as check:
        assert check.size == image.size
        check.verify()
    return {**e.binding(path), 'canvas_size': list(image.size)}


def main():
    assert not OUT.exists()
    OUT.mkdir()
    queue = e.read_jsonl(CONSUMER / 'blind-review-queue.jsonl')
    assert len(queue) == 32
    blind = []
    for item in queue:
        assert item['source_blind'] and all(not any('arm' in key or 'source' in key for key in p) for p in item['proposals'])
        source = _load_image(Path(item['image_path']))
        assert source.size == (item['image_width'], item['image_height'])
        path = OUT / 'blind32' / f"{item['image_id']:012d}.png"
        legend = [f"{i+1}: {p['description']} [{p['proposal_id'][:10]}]" for i, p in enumerate(item['proposals'])]
        receipt = card(path, source, source, panel(source, item['proposals'], ''), (f"Image {item['image_id']} / original", 'Mixed proposals / source blind'), legend)
        blind.append({'image_id': item['image_id'], 'review_id': item['review_id'], 'proposal_ids': [p['proposal_id'] for p in item['proposals']], 'image_path': item['image_path'], 'card': receipt})
    e.publish(OUT / 'blind32-manifest.json', {'status': 'rendered_not_reviewed', 'source_blind': True, 'queue': e.binding(CONSUMER / 'blind-review-queue.jsonl'), 'items': blind, 'boundary': 'Mixed proposal coverage only, not exhaustive recall. No source map is embedded.'})
    packet = e.read(ROOT / 'candidate-panel-bound-v11.json')
    targets = e.read(packet['trusted_target_ledger']['path'])['targets']
    stable = {r['image_id']: r for r in e.read(CONSUMER / 'stable-consumer.json')}
    scaled = {r['image_id']: r for r in e.read(CONSUMER / 'scaled-terminal-consumer.json')}
    pairs = []
    for image_id in sorted({t['image_id'] for t in targets}):
        selected = [t for t in targets if t['image_id'] == image_id]
        source = _load_image(Path(selected[0]['image']['image_path']))
        a, b = stable[image_id]['parsed']['pred'], scaled[image_id]['parsed']['pred']
        legend = [f"L{i+1}: {p['description']}" for i, p in enumerate(a)] + [f"R{i+1}: {p['description']}" for i, p in enumerate(b)]
        legend += [f"T{t['ordinal']}: reviewed c / {t['target']['scoring_object']['description']}" for t in selected]
        path = OUT / 'admitted11' / f'{image_id:012d}.png'
        receipt = card(path, source, panel(source, a, '', selected), panel(source, b, '', selected), ('Stable50 / cyan reviewed c targets', 'Scaled terminal / cyan reviewed c targets'), legend)
        pairs.append({'image_id': image_id, 'target_case_ids': [t['case_id'] for t in selected], 'card': receipt})
    assert len(pairs) == 11 and sum(len(p['target_case_ids']) for p in pairs) == 16
    e.publish(OUT / 'admitted11-manifest.json', {'status': 'rendered_not_reviewed', 'source_blind': False, 'ledger': packet['trusted_target_ledger'], 'items': pairs, 'boundary': 'Literal natural predictions and reviewed c overlays; no extra inference or GT edits.'})
    print('Rendered 32 source-blind mixed-proposal cards and 11 admitted-image pairs / 16 targets.')


if __name__ == '__main__':
    main()
