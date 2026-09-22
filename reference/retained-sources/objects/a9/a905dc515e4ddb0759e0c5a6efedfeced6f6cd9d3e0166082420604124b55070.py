"""Current-round reporting from saved update receipts only; no model forward."""
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
BASE = OUT.parent.parent
rows = []
sources = []
for step in range(1, 257):
    root = 'third-fit-v1' if step <= 64 else 'fourth-fit-v1'
    p = BASE/root/'training/updates'/f'step-{step:05d}.json'
    d = json.loads(p.read_text())
    assert d['step'] == step and d['image_count'] == len(d['routes']) == 11
    row = {'step': step, 'objective_mean_over_images': d['objective_mean_over_images'],
           'ce_mean_over_images': statistics.mean(r['ce'] for r in d['routes']),
           'raw_axis_validity_hinge_mean_over_images': statistics.mean(r['raw_axis_validity_hinge'] for r in d['routes']),
           'gradient_norm_before_clip': d['gradient_norm_before_clip']}
    assert abs(row['objective_mean_over_images'] - row['ce_mean_over_images'] - .01*row['raw_axis_validity_hinge_mean_over_images']) < 1e-6
    rows.append(row)
    sources.append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
last = rows[-32:]
summary = {'schema': 'fourth_fit_saved_training_curve.v1',
           'definition': 'Equal mean of 11 per-image active-token CE means plus 0.01 times raw-axis expected-coordinate validity hinge; fixed corrected synthetic teacher routes.',
           'selected_steps': [r for r in rows if r['step'] in (1, 16, 32, 64, 128, 192, 256)],
           'last32': {'first': last[0]['objective_mean_over_images'], 'last': last[-1]['objective_mean_over_images'],
                      'min': min(r['objective_mean_over_images'] for r in last),
                      'max': max(r['objective_mean_over_images'] for r in last),
                      'mean': statistics.mean(r['objective_mean_over_images'] for r in last),
                      'positive_successive_changes': sum(b['objective_mean_over_images'] > a['objective_mean_over_images'] for a,b in zip(last,last[1:]))},
           'scope': 'Saved training fit only; no convergence, native owner coverage, generalization or causal mechanism claim.',
           'sources': sources}
(OUT/'training-curve-rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(OUT/'training-curve-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(8,4.2))
ax.plot([r['step'] for r in rows], [r['objective_mean_over_images'] for r in rows], label='Fixed-teacher training objective')
ax.axvline(64, color='grey', linestyle='--', label='Exact continuation at step 64')
ax.set(xlabel='Cumulative optimizer update', ylabel='Mean objective over 11 images',
       title='Training fit improves; native completeness requires separate review')
ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'training-curve.png',dpi=180)
print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))
