# Reproduce the bounded parser diagnosis

Run this read-only block from `/data/CoordExp/.worktrees/research-probes` with `PYTHONDONTWRITEBYTECODE=1`.

```python
import json
from pathlib import Path
from collections import Counter
from probes.dora_owner_learning.entrance_ce_eval import aggregate_scores
from probes.dora_owner_learning.candidate_opportunity import score, digest
from probes.source_rweak_row_cross.run import native_record

p=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48-strong100/evaluation')
m=json.loads((p/'manifest.json').read_text())
c=json.loads((p/'consumer.json').read_text())
reduction=json.loads((p/'reduction.json').read_text())
f={r['example_id']:r for r in m['records']}
manifest_hash=digest(m)
assert len(c)==130 and {r['example_id'] for r in c}==set(f)
assert all(r['manifest_sha256']==manifest_hash for r in c)
dev=[r for r in c if f[r['example_id']]['split']!='train']
assert len(dev)==128

groups={
    'all128':dev,
    'guard16':[r for r in dev if f[r['example_id']]['split']=='guard'],
    'equal_drop_count':[r for r in dev if f[r['example_id']]['baseline_score']['parser_drops']==r['score']['parser_drops']],
    'paired_drop_free':[r for r in dev if f[r['example_id']]['baseline_score']['parser_drops']==r['score']['parser_drops']==0],
    'drops_increased':[r for r in dev if f[r['example_id']]['baseline_score']['parser_drops']<r['score']['parser_drops']],
    'drops_decreased':[r for r in dev if f[r['example_id']]['baseline_score']['parser_drops']>r['score']['parser_drops']],
}
for name,selected in groups.items():
    a=aggregate_scores([f[r['example_id']]['baseline_score'] for r in selected])
    b=aggregate_scores([r['score'] for r in selected])
    if name=='all128':
        assert a==reduction['splits']['dev128_descriptive']['source']
        assert b==reduction['splits']['dev128_descriptive']['strong100']
    print(name,len(selected),a['50'],b['50'],'drops',a['parser_drops'],b['parser_drops'],
          'repeats',a['strict_repeats'],b['strict_repeats'],
          'drop_burden_sensitivity',*[2*s['50']['tp']/(2*s['50']['tp']+s['50']['fp']+s['50']['fn']+s['parser_drops']) for s in (a,b)])

for r in dev:
    old=f[r['example_id']];a=old['baseline_score'];b=r['score']
    if a['parser_drops']!=b['parser_drops']:
        print('changed',old['image_id'],old['split'],'drops',a['parser_drops'],b['parser_drops'],
              'TP_delta',b['50']['tp']-a['50']['tp'],'FP_delta',b['50']['fp']-a['50']['fp'],
              'valid_delta',b['prediction_count']-a['prediction_count'],
              'repeat_delta',b['strict_repeats']-a['strict_repeats'],
              'token_delta',b['complete_token_length']-a['complete_token_length'])

for label in ('source','candidate'):
    print(label,'drop_reasons',Counter(d['reason'] for r in dev for d in
          (f[r['example_id']]['baseline'] if label=='source' else r['parsed'])['dropped_predictions']))

for iid in (59571,39654,70033):
    r=next(r for r in dev if f[r['example_id']]['image_id']==iid);old=f[r['example_id']]
    pairs=[('source',old['baseline'],old['baseline_ids'],old['baseline_score']),
           ('candidate',r['parsed'],r['action_ids'],r['score'])]
    for label,parsed,ids,saved in pairs:
        text=parsed['raw_decode_text'];stop=parsed['decode_stop_reason']
        assert native_record(text,old['case'],old['baseline'],stop)==parsed
        assert score(parsed,seed=-1,length=len(ids),stop=stop)==saved
        print(iid,label,'drop_positions',[d['generated_order'] for d in parsed['dropped_predictions']],
              'drop_boxes',[[v['text'] for v in d['coord_token_spans']] for d in parsed['dropped_predictions']])
    print(iid,'owner_changes50',r['owner_changes']['50'],'reassignment',r['matching_reassignment']['50'])
```
