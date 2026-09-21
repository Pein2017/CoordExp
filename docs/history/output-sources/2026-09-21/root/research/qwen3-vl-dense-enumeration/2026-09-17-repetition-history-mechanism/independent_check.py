"""Saved-output contract falsification and deterministic consumer replay."""
import json,re,hashlib
from pathlib import Path
from probes.training_set_completion.repetition_history_reduce import consume
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
first=read(R/'reduction.json');second=consume(R);assert first==second
manifest=read(R/'runtime-manifest.json');assert len([c for c in manifest['cells'] if c['stage']=='C'])==24
rows=[]
pattern=re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|><\|coord_(\d+)\|><\|coord_(\d+)\|><\|coord_(\d+)\|><\|coord_(\d+)\|><\|box_end\|>',re.S)
for key,c in first['cells'].items():
 raw=read(Path(c['raw']['path']));panel=read(Path(c['entry']['panel']['path']));t=panel['cases'][0]['target_position'];row=raw['rows'][t];parsed=pattern.findall(row['text']);assert len(parsed)==c['full']['burden']['complete_rows']
 invalid=sum(not(int(x[1])<int(x[3]) and int(x[2])<int(x[4])) for x in parsed);assert invalid==c['full']['burden']['invalid']
 assert bool(row['token_ids'][-1]==151645)==bool(c['full']['burden']['eos'])
 if c['entry']['stage']=='C':
  pulse=raw['sampling_pulse'];assert c['release']==pulse['last_active_offset']+1
  assert len(pulse['steps'])<=32 and row['token_ids'][:54]==panel['cases'][0]['target_prefix_token_ids']
  own=set(c['intervention']['matches']['covered_owner_ids'])|set(c['crossing_supplied_owner_ids'])
  assert not(own&set(c['per_trajectory_known_accounting']['autonomous_new_relative_prefix']))
  assert not(set(c['known_accounting']['excluded_union'])&set(c['known_accounting']['free']))
 rows.append(dict(cell=key,complete_rows=len(parsed),invalid=invalid))
cs=[c for k,c in first['cells'].items() if k.startswith('C/')];assert len(cs)==24
union=set().union(*(set(c['per_trajectory_known_accounting']['supplied_owner_ids']) for c in cs))
assert all(set(c['known_accounting']['excluded_union'])==union for c in cs)
# Teeth: a deliberately supplied identity cannot receive new-owner credit.
fixture={'supplied','prefix','new'};excluded={'supplied'};prefix={'prefix'};assert (fixture-excluded-prefix)=={'new'};assert (fixture-prefix)!={'new'}
out=dict(status='passed',json_exact_consumer_replay=True,cells=len(rows),checks=rows,sampling_union=sorted(union),credit_exclusion_falsification=True,consumer=bind(Path('probes/training_set_completion/repetition_history_reduce.py').resolve()),verifier=bind(Path(__file__)),reduction=bind(R/'reduction.json'))
(R/'independent-reduction-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status='passed',cells=len(rows),sampling_union=sorted(union))))
