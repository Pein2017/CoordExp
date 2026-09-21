import json,hashlib
from pathlib import Path
from probes.training_set_completion.numerical_feedback.select import rows
base=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration');a=base/'2026-09-19-recurrence-distribution-census';c=base/'2026-09-19-recurrence-conditional-mass'
panel=json.loads((a/'shared-panel.json').read_text());states={b['id']:b for b in panel['all_boundaries']};images={};n=0
for line in (c/'draws.jsonl').open():
 d=json.loads(line); b=states[d['state_id']]; con=d['conditioning']; prefix=b['native_tokens'][:b['source_row']['end']]
 assert prefix==con['prefix_token_ids']; assert con['prefix_token_ids_sha256']==b['source_prefix_hash']
 expected=[151646,*b['next_row']['description_tokens'],151647,151648];assert expected==con['row_prefix_token_ids']
 s=d['source_binding'];p=Path(s['image_path']);h=images.setdefault(str(p),hashlib.sha256(p.read_bytes()).hexdigest());assert h==s['image_sha256']
 assert b['condition'].endswith('-original')
 assert d['source_policy'] is None or d['source_policy']=='original' or d['source_policy'].endswith('-original')
 assert d['horizon']==5 and d['draw_count']==256
 n+=1
assert n==45
out={'status':'pass','states':n,'image_hashes':len(images),'exact_native_prefix_and_next_row_description':True,'conditioning_limit':'next-row description differs from preceding source row in seven prospective proxies; their repeat union is empty'}
(a/'integration/mass-conditioning-check.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
