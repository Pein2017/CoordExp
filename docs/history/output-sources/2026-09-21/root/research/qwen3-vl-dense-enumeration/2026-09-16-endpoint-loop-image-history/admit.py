import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=R.parent/'2026-09-16-corner-loop-bridge-factorial'
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
p=read(R/'panel.json');rec=read(R/'runtime/I00/receipt.json');raw=read(R/'runtime/I00/raw.json');old=read(OLD/'runtime/C00/raw.json');assert raw['rows']==old['rows'];assert rec['status']=='candidate_complete' and rec['full_original_batch_identity']
for cell,key in [('C00','I00'),('C10','D10')]:
 ref=read(OLD/f'runtime/{cell}/receipt.json');rr=read(OLD/f'runtime/{cell}/raw.json')
 for field in ['loaded_identity','batch_shape','generate_settings','prompt_positions_sha256','positions']:assert rec[field]==ref[field],field
 assert p['cells'][key]['history_ids']==rr['rows'][1]['token_ids'][:1233]
 assert all(x['supplied_token'] is None for x in rec['seam_logits'] if x['action_offset']>=1233)
assert rec['counts']['model_forwards']==3084
(R/'admission.json').write_text(json.dumps(dict(status='passed',I00_exact_all_four_sequences=True,original_C10_reuse='Same complete literal prefix/model/runtime; processor supplies only exact prefix and leaves every free/companion logit native. Identity path admitted on I00; no C10 rerun.',receipt=bind(R/'runtime/I00/receipt.json'),raw=bind(R/'runtime/I00/raw.json'),source_diff=bind(R/'producer.diff')),indent=2)+'\n');print('Admission passed: I00 all4 sequences exact; C10 literal/runtime equivalence valid')
