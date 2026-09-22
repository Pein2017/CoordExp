import json,re,hashlib
from pathlib import Path
from transformers import AutoTokenizer
from probes.training_set_completion import source256_evaluation as ev
R=Path(__file__).resolve().parent;OLD=R.parent/'2026-09-16-source256-output-ranking-repair';PHASE=R.parent/'2026-09-16-corner-loop-mechanism'
read=lambda p:json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
phase=read(PHASE/'panel.json');config=phase['config'];config['adapter']['path']=phase['checkpoints']['R16'];tok=AutoTokenizer.from_pretrained(config['model']['base_model'],local_files_only=True)
sources=[bind(PHASE/'lead-acceptance.json'),bind(PHASE/'panel.json')];paths=[OLD/'visuals/human-review-pr16-v1/477415-R-source.json',OLD/'visuals/human-review-pr16-v1/477415-P-source.json',OLD/'runtime/main-v1/R/readback/train/shard-05.json',OLD/'preparation/prepared.json'];sources += [bind(p) for p in paths]
rawR,rawP,shard,prepared=map(read,paths);group=[r for r in shard['generation']['rows'] if r['batch_index']==6];assert [r['image_id'] for r in group]==[460038,477415,496747,511251]
raw=rawR['saved_generation'];assert group[1]==raw
pattern=re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>'+r'<\|coord_(\d+)\|>'*4+r'<\|box_end\|>')
spans=list(pattern.finditer(raw['raw_decode_text']));prefix=tok.encode(raw['raw_decode_text'][:spans[136].start()],add_special_tokens=False);assert len(prefix)==1224 and prefix==raw['generated_token_ids'][:1224]
assert tok.encode(spans[136].group(),add_special_tokens=False)==[151646,34196,151647,151648,151670,152669,152669,152669,151649]
pbridge=list(pattern.finditer(rawP['saved_generation']['raw_decode_text']))[136].group();assert 'person' in pbridge
cells={}
for name,category,y1 in [('C00','chair',999),('C10','person',999),('C01','chair',0),('C11','person',0)]:
 text=f'<|object_ref_start|>{category}<|object_ref_end|><|box_start|><|coord_0|><|coord_{y1}|><|coord_999|><|coord_999|><|box_end|>';tokens=tok.encode(text,add_special_tokens=False);assert len(tokens)==9
 cells[name]={'category':category,'y1':y1,'text':text,'tokens':tokens,'history_ids':prefix+tokens,'history_sha256':digest(prefix+tokens),'edits':[{'row_token_offset':j,'old':a,'new':b} for j,(a,b) in enumerate(zip(raw['generated_token_ids'][1224:1233],tokens)) if a!=b]}
assert cells['C11']['text']==pbridge;assert {k:[e['row_token_offset'] for e in v['edits']] for k,v in cells.items()}=={'C00':[],'C10':[1],'C01':[5],'C11':[1,5]}
prepbind=prepared['sources']['preparation'];assert bind(Path(prepbind['path']))==prepbind;bank=read(Path(prepbind['path']));rec=next(r for r in bank['bank']['records'] if r['image_id']==477415);targets=[ev._target(image_id=477415,owner_id=o['owner_id'],description=o['description'],coord_bins=o['coord_bins']) for o in rec['owners']];assert len(targets)==27
sources.append(prepbind)
for p in Path(config['adapter']['path']).iterdir():
 if p.is_file():sources.append(bind(p))
for p in Path(config['embedding_delta']['path']).iterdir():
 if p.is_file():sources.append(bind(p))
cases=[prepared['canonical_routes'][str(g['image_id'])]['case'] for g in group]
context={k:cases[1][k] for k in ['row_id','row_index','image_width','image_height']}
panel={'schema':'corner_loop.bridge_factorial.v1','status':'frozen','config':config,'config_sha256':digest(config),'sources':sources,'batch_group':group,'cases':cases,'target_batch_position':1,'common_history_ids':prefix,'common_history_sha256':digest(prefix),'cells':cells,'owner_targets':targets,'owner_bank_sha256':digest(targets),'parse_context':context,'original_action_cap':3084,'supplied_tokens':1233,'free_budget':1851,'eos_id':151645,'runtime':'exact original heterogeneous bs4, native generate_continuations, FP32/SDPA, greedy RP1; generation processor verifies common history and fixes ONLY completed row137; never forces the next opener','ceilings':{'cells':4,'model_forwards':20000,'per_cell_model_forwards':5000,'intended_model_forwards':12336,'free_tokens_per_target':1851,'intended_gpu_seconds_per_cell':1800},'credit':'exclude supplied137 from matching; common136 + free suffix one-to-one; separate free-only and history assignment swaps; supplied137 may be a duplicate history reference'}
(R/'panel.json').write_text(json.dumps(panel,indent=2,sort_keys=True)+'\n');print('frozen',digest(panel),'rows',len(prefix)//9,'targets',len(targets))
