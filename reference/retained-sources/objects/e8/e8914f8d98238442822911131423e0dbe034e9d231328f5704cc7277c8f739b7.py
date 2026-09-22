import copy,difflib,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
R=Path(__file__).resolve().parent;OLD=R.parent/'2026-09-16-corner-loop-bridge-factorial'
read=lambda p:json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
p=read(OLD/'panel.json');accept=read(OLD/'lead-acceptance.json');print('predecessor status',accept['status'])
assert accept['all_saved_row_reductions_equal'] and accept['credit_exclusion_falsification_passed']
for b in p['sources']:assert bind(Path(b['path']))==b
oldsource=(OLD/'producer.py').read_text();assert read(OLD/'runtime/C00/receipt.json')['producer']['sha256']==bind(OLD/'producer.py')['sha256']
new=oldsource.replace('Four frozen recent-row interventions through the original heterogeneous native batch.','One still-invalid y2 edit through the accepted original heterogeneous native batch.')
new=new.replace(str(OLD),str(R))
new=new.replace("ROOT/'runtime/C00/receipt.json'",f"Path('{OLD}/runtime/C00/receipt.json')")
new=new.replace("choices=['C00','C10','C01','C11']","choices=['Y998']")
assert new!=oldsource
(R/'producer.py').write_text(new)
(R/'producer.diff').write_text(''.join(difflib.unified_diff(oldsource.splitlines(True),new.splitlines(True),fromfile=str(OLD/'producer.py'),tofile=str(R/'producer.py'))))
tok=AutoTokenizer.from_pretrained(p['config']['model']['base_model'],local_files_only=True);native=p['cells']['C00']['tokens'];tokens=native.copy();token998=tok.convert_tokens_to_ids('<|coord_998|>');assert native==[151646,34196,151647,151648,151670,152669,152669,152669,151649];assert tok.encode('<|coord_998|>',add_special_tokens=False)==[token998];tokens[7]=token998
text=tok.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False);assert text=='<|object_ref_start|>chair<|object_ref_end|><|box_start|><|coord_0|><|coord_999|><|coord_999|><|coord_998|><|box_end|>'
assert [i for i,(a,b) in enumerate(zip(native,tokens)) if a!=b]==[7]
history=p['common_history_ids']+tokens;assert len(history)==1233
p['cells']['Y998']={'category':'chair','y1':999,'y2':998,'text':text,'tokens':tokens,'history_ids':history,'history_sha256':digest(history),'edits':[{'row_token_offset':7,'old':152669,'new':token998}]}
p.update(schema='corner_loop.one_bin_control.v1',active_cells=['Y998'],inherited_cells=['C00','C01','C10','C11'],ceilings={'cells':1,'model_forwards':5000,'intended_model_forwards':3084,'free_tokens':1851,'intended_gpu_seconds':1800},interpretation='Numerical one-bin change need not be small in embedding/probability space. Zero-height -> inverted-height is still invalid, not identical semantics/norm matching. Success shows only this edit sufficient; failure shows only its insufficiency.')
p['controls']={c:{'receipt':bind(OLD/f'runtime/{c}/receipt.json'),'raw':bind(OLD/f'runtime/{c}/raw.json')} for c in p['inherited_cells']}
p['sources'] +=[bind(OLD/f) for f in ['panel.json','producer.py','reduce.py','result.json','terminal.json','lead-acceptance.json']]+[b for controls in p['controls'].values() for b in controls.values()]
p['derivative']={'predecessor':bind(OLD/'producer.py'),'executed':bind(R/'producer.py'),'diff':bind(R/'producer.diff'),'changes':'output root, bound external C00 gate, single Y998 CLI choice, description only; native generation and scoring unchanged'}
(R/'panel.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');print('token998',token998,'single delta',p['cells']['Y998']['edits'],'same bank/config',p['owner_bank_sha256'],p['config_sha256'])
