"""Run child aggregate without mutating candidate files, compare JSON exactly."""
import contextlib,io,json,runpy
from pathlib import Path
base=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/broad-v1')
out=Path(__file__).parent/'spatial-aggregate-recomputed';out.mkdir(exist_ok=True)
write=Path.write_text;checked=[]
def capture(path,text,*args,**kwargs):
 assert path.parent==base and path.name in ['result.json','artifact-map.json','results.md'],path
 write(out/path.name,text,*args,**kwargs)
 if path.suffix=='.json':assert json.loads(text)==json.loads(path.read_text()),path
 else:assert text==path.read_text(),path
 checked.append(path.name);return len(text)
Path.write_text=capture
try:
 with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(base/'summarize_broad.py'),run_name='__main__')
finally:Path.write_text=write
assert len(checked)==3
receipt={'status':'pass','json_and_markdown_exact':checked,'model_calls':0,'candidate_mutations':0}
(Path(__file__).parent/'spatial-aggregate-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
