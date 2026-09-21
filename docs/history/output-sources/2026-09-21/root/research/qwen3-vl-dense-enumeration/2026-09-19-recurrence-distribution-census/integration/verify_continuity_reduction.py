import json, sys, hashlib
from pathlib import Path
from probes.training_set_completion.coordinate_continuity import reduce as r
root=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-input-continuity')
out=Path(__file__).parent/'continuity-reduction-recomputed'
out.mkdir(exist_ok=True)
checks=[]
def capture(path,value):
    path=Path(path)
    target=out/path.name
    target.write_text(json.dumps(value,indent=2)+'\n')
    old=json.loads(path.read_text())
    extra=sorted(set(old)-set(value))
    equal=all(old.get(k)==v for k,v in value.items())
    checks.append({'file':str(path),'recomputed_fields_json_exact':equal,'non_reducer_fields':extra})
    assert equal, path
r.write_json=capture
sys.argv=['reduce.py','--output-root',str(root)]
r.main()
(Path(__file__).parent/'continuity-reduction-check.json').write_text(json.dumps({'checks':checks,'status':'pass'},indent=2)+'\n')
