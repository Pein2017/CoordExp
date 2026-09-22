"""CPU verification of saved continuity captures; no model execution."""
import json
import hashlib
from pathlib import Path
import torch
b=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
d=b/'2026-09-19-coordinate-input-continuity'
plan=json.loads((d/'execution-plan.json').read_text())
weights={m:torch.load(b/'2026-09-18-untied-highconfidence18-natural/weights'/f'{m}-original'/'weights.pt',map_location='cpu',weights_only=False)['input_rows'] for m in ('tied','untied')}
checks=mutations=bindings=0
for state in plan['states']:
    root=d/'runtime'/state['id']
    native=torch.load(root/'native/native_no_hook/capture.pt',map_location='cpu',weights_only=False)
    for variant in state['variants']:
        folder=root/variant['name']
        receipt=json.loads((folder/'receipt.json').read_text())
        one=torch.load(folder/'native_no_hook/capture.pt',map_location='cpu',weights_only=False)
        hooked=torch.load(folder/'observational_hook/capture.pt',map_location='cpu',weights_only=False)
        for field in ('input_ids','logits','coordinate_logits','input_delta'):
            assert torch.equal(one[field],hooked[field]),(state['id'],variant['name'],field)
        assert one['positions']==hooked['positions']==native['positions']
        assert torch.isfinite(one['logits']).all()
        changes=(one['input_ids']!=native['input_ids']).nonzero().flatten()
        if variant['name']=='native':assert changes.numel()==0 and torch.count_nonzero(one['input_delta'])==0
        else:
            assert changes.numel()==1
            i=int(changes[0]);old=int(native['input_ids'][i]);new=int(one['input_ids'][i])
            assert new==variant['token_id'] and abs(new-old)==1
            E=weights[state['model']]
            assert torch.equal(one['input_delta'],E[new-151670+4]-E[old-151670+4])
            mutations+=1
        def audit(obj):
            global bindings
            if isinstance(obj,dict):
                if 'path' in obj and 'sha256' in obj:
                    assert hashlib.sha256(Path(obj['path']).read_bytes()).hexdigest()==obj['sha256'];bindings+=1
                else:
                    for v in obj.values():audit(v)
            elif isinstance(obj,list):
                for v in obj:audit(v)
        audit(receipt);checks+=1
result={'status':'passed','states':len(plan['states']),'bitwise_hook_pairs':checks,'exact_one_token_mutations':mutations,'actual_trained_input_deltas_exact':True,'capture_bindings':bindings}
(b/'2026-09-19-recurrence-distribution-census/integration/continuity-capture-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
