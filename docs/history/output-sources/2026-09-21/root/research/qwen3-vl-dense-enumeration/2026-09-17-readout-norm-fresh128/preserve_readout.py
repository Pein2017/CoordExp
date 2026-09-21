from pathlib import Path
import torch,json,hashlib
r=Path(__file__).resolve().parent
source=r.parent/'2026-09-16-endpoint-loop-readout-state/runtime/R16-351017/tensors.pt'
t=torch.load(source,map_location='cpu',weights_only=True)
keys=[k for k in t if not isinstance(t[k],torch.Tensor) or k in ['output_rows','input_rows','bias','base_rows','delta_rows']]
out={k:t[k] for k in ['output_rows','input_rows','bias']}
out.update(coordinate_ids=torch.arange(151670,152670),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_path=str(source))
torch.save(out,r/'effective-readout.pt')
print({k:list(v.shape) for k,v in out.items() if isinstance(v,torch.Tensor)})
