from pathlib import Path
from types import MappingProxyType
import src.label_studio_coco_refinement.project as p
ROOT = Path('/data/CoordExp/outputs/_probe_labelstudio_bootstrap_1k_20260716').resolve()
contracts = {}
for split, n, rel, image_subdir in [
    (p.Split.TRAIN, 1000, 'slices/train.norm.jsonl', 'train2017'),
    (p.Split.VAL, 200, 'slices/val.norm.jsonl', 'val2017'),
]:
    path = ROOT / rel
    rows = [__import__('json').loads(line) for line in path.read_text().splitlines()]
    boxes = sum(len(r['objects']) for r in rows)
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    contracts[split] = p.SourceContract(split=split, relative_path=path, sha256=sha, row_count=n, box_count=boxes, image_subdirectory=image_subdir)
p.SOURCE_CONTRACTS = MappingProxyType(contracts)
def _layout(repo_root):
    root = ROOT / 'runtime'
    return p.RuntimeLayout(repo_root=Path(repo_root).resolve(), root=root,
        label_studio_state=root / 'label-studio' / 'state',
        image_root=Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images'))
p.RuntimeLayout.for_repo = classmethod(lambda cls, repo_root: _layout(repo_root))
