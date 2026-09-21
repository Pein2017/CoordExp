"""Partial-review inventory after user stop; no physical population estimator."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent/'physical-review'
def main():
 l=json.load(open(R/'reviewer-luna/partial-summary.json'));t=json.load(open(R/'reviewer-terra/partial-review.json'));missing=[];reported={}
 for name,inventory in [('luna',l['viewed']),('terra',{str(x['image_id']):x['viewed_paths'] for x in t['images']})]:
  for iid,paths in inventory.items():
   reported.setdefault(iid,[]).append(name)
   for path in paths:
    if not (R/iid/path).is_file():missing.append(dict(reviewer=name,image_id=iid,path=path))
 result=dict(status='partial_selective_model_assisted_only',user_stop_honored=True,reported_images=len(reported),planned_sample=32,reported_image_ids=sorted(map(int,reported)),unreviewed_images=sorted(set(x['image_id'] for x in json.load(open(R/'selection.json'))['selected'])-set(map(int,reported))),invalid_view_inventory_entries=missing,physical_population_estimate=None,known_bank_unchanged=True,claim='Reported partial viewing is not complete adjudication. Unsupported/nonexistent inventory paths are excluded from verified coverage; affected image correspondence claims held. No quantitative physical net estimate or GT admission.',sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [R/'reviewer-luna/partial-summary.json',R/'reviewer-terra/partial-review.json',R/'root-selective-check.json']],root_additional_views=1)
 (R/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
