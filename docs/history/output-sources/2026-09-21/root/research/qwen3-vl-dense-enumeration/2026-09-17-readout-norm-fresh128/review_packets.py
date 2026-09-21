"""Frozen stratified selection and blinded paired evidence rendering only."""
import collections,hashlib,json,random
from pathlib import Path
from PIL import Image,ImageDraw
from probes.training_set_completion.review_packets import _draw_box,_font,_pixel_from_bins_for_viewing,_pixel_from_bins
R=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def main():
 result=read(R/'fresh-result.json');panel=read(R/'panel.json');strata={k:[] for k in ['loss','gain','burden_only','stable']}
 for iid,x in result['images'].items():
  if x['cohort']!='fresh':continue
  k='loss' if x['L'] else 'gain' if x['G'] else 'burden_only' if x['baseline']['burden']!=x['treated']['burden'] else 'stable'
  strata[k].append(iid)
 rng=random.Random(19);shuffled={k:rng.sample(sorted(v,key=int),len(v)) for k,v in strata.items()};counts={k:min(8,len(v)) for k,v in strata.items()}
 while sum(counts.values())<32:
  for k in strata:
   if counts[k]<len(strata[k]) and sum(counts.values())<32:counts[k]+=1
 selected=[dict(image_id=int(iid),stratum=k,inclusion_probability=counts[k]/len(strata[k]),weight=len(strata[k])/counts[k]) for k in strata for iid in shuffled[k][:counts[k]]]
 root=R/'physical-review';root.mkdir(exist_ok=False)
 (root/'selection.json').write_text(json.dumps(dict(seed=19,stratum_sizes={k:len(v) for k,v in strata.items()},allocations=counts,selected=selected,result_sha256=hashlib.sha256((R/'fresh-result.json').read_bytes()).hexdigest()),indent=2)+'\n')
 cases={str(c['input_record']['image_id']):c for g in panel['groups'] for c in g['cases']};mapping={};packets=[]
 for s in selected:
  iid=str(s['image_id']);case=cases[iid];x=result['images'][iid];folder=root/iid;folder.mkdir();im=Image.open(case['image_path']).convert('RGB');im.save(folder/'original.png')
  order=['baseline','treated']
  if int(hashlib.sha256(('blind19:'+iid).encode()).hexdigest(),16)%2:order.reverse()
  mapping[iid]=dict(zip(['A','B'],order));packet=dict(image_id=int(iid),original=str(folder/'original.png'),policies={},instructions='Blind physical comparison; known-reference counts deliberately omitted. Use full image and BOTH policy outputs. Numeric IoU is not identity. Identical repeats grouped for viewing only. Unknown stays unknown. Do not alter any bank.')
  for letter,arm in zip(['A','B'],order):
   rows=x[arm]['complete_rows'];groups={}
   for row in rows:
    key=(row['description'],tuple(row['box']))
    if key not in groups:groups[key]=dict(description=row['description'],box=row['box'],rows=[])
    groups[key]['rows'].append(row['row'])
   entries=list(groups.values());pages=[]
   for start in range(0,len(entries),12):
    canvas=im.copy();draw=ImageDraw.Draw(canvas)
    for j,e in enumerate(entries[start:start+12],start):
     box=(_pixel_from_bins(e['box'],im.width,im.height) if e['box'][0]<e['box'][2] and e['box'][1]<e['box'][3] else _pixel_from_bins_for_viewing(e['box'],im.width,im.height));_draw_box(draw,box,(255,40,40),f"{letter}{j+1} {e['description']} x{len(e['rows'])}",2)
    path=folder/f'{letter}-page{start//12+1}.png';canvas.save(path);pages.append(str(path))
   packet['policies'][letter]=dict(unique_rows=entries,pages=pages,total_complete_rows=len(rows),stop=x[arm]['stop'],malformed=x[arm]['burden']['malformed'])
  path=folder/'packet.json';path.write_text(json.dumps(packet,indent=2)+'\n');packets.append(dict(image_id=int(iid),packet=str(path)))
 (root/'blinding-key.json').write_text(json.dumps(mapping,indent=2)+'\n');(root/'packets.json').write_text(json.dumps(packets,indent=2)+'\n');print(json.dumps(dict(images=len(packets),strata={k:len(v) for k,v in strata.items()},allocations=counts)))
if __name__=='__main__':main()
