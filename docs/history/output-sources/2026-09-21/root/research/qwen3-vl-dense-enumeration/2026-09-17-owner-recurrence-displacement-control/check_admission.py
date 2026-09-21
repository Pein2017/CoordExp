import hashlib,json,re,math
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
PRE=ROOT.parent/'2026-09-17-owner-recurrence-row-branch'
PAT=re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>'+r'<\|coord_(\d+)\|>'*4+r'<\|box_end\|>')
def binding(p):
 p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
manifest=json.loads((PRE/'source-manifest.json').read_text());cases={c['image_id']:c for c in manifest['cases']}
def rows(image,arm):
 c=cases[image];p=Path(c['raw'][arm]['path']);assert binding(p)['sha256']==c['raw'][arm]['sha256']
 r=next(x for x in json.loads(p.read_text())['rows'] if x['image_id']==image)
 return [{'index':i,'category':m[0],'box':list(map(int,m[1:]))} for i,m in enumerate(PAT.findall(r['text']))]
def movement(a,b):
 ca=[(a[i]+a[i+2])/2 for i in range(2)];cb=[(b[i]+b[i+2])/2 for i in range(2)];d=[cb[i]-ca[i] for i in range(2)]
 return {'center_delta':d,'center_distance':math.hypot(*d),'target_extent':[b[2]-b[0],b[3]-b[1]],'changed_coordinate_fields':[k for k,i in zip(['x1','y1','x2','y2'],range(4)) if a[i]!=b[i]]}
selections=[(181260,89,[('covered O87','O',87),('nearby N76','N',76)]),(253212,16,[('covered O9','O',9),('future O20','O',20)])]
checks=[]
for image,index,alts in selections:
 original=rows(image,'O');base=original[index];entry={'image_id':image,'recurrence_candidate':base,'alternatives':[]}
 for label,arm,j in alts:
  row=rows(image,arm)[j];entry['alternatives'].append({'label':label,'source_arm':arm,**row,'literal_precedes_recurrence':arm=='O' and j<index,'movement':movement(base['box'],row['box'])})
 # Plot only the selected hypotheses, not all proposals. Colors designate role, not truth/match.
 im=Image.open(cases[image]['image']['path']).convert('RGB');draw=ImageDraw.Draw(im);w,h=im.size
 for label,row,color in [('native candidate',base,'red')]+[(v['label'],v,c) for v,c in zip(entry['alternatives'],['cyan','yellow'])]:
  b=row['box'];xy=[b[0]*w/999,b[1]*h/999,b[2]*w/999,b[3]*h/999];draw.rectangle(xy,outline=color,width=2);draw.text((xy[0],max(0,xy[1]-12)),label,fill=color)
 im.save(ROOT/f'{image}-selected-full.png');checks.append(entry)
# Sensitivity: a row earlier than the boundary can never be an uncovered candidate.
boundary=89;proposed=87;assert proposed<boundary
assert not (proposed>=boundary)
for c in cases.values():
 for arm in ['O','N']:rows(c['image_id'],arm)
result={'status':'admission-HOLD','model_calls':0,'screen_images':sorted(cases),'screen_scope':'Existing eight-image screen only; selected metadata alternatives, not exhaustive physical census.','selected_checks':checks,'prefix_counterexample':{'image_id':181260,'boundary_O':89,'proposed_uncovered_O':87,'verdict':'already in literal common prefix; cannot serve as uncovered'},'bindings':[binding(PRE/'source-manifest.json'),binding(PRE/'lead-acceptance.json')]+[binding(c['raw'][a]['path']) for c in cases.values() for a in ['O','N']]+[binding(c['image']['path']) for c in cases.values()]}
(ROOT/'admission-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'bindings':len(result['bindings']),'selected_checks':checks},indent=2))
