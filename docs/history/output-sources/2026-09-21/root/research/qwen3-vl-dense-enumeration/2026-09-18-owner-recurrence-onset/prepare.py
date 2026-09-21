import json,hashlib,pathlib,datetime
from PIL import Image,ImageDraw
R=pathlib.Path(__file__).parent;BASE=R.parent;N=BASE/'2026-09-16-endpoint-loop-natural-readout-norm';S=BASE/'2026-09-17-history-rereading-mechanism/human-evaluation/annotation-snapshot-v1'
def bind(p):
 p=pathlib.Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
p=json.load(open(N/'panel.json'));g=next(g for g in p['groups'] if 417044 in g['focus_ids']);target=next(i for i,x in enumerate(g['rows']) if x['image_id']==417044)
native=json.load(open(N/f"runtime/{g['key']}-identity/raw.json"))['rows'];treated=json.load(open(N/f"runtime/{g['key']}-norm/raw.json"))['rows']
def rows(d):
 ids=d['token_ids']; starts=[i for i,t in enumerate(ids) if t==151646];out=[]
 for a in starts:
  if 151649 not in ids[a:]:continue
  e=ids.index(151649,a)+1;t=ids[a:e];b=t.index(151648)+1
  out.append(dict(offset=a,end=e,tokens=t,box=[x-151670 for x in t[b:b+4]],text=d['text'].split('<|object_ref_start|>')[len(out)+1].split('<|box_end|>')[0]+'<|box_end|>'))
 return out
nr=rows(native[target]);tr=rows(treated[target]);ann=next(json.loads(l) for l in open(S/'working.norm.jsonl') if json.loads(l)['image_id']==417044)
m=json.load(open(S/'manifest.json'));checks={k:bind(v['path'])==v for k,v in m['source'].items() if isinstance(v,dict) and set(['path','sha256','size_bytes'])<=v.keys()};assert checks['working_norm'] and checks['source_norm'] and checks['last_export'];ib=m['image_bindings']['417044'];assert bind(ib['resolved_path'])['sha256']==ib['sha256'];write(R/'annotation-identity.json',dict(checks=checks,image=ib,positive_count=len(ann['objects']),ignore='No explicit ignore fields; unmatched is UNKNOWN',snapshot=bind(S/'manifest.json')))
# A: native rows5/6 same main donut despite left-neighbor overlap. B: treated row4 distinct right neighboring donut.
candidates={name:dict(row,owner_id=owner,support=support) for name,row,owner,support in [('A_seed',nr[4],-1693019979812657,'native row5'),('A_recurrence',nr[5],-1693019979812657,'native row6'),('B_native_geometry',tr[3],-8380415314849442,'saved treated row4')]}
b=dict(candidates['B_native_geometry']);b['tokens']=b['tokens'].copy();q=b['tokens'].index(151648)+1;b['box']=[57,274,105,327];b['tokens'][q:q+4]=[151670+x for x in b['box']];b['support']='current refined extent, secondary to saved native geometry';candidates['B_refined_geometry']=b
windows=[]
for label,index in [('possible_seed',2),('possible_recurrence',3),('confirmed_seed',4),('confirmed_recurrence',5)]:
 c=dict(candidates) if index>=4 else {'observed':nr[index],'previous':nr[index-1]}
 windows.append(dict(name=label,policy_history='original natural',row=index+1,offset=nr[index]['offset'],history=native[target]['token_ids'][:nr[index]['offset']],candidates=c))
windows.append(dict(name='normal_transition',policy_history='saved equal-norm natural; scored under original readout',row=4,offset=tr[3]['offset'],history=treated[target]['token_ids'][:tr[3]['offset']],candidates=candidates))
assert len(windows)==5
sources=p['sources']+[bind(N/'panel.json'),bind(N/'coefficients.pt'),bind(N/f"runtime/{g['key']}-identity/raw.json"),bind(N/f"runtime/{g['key']}-identity/receipt.json"),bind(N/f"runtime/{g['key']}-norm/raw.json"),bind(S/'working.norm.jsonl'),bind(S/'manifest.json')]
write(R/'panel.json',dict(schema='owner_recurrence_onset.v1',image_id=417044,target=target,config=p['config'],group=g,sources=sources,coordinate_ids=p['coordinate_ids'],coefficients=p['coefficients'],windows=windows,native_raw=bind(N/f"runtime/{g['key']}-identity/raw.json"),native_receipt=bind(N/f"runtime/{g['key']}-identity/receipt.json"),bounds=dict(replay_calls=3084,score_calls=sum(len(w['candidates']) for w in windows),total_planned=3100,hard_calls=20000,gpu_hours=3,tensor_bytes=1073741824),native_capture_offsets=list(range(19,70))))
write(R/'review/admission.json',dict(inspected_native_rows=list(range(1,9)),inspected_treated_rows=[4],inspected_target_rows=9,ceiling=12,earliest_possible=dict(row=4,seed=3,status='HOLD identity/extent competition',reason='Both overlap left-border partial donut and adjacent donut; cannot uniquely assign emitted intent from boxes alone'),earliest_confirmed=dict(row=6,seed=5,owner_id=-1693019979812657,status='admitted',reason='Both boxes enclose the same dominant glazed donut immediately above bottom-left donut, with extent overlap on left neighbor'),B=dict(owner_id=-8380415314849442,row=4,route='treated',unvisited_through_native_row5=True,reason='Distinct neighboring donut on the right, outside native rows1..5 extents apart from small border overlap'),subsequent_burst=dict(rows=[7,8],description='row7 returns toward ambiguous upper region; row8 returns to A'),physical_limit='Owner intent is adjudicated from visible dominant instance, not guaranteed by IoU; row3/4 ambiguity prevents claiming row6 earliest actual recurrence',controls='351017/7116 not executed; primary finite diagnosis sufficient'))
img=Image.open(ib['resolved_path']).convert('RGB');draw=ImageDraw.Draw(img)
for label,box,color in [('A5',nr[4]['box'],'red'),('A6',nr[5]['box'],'orange'),('B',tr[3]['box'],'lime')]:
 xy=[box[0]*img.width/999,box[1]*img.height/999,box[2]*img.width/999,box[3]*img.height/999];draw.rectangle(xy,outline=color,width=3);draw.text((xy[2]+3,xy[1]),label,fill=color)
img.save(R/'review/contrast-full.png');img.crop((0,170,180,310)).resize((900,700)).save(R/'review/contrast-local.png')
