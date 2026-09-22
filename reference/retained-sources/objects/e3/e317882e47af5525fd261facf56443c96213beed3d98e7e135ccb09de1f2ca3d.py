import json, pathlib, hashlib
B=pathlib.Path(__file__).parent.parent; S=B/'third-fit-selectors-v1'; P=B/'third-fit-review-packets-v1'; ids=[25274,59571,99937,210457,219546,323322,351017,388795,417044,477415,528944]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def canon(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
assert sha(B/'target-owners-complete-v4.json')=='b5c6534259a46583466dc5b92ae0b4b7854cc5adc4de8bc84801f3d96bc69f64'
assert sha(B/'parent16-v4-physical-ledger-v1/ledger.json')=='f0536ed6a5a158f8bf296f1e736a5b0bdbee9d52349179f4b0ef1064bfcfce5e'
for s in (16,32,64):
 d=json.load(open(S/f'scored-step-{s}.json')); assert [int(x['image_id']) for x in d['rows']]==ids and d['missing_image_ids']==[]
 for x in d['rows']:
  assert len(x['raw_rows'])==x['raw_row_count']==len({z['prediction_id'] for z in x['raw_rows']})
  assert x['valid_prediction_count']==sum(z['status']=='parsed_valid' for z in x['raw_rows'])
r=json.load(open(P/'receipt.json')); assert r['rendered_image_ids']==ids and r['missing_image_ids']==[]
for q in r['packets']:
 d=json.load(open(q['packet'])); assert d['image_id'] in ids and d['checkpoint_step']==32
 assert d['counts']['raw_row_count']==len(d['rendered']['raw_rows'])
 assert d['counts']['valid_prediction_count']==sum(x['status']=='parsed_valid' for x in d['rendered']['raw_rows'])
 assert d['counts']['raw_invalid_or_dropped_count']==sum(x['status']!='parsed_valid' for x in d['rendered']['raw_rows'])
 assert len({x['prediction_id'] for x in d['rendered']['raw_rows']})==d['counts']['raw_row_count']
 for x in d['rendered']['raw_rows']:
  assert pathlib.Path(x['shared_crop_paths']['tight']).exists() and pathlib.Path(x['shared_crop_paths']['context']).exists()
print(json.dumps({'validated':True,'selectors':33,'step32_packets':11,'step32_raw_rows':sum(q['raw_row_count'] for q in r['packets']),'target_sha256':sha(B/'target-owners-complete-v4.json'),'parent_ledger_sha256':sha(B/'parent16-v4-physical-ledger-v1/ledger.json')},sort_keys=True))
