import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent
bind=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
checks=[]
for name in sys.argv[1:]:
 panel_path=R/f'stage1/panels/309264-{name}.json';panel=json.loads(panel_path.read_text());case=panel['cases'][0]
 folder=R/f'stage1/runtime/{name}/309264/prefix';rec=json.loads((folder/'receipt.json').read_text());raw=json.loads((folder/'raw.json').read_text())
 assert rec['status']=='candidate_complete' and rec['panel']==bind(panel_path)
 assert rec['raw']==bind(folder/'raw.json')
 assert int((R/f'logs/{name}.exit').read_text())==0
 saved=json.loads(Path(case['saved_raw']['path']).read_text())
 expected={r['image_id']:r for r in saved['rows']}
 if case.get('expected_target_raw'):
  expected[309264]=next(r for r in json.loads(Path(case['expected_target_raw']['path']).read_text())['rows'] if r['image_id']==309264)
 for row in raw['rows']:
  if row['image_id']==309264:
   assert row['token_ids'][:63]==case['target_prefix_token_ids']
   if not case.get('expected_target_raw'):continue
  assert row['token_ids']==expected[row['image_id']]['token_ids']
  assert row['stop']==expected[row['image_id']]['stop']
 assert rec['model_forwards']<=3084
 checks.append(dict(cell=name,receipt=bind(folder/'receipt.json'),all_companions_exact=True,target_exact_saved=name in ['SS','FF'],forwards=rec['model_forwards']))
out=R/('stage1-verification-'+'-'.join(sys.argv[1:])+'.json');out.write_text(json.dumps({'status':'passed','checks':checks},indent=2)+'\n');print(json.dumps(checks))
