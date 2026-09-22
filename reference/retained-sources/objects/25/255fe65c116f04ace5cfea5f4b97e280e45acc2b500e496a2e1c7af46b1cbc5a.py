import asyncio,json
from pathlib import Path
import websockets
R=Path(__file__).resolve().parent
async def main():
 async with websockets.unix_connect('/data/CoordExp/.codex/app-server-control/app-server-control.sock',uri='ws://localhost/rpc',proxy=None,compression=None,max_size=16777216) as ws:
  async def call(i,method,params):
   await ws.send(json.dumps(dict(jsonrpc='2.0',id=i,method=method,params=params)))
   while True:
    x=json.loads(await ws.recv())
    if x.get('id')==i:return x
  await call(1,'initialize',dict(clientInfo=dict(name='research-execution-milestone',version='1.0'),capabilities=dict(experimentalApi=True)))
  await ws.send(json.dumps(dict(jsonrpc='2.0',method='initialized',params={})))
  message='Execution milestone from existing worker task01a0a81a-9e32-7db1-bd07-86fa601f4276, not terminal/acceptance request: fresh128 paired inference is complete. Known matches578->592, G30/L16, net+14; image-bootstrap mean CI[-0.015625,0.28125] includes zero. Structurally healthy119 net+1 (G13/L12); unhealthy9 net+13. Invalid476->5, caps2->0; one newly invalid image339094. Fixed32-image blinded bidirectional physical review now assigned to Luna/high and Terra/high; last old diagnostic replay still finishing. No tuning/training. Data source is actual processed rescale_32_1024 len12000, excludes train256/dev128; current selected128 rows cross-view verified despite documented older global manifest drift. Reusable artifact map: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/ARTIFACTS.md. Preserve scope; I will return final candidate with independent saved-output verification and review. Your existing settlement monitor remains appropriate.'
  result=await call(2,'turn/start',dict(threadId='01a0a3d5-dc45-7693-8467-4801aa7190df',input=[dict(type='text',text=message)]))
  (R/'lead-milestone-delivery.json').write_text(json.dumps(result,indent=2)+'\n');print({'delivered': 'result' in result,'error':result.get('error')})
asyncio.run(main())
