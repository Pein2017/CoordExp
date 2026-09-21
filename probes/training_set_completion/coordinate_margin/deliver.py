"""One authorized worker-to-lead completion delivery, no settings overrides."""
import asyncio
import hashlib
import json
import sys
from pathlib import Path
import aiohttp

async def main(message_path, receipt_path):
    root='01a0a3d5-dc45-7693-8467-4801aa7190df'
    home=Path('/data/CoordExp/.codex')
    message=message_path.read_text()
    with receipt_path.open('x') as out:
        record={'root':root,'message_sha256':hashlib.sha256(message.encode()).hexdigest(),'status':'prepared'}
        def save(**kw):
            record.update(kw);out.seek(0);json.dump(record,out,indent=2);out.write('\n');out.truncate();out.flush()
        save()
        async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=str(home/'app-server-control/app-server-control.sock'))) as session:
            async with session.ws_connect('http://localhost/',max_msg_size=0) as ws:
                serial=0
                async def call(method,params):
                    nonlocal serial
                    serial+=1;ident=serial
                    await ws.send_json({'id':ident,'method':method,'params':params})
                    async with asyncio.timeout(30):
                        async for raw in ws:
                            if raw.type!=aiohttp.WSMsgType.TEXT:raise RuntimeError('non-text socket response')
                            r=raw.json()
                            if r.get('id')==ident:
                                if 'error' in r:raise RuntimeError(r['error'])
                                return r['result']
                    raise RuntimeError('response missing')
                try:
                    init=await call('initialize',{'clientInfo':{'name':'coordinate_margin_worker_return','version':'1.0'},'capabilities':{'experimentalApi':True,'requestAttestation':False}})
                    assert Path(init['codexHome']).resolve()==home
                    await ws.send_json({'method':'initialized'})
                    thread=(await call('thread/read',{'threadId':root,'includeTurns':True}))['thread']
                    if thread['status']['type']=='notLoaded':
                        await call('thread/resume',{'threadId':root})
                        thread=(await call('thread/read',{'threadId':root,'includeTurns':True}))['thread']
                    assert thread['id']==root
                    status=thread['status']['type']
                    params={'threadId':root,'input':[{'type':'text','text':message}]}
                    if status=='idle':method='turn/start'
                    elif status=='active':
                        active=[t for t in thread['turns'] if t['status']=='inProgress']
                        assert len(active)==1, 'active turn ambiguous'
                        params['expectedTurnId']=active[0]['id'];method='turn/steer'
                    else:raise RuntimeError(f'unsupported root status {status}')
                    save(root_status=thread['status'],model=thread.get('model'),reasoningEffort=thread.get('reasoningEffort'),method=method,status='delivery_unknown')
                    result=await call(method,params)
                    save(status='submitted',response=result)
                except Exception as exc:
                    save(error_type=type(exc).__name__,error=str(exc));raise
    print(json.dumps(record))

if __name__=='__main__':asyncio.run(main(Path(sys.argv[1]),Path(sys.argv[2])))
