import asyncio
import hashlib
import json
from pathlib import Path
import aiohttp

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-wrapper-history-factorial/coordination')
THREAD = '01a0a81a-9e32-7db1-bd07-86fa601f4276'
SOCKET = '/data/CoordExp/.codex/app-server-control/app-server-control.sock'

async def main():
    async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=SOCKET)) as session, session.ws_connect('http://localhost/', max_msg_size=0) as ws:
        serial = 0
        async def call(method, params):
            nonlocal serial
            serial += 1
            request_id = serial
            await ws.send_json({'id': request_id, 'method': method, 'params': params})
            async with asyncio.timeout(45):
                async for raw in ws:
                    if raw.type != aiohttp.WSMsgType.TEXT:
                        raise RuntimeError('unexpected WebSocket message type: '+str(raw.type))
                    reply = raw.json()
                    if reply.get('id') == request_id:
                        if 'error' in reply:
                            raise RuntimeError(json.dumps(reply['error']))
                        return reply['result']
            raise RuntimeError('connection ended without response')
        initialized = await call('initialize', {'clientInfo': {'name': 'coordexp_authorized_lead_dispatch', 'version': '1.0'}, 'capabilities': {'experimentalApi': True, 'requestAttestation': False}})
        assert Path(initialized['codexHome']).resolve() == Path('/data/CoordExp/.codex').resolve()
        await ws.send_json({'method': 'initialized'})
        before = (await call('thread/read', {'threadId': THREAD, 'includeTurns': False}))['thread']
        assert before['id'] == THREAD and before['cwd'] == '/data/CoordExp/.worktrees/research-probes', before.get('cwd')
        status = before['status']['type']
        assert status in ['idle', 'notLoaded'], status
        if status == 'notLoaded':
            await call('thread/resume', {'threadId': THREAD})
        message = (ROOT/'worker-message.txt').read_text()
        inputs = [{'type': 'text', 'text': message}, {'type': 'skill', 'name': 'native-agent-team-guidance', 'path': '/data/CoordExp/.codex/skills/native-agent-team-guide/SKILL.md'}, {'type': 'skill', 'name': 'research-flow', 'path': '/data/CoordExp/.codex/skills/research-flow/SKILL.md'}]
        result = await call('turn/start', {'threadId': THREAD, 'input': inputs})
        after = (await call('thread/read', {'threadId': THREAD, 'includeTurns': False}))['thread']
        assert after['id'] == THREAD and after['cwd'] == before['cwd']
        receipt = {'post_dispatch_status': after['status'], 'thread_id': THREAD, 'prior_status': status, 'message_sha256': hashlib.sha256(message.encode()).hexdigest(), 'turn': result['turn'], 'transport': 'existing app-server Unix WebSocket; no setting overrides'}
        (ROOT/'dispatch-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps({'thread_id': THREAD, 'turn_id': result['turn']['id'], 'status': result['turn']['status'], 'receipt': str(ROOT/'dispatch-receipt.json')}))

asyncio.run(main())
