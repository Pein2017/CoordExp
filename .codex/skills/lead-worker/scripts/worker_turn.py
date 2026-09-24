#!/usr/bin/env python3
"""Inspect, create, or send to a same-host lead/worker pair."""
import argparse
import asyncio
import hashlib
import json
import shlex
from pathlib import Path
from uuid import UUID

import aiohttp

HOME_DIR = Path('/data/CoordExp/.codex')


def check_pair(lead, worker, args):
    for thread, ident in ((lead, args.lead_thread), (worker, args.worker_thread)):
        if thread['id'] != ident or Path(thread['cwd']).resolve() != args.cwd:
            raise ValueError('Thread identity or cwd mismatch')
    if lead['id'] == worker['id']:
        raise ValueError('Lead and worker must be different tasks')
    target = {'lead': lead, 'worker': worker}[getattr(args, 'to', 'worker')]
    if target['status']['type'] not in {'idle', 'notLoaded', 'active'}:
        raise ValueError('Target is unavailable; reconcile it before dispatch')
    if target.get('canAcceptDirectInput') is False:
        raise ValueError('Target cannot accept direct input')
    return target


async def operate(call, args):
    if getattr(args, 'create', False):
        return await create_worker(call, args)

    async def pair():
        return [
            (await call('thread/read', {'threadId': ident, 'includeTurns': False}))['thread']
            for ident in (args.lead_thread, args.worker_thread)
        ]

    lead, worker = await pair()
    target = check_pair(lead, worker, args)
    summary = {
        'lead': {k: lead[k] for k in ('id', 'model', 'reasoningEffort', 'status')},
        'worker': {k: worker[k] for k in ('id', 'model', 'reasoningEffort', 'status')},
        'target': getattr(args, 'to', 'worker'), 'target_thread_id': target['id'],
        'cwd': str(args.cwd), 'status': 'inspected_only',
    }
    if not args.send:
        return summary
    target_settings = (target['model'], target['reasoningEffort'])
    message = args.message.read_text(encoding='utf-8')
    if not message.strip():
        raise ValueError('Assignment is empty')
    # One owner, one immutable dispatch path; unknown delivery must be reconciled.
    with args.receipt.open('x', encoding='utf-8') as receipt:
        def save(status, **extra):
            summary.update(status=status, **extra)
            receipt.seek(0)
            json.dump(summary, receipt, ensure_ascii=False, indent=2)
            receipt.write('\n')
            receipt.truncate()
            receipt.flush()
        save('prepared', message_sha256=hashlib.sha256(message.encode()).hexdigest())
        try:
            if target['status']['type'] == 'notLoaded':
                await call('thread/resume', {'threadId': target['id']})
            lead, worker = await pair()
            target = check_pair(lead, worker, args)
            summary['lead'] = {k: lead[k] for k in ('id', 'model', 'reasoningEffort', 'status')}
            summary['worker'] = {k: worker[k] for k in ('id', 'model', 'reasoningEffort', 'status')}
            if (target['model'], target['reasoningEffort']) != target_settings:
                raise ValueError('Target settings changed during preflight; reconcile before dispatch')
            params = {'threadId': target['id'], 'input': [{'type': 'text', 'text': message}]}
            if target['status']['type'] == 'active':
                page = await call('thread/turns/list', {
                    'threadId': target['id'], 'limit': 1, 'sortDirection': 'desc',
                    'itemsView': 'notLoaded',
                })
                turns = page['data']
                if len(turns) != 1 or turns[0]['status'] != 'inProgress' or not turns[0].get('id'):
                    raise ValueError('Active target has no exact current turn; reconcile before dispatch')
                params['expectedTurnId'] = turns[0]['id']
                save('delivery_unknown', delivery_method='turn/steer')
                result = await call('turn/steer', params)
                if result['turnId'] != params['expectedTurnId']:
                    raise ValueError('Unexpected steered turn; reconcile delivery')
                save('submitted', turn_id=result['turnId'])
            else:
                if target['status']['type'] != 'idle':
                    raise ValueError('Target did not become ready after resume')
                save('delivery_unknown', delivery_method='turn/start')
                result = await call('turn/start', params)
                save('submitted', turn_id=result['turn']['id'], turn_status=result['turn']['status'])
        except Exception as exc:
            save(summary['status'], error_type=type(exc).__name__)
            raise
    return summary


async def create_worker(call, args):
    lead = (await call('thread/read', {
        'threadId': args.lead_thread, 'includeTurns': False,
    }))['thread']
    if lead['id'] != args.lead_thread or Path(lead['cwd']).resolve() != args.cwd:
        raise ValueError('Lead identity or cwd mismatch')
    if args.worker_thread is not None or args.to != 'worker' or not args.send:
        raise ValueError('Creation requires --create, --to worker, and --send')
    if not all(isinstance(value, str) and value.strip()
               for value in (args.name, args.model, args.effort)):
        raise ValueError('Creation requires nonempty name, model, and effort')
    if args.message is None or args.receipt is None:
        raise ValueError('Creation requires message and receipt paths')
    message = args.message.read_text(encoding='utf-8')
    if not message.strip():
        raise ValueError('Assignment is empty')
    summary = {
        'lead': {k: lead[k] for k in ('id', 'model', 'reasoningEffort', 'status')},
        'worker_name': args.name, 'requested_model': args.model,
        'requested_effort': args.effort, 'cwd': str(args.cwd),
    }
    with args.receipt.open('x', encoding='utf-8') as receipt:
        def save(status, **extra):
            summary.update(status=status, **extra)
            receipt.seek(0)
            json.dump(summary, receipt, ensure_ascii=False, indent=2)
            receipt.write('\n')
            receipt.truncate()
            receipt.flush()

        save('creation_unknown', phase='thread/start',
             message_sha256=hashlib.sha256(message.encode()).hexdigest())
        try:
            result = await call('thread/start', {
                'model': args.model, 'cwd': str(args.cwd), 'ephemeral': False,
                'config': {'model_reasoning_effort': args.effort},
            })
            worker = result['thread']
            worker_id = worker['id']
            save('created', phase='validate_worker', worker_thread_id=worker_id,
                 worker={k: worker.get(k) for k in
                         ('id', 'cwd', 'model', 'reasoningEffort', 'status')})
            if (str(UUID(worker_id)) != worker_id or worker_id == lead['id']
                    or Path(worker['cwd']).resolve() != args.cwd
                    or worker['model'] != args.model
                    or worker['reasoningEffort'] != args.effort):
                raise ValueError('Created worker identity, cwd, or settings mismatch')
            save('created', phase='thread/name/set')
            await call('thread/name/set', {'threadId': worker_id, 'name': args.name})
            return_command = shlex.join([
                'python', str(Path(__file__).resolve()), '--lead-thread', lead['id'],
                '--worker-thread', worker_id, '--cwd', str(args.cwd), '--to', 'lead',
                '--send', '--message', 'REPORT_FILE', '--receipt', 'NEW_RECEIPT_FILE',
            ])
            assignment = (message + '\n\nTransport context: Lead task ' + lead['id']
                          + '; worker task ' + worker_id
                          + '. Return findings directly to the lead with `'
                          + return_command + '`.')
            save('delivery_unknown', phase='turn/start', delivery_method='turn/start',
                 sent_sha256=hashlib.sha256(assignment.encode()).hexdigest())
            turn = (await call('turn/start', {
                'threadId': worker_id, 'input': [{'type': 'text', 'text': assignment}],
            }))['turn']
            save('submitted', phase='submitted', turn_id=turn['id'], turn_status=turn['status'])
        except Exception as exc:
            save(summary['status'], error_type=type(exc).__name__)
            raise
    return summary


async def main(args):
    socket = HOME_DIR / 'app-server-control/app-server-control.sock'
    async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=str(socket))) as session:
        async with session.ws_connect('http://localhost/', max_msg_size=0) as ws:
            serial = 0

            async def call(method, params):
                nonlocal serial
                serial += 1
                ident = serial
                await ws.send_json({'id': ident, 'method': method, 'params': params})
                async with asyncio.timeout(30):
                    async for raw in ws:
                        if raw.type != aiohttp.WSMsgType.TEXT:
                            raise RuntimeError('App Server socket ended or returned non-text data')
                        response = raw.json()
                        if response.get('id') == ident:
                            if 'error' in response:
                                raise RuntimeError(response['error'])
                            return response['result']
                raise RuntimeError('App Server ended without a response')

            init = await call('initialize', {
                'clientInfo': {'name': 'lead_worker', 'version': '1.0'},
                'capabilities': {'experimentalApi': True, 'requestAttestation': False},
            })
            if Path(init['codexHome']).resolve() != HOME_DIR:
                raise ValueError('Unexpected Codex home')
            await ws.send_json({'method': 'initialized'})
            print(json.dumps(await operate(call, args), ensure_ascii=False, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-thread', required=True, type=lambda s: str(UUID(s.removeprefix('codex://threads/'))))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--worker-thread', type=lambda s: str(UUID(s.removeprefix('codex://threads/'))))
    mode.add_argument('--create', action='store_true')
    parser.add_argument('--cwd', required=True, type=Path)
    parser.add_argument('--to', choices=('worker', 'lead'), default='worker')
    parser.add_argument('--send', action='store_true')
    parser.add_argument('--message', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--model')
    parser.add_argument('--effort')
    args = parser.parse_args(argv)
    args.cwd = args.cwd.resolve(strict=True)
    if not args.cwd.is_dir():
        parser.error('--cwd must be an existing directory')
    if args.send and (args.message is None or args.receipt is None):
        parser.error('--send requires --message and a new --receipt path')
    if args.create and (not args.send or args.to != 'worker' or
                        not all(value and value.strip() for value in
                                (args.name, args.model, args.effort))):
        parser.error('--create requires --name, --model, --effort, --send, and --to worker')
    if not args.create and any(value is not None for value in
                               (args.name, args.model, args.effort)):
        parser.error('--name, --model, and --effort require --create')
    return args


if __name__ == '__main__':
    asyncio.run(main(parse_args()))
