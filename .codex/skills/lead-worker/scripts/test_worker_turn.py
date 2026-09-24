"""CPU-only consumer checks; never contacts a real task."""
import asyncio
import copy
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import worker_turn
from worker_turn import operate


class WorkerTurnTest(unittest.TestCase):
    def test_create_and_first_turn_on_one_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'space $ (x)'
            root.mkdir()
            message = root / 'assignment.txt'
            message.write_text('Inspect the named artifact; return evidence and stop.', encoding='utf-8')
            lead_id = '00000000-0000-4000-8000-000000000001'
            worker_id = '00000000-0000-4000-8000-000000000002'
            args = SimpleNamespace(lead_thread=lead_id, worker_thread=None, create=True,
                                   cwd=root, name='bounded-worker', model='gpt-6-sol',
                                   effort='xhigh', to='worker', send=True, message=message,
                                   receipt=root / 'create.json')
            lead = dict(id=lead_id, cwd=str(root), model='gpt-6-astra',
                        reasoningEffort='ultra', status={'type': 'active'})
            worker = dict(id=worker_id, cwd=str(root), model=args.model,
                          reasoningEffort=args.effort, status={'type': 'idle'})
            calls = []
            failure = None

            async def call(method, params):
                calls.append((method, params))
                if method == 'thread/read':
                    self.assertEqual(params['threadId'], lead_id)
                    return {'thread': copy.deepcopy(lead)}
                if method == 'thread/start':
                    if failure == 'create_timeout':
                        raise TimeoutError('creation response lost')
                    return {'thread': copy.deepcopy(worker)}
                if method == 'thread/name/set':
                    if failure == 'name':
                        raise RuntimeError('name rejected')
                    return {}
                if method == 'turn/start':
                    if failure == 'send_timeout':
                        raise TimeoutError('send response lost')
                    return {'turn': {'id': 'first-turn', 'status': 'inProgress'}}
                raise AssertionError(method)

            result = asyncio.run(operate(call, args))
            self.assertEqual(result['status'], 'submitted')
            self.assertEqual(result['worker_thread_id'], worker_id)
            self.assertEqual([method for method, _ in calls],
                             ['thread/read', 'thread/start', 'thread/name/set', 'turn/start'])
            self.assertEqual(calls[1][1], {
                'model': args.model, 'cwd': str(root), 'ephemeral': False,
                'config': {'model_reasoning_effort': args.effort},
            })
            self.assertEqual(calls[2][1], {'threadId': worker_id, 'name': args.name})
            sent = calls[3][1]['input'][0]['text']
            self.assertTrue(sent.startswith(message.read_text(encoding='utf-8')))
            self.assertIn(lead_id, sent)
            self.assertIn(worker_id, sent)
            self.assertIn('--to lead', sent)
            command = sent.rsplit('`', 2)[1]
            self.assertEqual(shlex.split(command), [
                'python', str(Path(worker_turn.__file__).resolve()),
                '--lead-thread', lead_id, '--worker-thread', worker_id,
                '--cwd', str(root), '--to', 'lead', '--send',
                '--message', 'REPORT_FILE', '--receipt', 'NEW_RECEIPT_FILE',
            ])
            receipt = json.loads(args.receipt.read_text())
            self.assertEqual(receipt['turn_id'], 'first-turn')
            self.assertEqual(receipt['phase'], 'submitted')
            self.assertEqual(receipt['sent_sha256'], hashlib.sha256(sent.encode()).hexdigest())
            calls.clear()
            with self.assertRaises(FileExistsError):
                asyncio.run(operate(call, args))
            self.assertEqual([method for method, _ in calls], ['thread/read'])

            for changed, value in [('model', 'other-model'), ('reasoningEffort', 'low'),
                                   ('cwd', '/wrong'), ('id', lead_id)]:
                with self.subTest(changed=changed):
                    args.receipt = root / f'bad-{changed}.json'
                    worker[changed] = value
                    calls.clear()
                    with self.assertRaises(ValueError):
                        asyncio.run(operate(call, args))
                    self.assertEqual([method for method, _ in calls],
                                     ['thread/read', 'thread/start'])
                    saved = json.loads(args.receipt.read_text())
                    self.assertEqual(saved['worker_thread_id'], value if changed == 'id' else worker_id)
                    self.assertEqual(saved['worker'][changed], value)
                    worker[changed] = worker_id if changed == 'id' else (str(root) if changed == 'cwd' else
                                      args.model if changed == 'model' else args.effort)

            for phase, expected in [('name', 'created'), ('create_timeout', 'creation_unknown'),
                                    ('send_timeout', 'delivery_unknown')]:
                with self.subTest(phase=phase):
                    failure = phase
                    args.receipt = root / f'{phase}.json'
                    calls.clear()
                    with self.assertRaises((RuntimeError, TimeoutError)):
                        asyncio.run(operate(call, args))
                    saved = json.loads(args.receipt.read_text())
                    self.assertEqual(saved['status'], expected)
                    self.assertEqual(saved['phase'], 'thread/start' if phase == 'create_timeout' else
                                     'thread/name/set' if phase == 'name' else 'turn/start')
                    self.assertEqual(saved.get('worker_thread_id'),
                                     None if phase == 'create_timeout' else worker_id)
                    self.assertEqual([method for method, _ in calls],
                                     ['thread/read'] + (['thread/start'] if phase == 'create_timeout' else
                                      ['thread/start', 'thread/name/set'] if phase == 'name' else
                                      ['thread/start', 'thread/name/set', 'turn/start']))
                    calls.clear()
                    with self.assertRaises(FileExistsError):
                        asyncio.run(operate(call, args))
                    self.assertEqual([method for method, _ in calls], ['thread/read'])

            failure = None
            lead['cwd'] = '/wrong'
            args.receipt = root / 'bad-lead.json'
            calls.clear()
            with self.assertRaisesRegex(ValueError, 'Lead identity or cwd mismatch'):
                asyncio.run(operate(call, args))
            self.assertEqual([method for method, _ in calls], ['thread/read'])
            self.assertFalse(args.receipt.exists())
            lead['cwd'] = str(root)

            for invalid in ('empty', 'lead_target', 'blank_name', 'missing_effort'):
                with self.subTest(invalid=invalid):
                    args.receipt = root / f'{invalid}.json'
                    if invalid == 'empty':
                        message.write_text('  ', encoding='utf-8')
                    elif invalid == 'lead_target':
                        args.to = 'lead'
                    elif invalid == 'blank_name':
                        args.name = '  '
                    else:
                        args.effort = None
                    calls.clear()
                    with self.assertRaises(ValueError):
                        asyncio.run(operate(call, args))
                    self.assertFalse(any(method != 'thread/read' for method, _ in calls))
                    self.assertFalse(args.receipt.exists())
                    message.write_text('Inspect the named artifact; return evidence and stop.', encoding='utf-8')
                    args.to, args.name, args.effort = 'worker', 'bounded-worker', 'xhigh'

    def test_create_cli_requires_complete_explicit_assignment(self):
        script = Path(__file__).with_name('worker_turn.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lead = '00000000-0000-4000-8000-000000000001'
            base = [sys.executable, str(script), '--lead-thread', lead, '--cwd', str(root),
                    '--create', '--send', '--message', str(root / 'message.txt'),
                    '--receipt', str(root / 'receipt.json'), '--name', 'worker',
                    '--model', 'gpt-6-sol', '--effort', 'high']
            for absent in ('--name', '--model', '--effort', '--send', '--message', '--receipt'):
                with self.subTest(absent=absent):
                    command = base.copy()
                    index = command.index(absent)
                    del command[index:index + (1 if absent == '--send' else 2)]
                    self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual(subprocess.run(base + ['--worker-thread', lead],
                                            capture_output=True).returncode, 2)
            self.assertEqual(subprocess.run(base + ['--to', 'lead'],
                                            capture_output=True).returncode, 2)

    def test_existing_pair_rejects_create_only_flags_before_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            lead = '00000000-0000-4000-8000-000000000001'
            worker = '00000000-0000-4000-8000-000000000002'
            base = ['--lead-thread', lead, '--worker-thread', worker, '--cwd', directory]
            self.assertEqual(worker_turn.parse_args(base).worker_thread, worker)
            for flag, value in (('--name', 'worker'), ('--model', 'gpt-6-sol'),
                                ('--effort', 'high')):
                with self.subTest(flag=flag):
                    with self.assertRaises(SystemExit) as error:
                        worker_turn.parse_args(base + [flag, value])
                    self.assertEqual(error.exception.code, 2)

    def test_direct_return_to_active_lead(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            message = root / 'report.txt'
            message.write_text('HOLD: input unavailable. Evidence: /bound/manifest.json.')
            args = SimpleNamespace(lead_thread='lead', worker_thread='worker', cwd=root,
                                   message=message, receipt=root / 'return.json', send=True,
                                   to='lead')
            state = {
                ident: dict(id=ident, cwd=str(root), model='gpt-6-astra',
                            reasoningEffort=effort, status={'type': 'active'})
                for ident, effort in [('lead', 'ultra'), ('worker', 'low')]
            }
            calls = []
            active_status = 'inProgress'
            timeout = False
            resume_override = {}

            async def call(method, params):
                calls.append((method, params))
                if method == 'thread/read':
                    return {'thread': copy.deepcopy(state[params['threadId']])}
                if method == 'thread/resume':
                    state[params['threadId']]['status']['type'] = 'idle'
                    state[params['threadId']].update(resume_override)
                    return {}
                if method == 'thread/turns/list':
                    return {'data': [{'id': params['threadId'] + '-turn', 'status': active_status}],
                            'nextCursor': None}
                if method == 'turn/steer':
                    if timeout:
                        raise TimeoutError('steering response lost')
                    return {'turnId': params['expectedTurnId']}
                if method == 'turn/start':
                    return {'turn': {'id': params['threadId'] + '-turn', 'status': 'inProgress'}}
                raise AssertionError(method)

            result = asyncio.run(operate(call, args))
            self.assertEqual(result['status'], 'submitted')
            self.assertEqual(result['target'], 'lead')
            self.assertEqual(result['turn_id'], 'lead-turn')
            self.assertNotIn('turn/start', [m for m, _ in calls])
            self.assertEqual(next(p for m, p in calls if m == 'turn/steer'), {
                'threadId': 'lead', 'expectedTurnId': 'lead-turn',
                'input': [{'type': 'text', 'text': message.read_text()}],
            })
            self.assertEqual(next(p for m, p in calls if m == 'thread/turns/list'), {
                'threadId': 'lead', 'limit': 1, 'sortDirection': 'desc', 'itemsView': 'notLoaded',
            })

            for role, status in [('lead', 'idle'), ('lead', 'notLoaded'), ('worker', 'active')]:
                with self.subTest(target=role, status=status):
                    args.to = role
                    args.receipt = root / f'{role}-{status}.json'
                    state[role]['status']['type'] = status
                    calls.clear()
                    result = asyncio.run(operate(call, args))
                    method = 'turn/steer' if status == 'active' else 'turn/start'
                    self.assertEqual(result['target_thread_id'], role)
                    self.assertEqual(result['delivery_method'], method)
                    self.assertEqual([m for m, _ in calls if m.startswith('turn/')], [method])
                    self.assertEqual(next(p for m, p in calls if m == method)['threadId'], role)

            args.to = 'lead'
            state['lead']['status']['type'] = 'active'
            args.receipt = root / 'stale-active.json'
            active_status = 'completed'
            calls.clear()
            with self.assertRaisesRegex(ValueError, 'no exact current turn'):
                asyncio.run(operate(call, args))
            self.assertFalse(any(m.startswith('turn/') for m, _ in calls))
            self.assertEqual(json.loads(args.receipt.read_text())['status'], 'prepared')

            active_status = 'inProgress'
            timeout = True
            args.receipt = root / 'unknown-steering.json'
            with self.assertRaises(TimeoutError):
                asyncio.run(operate(call, args))
            self.assertEqual(json.loads(args.receipt.read_text())['status'], 'delivery_unknown')
            calls.clear()
            with self.assertRaises(FileExistsError):
                asyncio.run(operate(call, args))
            self.assertTrue(all(m == 'thread/read' for m, _ in calls))

            state['lead']['status']['type'] = 'notLoaded'
            resume_override = {'reasoningEffort': 'xhigh'}
            args.receipt = root / 'lead-settings-changed.json'
            calls.clear()
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                asyncio.run(operate(call, args))
            self.assertFalse(any(m.startswith('turn/') for m, _ in calls))
            self.assertEqual(json.loads(args.receipt.read_text())['lead']['reasoningEffort'], 'xhigh')

    def test_dispatch_boundaries_and_ambiguous_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            message = root / 'assignment.txt'
            message.write_text('Read the assigned artifact and return a candidate. Stop.')
            args = SimpleNamespace(lead_thread='lead', worker_thread='worker', cwd=root,
                                   message=message, receipt=root / 'dispatch.json', send=False)
            initial = {
                ident: dict(id=ident, cwd=str(root), model='gpt-6-astra',
                            reasoningEffort=effort, status={'type': 'idle'})
                for ident, effort in [('lead', 'low'), ('worker', 'xhigh')]
            }
            initial['worker']['model'] = 'gpt-6-sol'
            state, calls = copy.deepcopy(initial), []
            timeout = False
            resume_override = {}

            async def call(method, params):
                calls.append((method, params))
                if method == 'thread/read':
                    return {'thread': copy.deepcopy(state[params['threadId']])}
                if method == 'thread/resume':
                    state['worker']['status']['type'] = 'idle'
                    state['worker'].update(resume_override)
                    return {}
                if method == 'turn/start':
                    if timeout:
                        raise TimeoutError('response lost')
                    return {'turn': {'id': 'turn-1', 'status': 'inProgress'}}
                raise AssertionError(method)

            result = asyncio.run(operate(call, args))
            self.assertEqual(result['status'], 'inspected_only')
            self.assertTrue(all(m == 'thread/read' for m, _ in calls))
            self.assertFalse(args.receipt.exists())
            args.send = True
            for who, key, value in [('worker', 'cwd', '/wrong'),
                                    ('worker', 'id', 'wrong'),
                                    ('worker', 'canAcceptDirectInput', False),
                                    ('worker', 'status', {'type': 'systemError'})]:
                state, calls = copy.deepcopy(initial), []
                state[who][key] = value
                with self.assertRaises(ValueError):
                    asyncio.run(operate(call, args))
                self.assertFalse(args.receipt.exists())
                self.assertTrue(all(m == 'thread/read' for m, _ in calls))

            state, calls = copy.deepcopy(initial), []
            state['worker']['status']['type'] = 'notLoaded'
            result = asyncio.run(operate(call, args))
            self.assertEqual(result['status'], 'submitted')
            self.assertEqual(json.loads(args.receipt.read_text())['turn_id'], 'turn-1')
            self.assertEqual([m for m, _ in calls if m != 'thread/read'],
                             ['thread/resume', 'turn/start'])
            params = next(p for m, p in calls if m == 'turn/start')
            self.assertEqual(params, {'threadId': 'worker', 'input': [
                {'type': 'text', 'text': message.read_text()}]})
            calls.clear()
            with self.assertRaises(FileExistsError):
                asyncio.run(operate(call, args))
            self.assertTrue(all(m == 'thread/read' for m, _ in calls))

            args.receipt = root / 'unknown.json'
            timeout = True
            with self.assertRaises(TimeoutError):
                asyncio.run(operate(call, args))
            self.assertEqual(json.loads(args.receipt.read_text())['status'], 'delivery_unknown')
            calls.clear()
            with self.assertRaises(FileExistsError):
                asyncio.run(operate(call, args))
            self.assertTrue(all(m == 'thread/read' for m, _ in calls))


            for key, value in [('model', 'gpt-6-luna'), ('reasoningEffort', 'low')]:
                with self.subTest(changed_setting=key):
                    state, calls = copy.deepcopy(initial), []
                    state['worker']['status']['type'] = 'notLoaded'
                    resume_override = {key: value}
                    timeout = False
                    args.receipt = root / f'{key}-changed.json'
                    with self.assertRaisesRegex(ValueError, 'settings changed'):
                        asyncio.run(operate(call, args))
                    self.assertEqual([m for m, _ in calls if m != 'thread/read'],
                                     ['thread/resume'])
                    saved = json.loads(args.receipt.read_text())
                    self.assertEqual(saved['status'], 'prepared')
                    self.assertEqual(saved['worker'][key], value)
                    self.assertEqual(saved['worker']['status'], {'type': 'idle'})
                    self.assertEqual(saved['error_type'], 'ValueError')


if __name__ == '__main__':
    unittest.main()
