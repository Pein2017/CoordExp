"""CPU-only consumer checks; never contacts a real task."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from worker_turn import operate


class WorkerTurnTest(unittest.TestCase):
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
