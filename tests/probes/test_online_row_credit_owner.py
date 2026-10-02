"""Actual owner boundary, with synthetic /proc and deterministic CPU dependencies."""
import contextlib
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from probes import online_row_credit_owner as owner


CWD = Path(__file__).resolve().parents[2]
EVIDENCE = CWD / 'outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/cpu-01'


class Boundary:
    """Fixtures replace dependencies, never the owner's deadline computation."""
    def __init__(self, root, case='success'):
        self.root, self.case, self.clock = root, case, 0.0
        self.mode, self.updates, self.total, self.cutoff = 'paired1', None, 2700, 2670
        self.waits, self.launches, self.signals = [], [], []
        self.pids = {1000: dict(ppid=999, state='S')}
        self.child_waits = 0
        self.finish = False
        self.release = dict(total_wall_ceiling_seconds=2700, source_commit='fixture-source',
                            lead_thread='fixture-lead', worker_thread='fixture-worker',
                            bindings={'probes/online_row_credit_owner.py': self.digest(Path(owner.__file__))})
        self.dump(root/'lead-release.json', self.release)
        self.release_sha = self.digest(root/'lead-release.json')
        self.commands = [dict(arm=a, stage=s, argv=['python', '-c', 'pass']) for a, s in
                         [('control', 'run'), ('control', 'readback'), ('treatment', 'run'),
                          ('treatment', 'readback'), ('control', 'offline'), ('treatment', 'offline')]]
        self.dump(root/'argv.json', self.commands)
        self.dump(root/'qualification.json', dict(source=dict(files=[]), sha256={}, test_source_sha256={},
                   runtime={}, pairs={'1': {a: str(root/a) for a in ('control', 'treatment')}}))

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def dump(path, value):
        path.write_text(json.dumps(value))

    def timed(self, kind, timeout, advance=0):
        self.waits.append(dict(kind=kind, timeout=timeout, at=self.clock))
        self.clock += advance

    def path(self, value):
        if str(value).startswith('/proc'):
            return Proc(self, str(value))
        return Path(value)

    def check_output(self, argv, **kwargs):
        self.timed('guard', kwargs.get('timeout'))
        if self.case == 'guard_expiry':
            self.clock = self.cutoff
        return 'fixture-source' if argv[-1] == 'HEAD' else b''

    def kill(self, pid, sig):
        self.signals.append((pid, sig))
        if self.case != 'survivor':
            self.pids[pid]['state'] = 'Z'

    def popen(self, argv, **kwargs):
        self.launches.append(dict(argv=argv, at=self.clock))
        pid = 2000+len(self.launches)
        self.pids[pid] = dict(ppid=1000, state='S')
        if self.case == 'stale_wait':
            self.clock = self.cutoff-1
        if self.case == 'post_launch_expiry':
            self.clock = self.cutoff
        boundary = self

        class Child:
            returncode = None

            def wait(self, timeout):
                boundary.child_waits += 1
                advance = 0
                if boundary.case in ('expiry', 'reap_timeout', 'unresolved','cleanup_edge','full_expiry'):
                    if boundary.child_waits == 1:
                        advance = timeout
                    elif boundary.case in ('reap_timeout','cleanup_edge') and boundary.child_waits == 2:
                        advance = timeout
                    elif boundary.case == 'unresolved':
                        advance = timeout
                    else:
                        self.returncode = -9
                else:
                    self.returncode = 7 if boundary.case == 'nonzero' else 0
                    advance = min(1, timeout)
                boundary.timed('child', timeout, advance)
                if boundary.case == 'cleanup_edge' and boundary.child_waits == 1:
                    boundary.clock = boundary.total-1
                if self.returncode is None:
                    raise subprocess.TimeoutExpired(argv, timeout)
                boundary.pids[pid]['state'] = 'Z' if boundary.case != 'survivor' else 'S'
                if boundary.case == 'overrun':
                    boundary.clock = boundary.total+1
                return self.returncode

        child = Child()
        child.pid = pid
        return child

    def thread(self, target, daemon):
        boundary = self

        class Thread:
            def start(self):
                if boundary.case == 'watch':
                    boundary.clock = boundary.cutoff-1
                    target()

            def join(self, timeout):
                boundary.timed('join', timeout)

            def is_alive(self):
                return boundary.case == 'unfinished_watcher'

        return Thread()

    def event(self):
        boundary = self

        class Event:
            def is_set(self):
                return boundary.finish

            def set(self):
                boundary.finish = True

            def wait(self, timeout):
                boundary.timed('event', timeout, 0 if boundary.finish else timeout)
                if boundary.case == 'watch':
                    boundary.finish = True

        return Event()

    def install_gate(self):
        for arm in ('control', 'treatment'):
            directory = self.root/arm
            directory.mkdir(exist_ok=True)
            records = []
            for version in (0, 1):
                d = directory/f'rollout-{version}'
                d.mkdir(exist_ok=True)
                self.dump(d/'frozen.json', dict(CPU_fixture=True, update=version))
                records.append(dict(update=version, requests=18, frozen_sha256=self.digest(d/'frozen.json')))
            self.dump(directory/'readback.json', records)

    def install_full_gate(self):
        self.run_output.mkdir(exist_ok=True)
        records=[]
        for update in range(self.updates+1):
            d=self.run_output/f'rollout-{update}';d.mkdir(exist_ok=True)
            self.dump(d/'frozen.json',dict(CPU_fixture=True,update=update))
            records.append(dict(update=update,requests=18,frozen_sha256=self.digest(d/'frozen.json')))
        self.dump(self.run_output/'readback.json',records)
        request=self.run_output/'rollout-0/rank-0';request.mkdir(parents=True,exist_ok=True)
        self.dump(request/'1.json',dict(generated_tokens=17))
        rank=self.run_output/'rank-0';rank.mkdir(exist_ok=True)
        self.dump(rank/'update-0.json',dict(forwards=[dict(tokens=5032,visual_tokens=1024)]))

    def full_label(self, updates=2):
        self.mode,self.updates='full-label',updates
        self.total=900 if updates==2 else 2700;self.cutoff=self.total-30
        state=json.loads((CWD/'research/experiments/2026-10-02-full-label-self-rollout-fit/state.json').read_text())
        prefix=self.root.name
        self.run_output=self.root.parent/f'{prefix}-native-qualification-{updates:02d}'
        qdir=self.root.parent/f'{prefix}-qualification';qdir.mkdir()
        self.canonical_qual=qdir/'qualification.json'
        self.qual=dict(schema='full-label-self-rollout-qualification-v1',
            runs={'qualification':str(self.run_output),'observation':str(self.run_output)},
            correction=state['recipe'],source=dict(files=[]),sha256={},
            decoder_runtime_identity=dict(distribution='vllm',version='fixture-vllm',source_sha256={}),
            pairs={str(updates):{'treatment':str(self.run_output)}})
        self.canonical_qual.write_text(json.dumps(self.qual,sort_keys=True,separators=(',',':'))+'\n')
        (self.root/'qualification.json').write_bytes(self.canonical_qual.read_bytes())
        self.commands=[]
        for stage in ('run','readback','offline'):
            command=list(state['argv'][str(updates)][stage])
            for flag,value in (('--output',str(self.run_output)),('--root',str(qdir))):
                command[command.index(flag)+1]=value
            self.commands.append(dict(arm='treatment',stage=stage,argv=command))
        self.dump(self.root/'argv.json',self.commands)
        tests={'tests/probes/test_online_row_credit_owner.py':self.digest(Path(__file__))}
        self.release=dict(mode='full-label',updates=updates,total_wall_ceiling_seconds=self.total,
            source_commit='fixture-source',lead_thread='fixture-lead',worker_thread='fixture-worker',
            owner_root=str(self.root.resolve()),run_output=str(self.run_output.resolve()),
            qualification_path=str(self.canonical_qual.resolve()),
            argv_sha256=self.digest(self.root/'argv.json'),qualification_sha256=self.digest(self.root/'qualification.json'),
            recipe_sha256=hashlib.sha256(json.dumps(self.qual['correction'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
            runtime={k:'fixture-'+k for k in ('torch','transformers','vllm')},test_source_sha256=tests,
            bindings={'probes/online_row_credit_owner.py':self.digest(Path(owner.__file__))})
        self.dump(self.root/'lead-release.json',self.release)
        self.release_sha=self.digest(self.root/'lead-release.json')

    def run(self):
        original_open = Path.open
        boundary = self

        class FixturePath:
            def __new__(cls, value):
                return boundary.path(value)

            @staticmethod
            def cwd():
                return CWD

        def fixture_open(path, *args, **kwargs):
            if path.name.startswith('evaluator-gate') and boundary.case == 'stale_launch':
                boundary.clock = boundary.cutoff
            if path.name == 'terminal.json' and boundary.case == 'finalization_overrun':
                boundary.clock = boundary.total+2
            return original_open(path, *args, **kwargs)

        def fixture_popen(argv, **kwargs):
            if boundary.mode=='full-label':
                row=boundary.commands[len(boundary.launches)]
                if row['stage']=='run':boundary.run_output.mkdir()
                if row['stage']=='readback':boundary.install_full_gate()
            elif len(boundary.launches) == 3:
                boundary.install_gate()
            return boundary.popen(argv, **kwargs)

        with contextlib.ExitStack() as stack:
            for obj, name, replacement in [
                (owner, 'Path', FixturePath), (owner.os, 'getpid', lambda: 1000),
                (owner.os, 'kill', self.kill), (owner.os, 'waitpid', lambda *a: (0, 0)),
                (owner.signal, 'signal', lambda *a: None), (owner.ctypes, 'CDLL', lambda *a: type('Lib', (), {'prctl': lambda *a: 0})()),
                (owner.shutil, 'which', lambda *a: owner.sys.executable), (owner.time, 'monotonic', lambda: self.clock),
                (owner.subprocess, 'check_output', self.check_output), (owner.subprocess, 'Popen', fixture_popen),
                (owner, 'version', lambda name:'fixture-'+name),
                (owner.threading, 'Thread', self.thread), (owner.threading, 'Event', self.event), (Path, 'open', fixture_open),
            ]:
                stack.enter_context(patch.object(obj, name, replacement))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = owner.run(self.root, self.release_sha,self.mode,self.updates)
        terminal = json.loads((self.root/'terminal.json').read_text())
        final = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')][-1]
        self.owner_receipt=json.loads((self.root/'owner.json').read_text())
        self.gate_receipt=json.loads((self.root/'evaluator-gate-2.json').read_text()) if (self.root/'evaluator-gate-2.json').exists() else None
        self.cost_receipt=json.loads((self.root/'stage-2-treatment-offline-cost.json').read_text()) if (self.root/'stage-2-treatment-offline-cost.json').exists() else None
        self.no_control_output=not (self.root/'control').exists()
        return code, terminal, final


class Proc:
    def __init__(self, boundary, value):
        self.b, self.value = boundary, value

    def __truediv__(self, name):
        return Proc(self.b, self.value+'/'+str(name))

    @property
    def name(self):
        return self.value.rsplit('/', 1)[-1]

    def exists(self):
        parts = self.value.split('/')
        return len(parts)>2 and parts[2].isdigit() and int(parts[2]) in self.b.pids

    def iterdir(self):
        return [Proc(self.b, '/proc/'+str(pid)) for pid in self.b.pids]

    def read_text(self):
        if self.value.endswith('boot_id'):
            return 'fixture-boot'
        parts = self.value.split('/');pid = int(parts[2]);record = self.b.pids[pid]
        if self.b.case == 'unconfirmed' and pid != 1000 and record['state'] == 'Z':
            raise PermissionError('owned CPU fixture observation unavailable')
        if self.name == 'stat':
            fields = [record['state'], str(record['ppid']), str(pid)]+['0']*19
            fields[19] = str(pid*100)
            return str(pid)+' (CPU fixture) '+' '.join(fields)
        return f"VmRSS:\t1 kB\nVmHWM:\t1 kB\nNSpid:\t{pid}\n"

    def read_bytes(self):
        return b'CPU-fixture\0'


class OwnerTest(unittest.TestCase):
    def exercise(self, case, full_updates=None):
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            boundary = Boundary(Path(directory), case)
            if full_updates is not None:boundary.full_label(full_updates)
            code, terminal, final = boundary.run()
            if os.environ.get('OWNER_CPU_EVIDENCE'):
                evidence = dict(case=case,code=code,terminal=terminal,final=final,waits=boundary.waits,
                                launches=boundary.launches,signals=boundary.signals,CPU_fixture_only=True)
                p=Path(os.environ['OWNER_CPU_EVIDENCE'])/(self.id().rsplit('.',1)[-1]+'-'+case+'.json')
                p.write_text(json.dumps(evidence,indent=2,sort_keys=True))
            return boundary, code, terminal, final

    def bounded(self, boundary):
        for wait in boundary.waits:
            self.assertIsNotNone(wait['timeout'], wait)
            ceiling = boundary.cutoff if wait['kind'] == 'guard' or wait['kind']=='child' and wait['at']<boundary.cutoff else boundary.total
            self.assertGreaterEqual(wait['timeout'], 0, wait)
            self.assertLessEqual(wait['timeout'], max(0, ceiling-wait['at']), wait)

    def test_full_label_two_and_sixteen_complete_one_treatment_only(self):
        for updates in (2,16):
            with self.subTest(updates=updates):
                b,code,terminal,final=self.exercise('success',full_updates=updates)
                self.bounded(b)
                self.assertEqual(code,0,terminal)
                self.assertEqual([(x['arm'],x['stage']) for x in terminal['issued_stages']],
                                 [('treatment','run'),('treatment','readback'),('treatment','offline')])
                self.assertEqual([x['argv'] for x in terminal['issued_stages']], [x['argv'] for x in b.commands])
                self.assertEqual(b.owner_receipt['deadline_wall_seconds'],900 if updates==2 else 2700)
                self.assertEqual(b.owner_receipt['execution_cutoff_seconds'],870 if updates==2 else 2670)
                self.assertEqual(b.owner_receipt['mode'],'full-label')
                self.assertEqual(b.owner_receipt['updates'],updates)
                self.assertEqual(b.gate_receipt['arms'].keys(),{'treatment'})
                self.assertEqual(sorted(map(int,b.gate_receipt['arms']['treatment']['freezes'])),list(range(updates+1)))
                self.assertEqual(b.cost_receipt,dict(requests=1,generated_tokens=17,HF_forwards=1,HF_input_tokens=5032,HF_visual_tokens=1024))
                self.assertTrue(b.no_control_output)
                self.assertEqual(final['gpu_hours'],final['elapsed']*8/3600)

    def test_full_label_first_failure_skips_remaining_treatment_stages(self):
        b,code,terminal,_=self.exercise('nonzero',full_updates=2)
        self.assertEqual(code,1)
        self.assertEqual([(x['arm'],x['stage']) for x in terminal['issued_stages']],[('treatment','run')])
        self.assertEqual([(x['arm'],x['stage']) for x in terminal['skipped_stages']],
                         [('treatment','readback'),('treatment','offline')])

    def test_full_label_two_update_cleanup_deadline_and_finalization_charge(self):
        b,code,terminal,final=self.exercise('full_expiry',full_updates=2)
        self.bounded(b)
        self.assertEqual(code,1)
        self.assertEqual(next(x['timeout'] for x in b.waits if x['kind']=='child'),870)
        self.assertEqual(terminal['issued_stages'][0]['exit_code'],-9)
        self.assertEqual(len(terminal['skipped_stages']),2)
        self.assertLessEqual(final['elapsed'],900)
        self.assertEqual(final['gpu_hours'],final['elapsed']*8/3600)
        b,code,_,final=self.exercise('finalization_overrun',full_updates=2)
        self.assertEqual(code,1)
        self.assertGreater(final['elapsed'],900)
        self.assertGreater(final['gpu_hours'],2)
        self.assertEqual(final['gpu_hours'],final['elapsed']*8/3600)
        self.assertLess(final['cleanup_completion_elapsed'],900)
        self.assertGreater(final['receipt_finalization_seconds'],0)

    def test_full_label_release_mode_updates_and_argv_paths_reject_before_launch(self):
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            b=Boundary(Path(directory));b.full_label(2);b.updates=16
            with self.assertRaises(AssertionError):b.run()
            self.assertEqual(b.launches,[])
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            b=Boundary(Path(directory));b.full_label(2);b.mode='paired1';b.updates=None
            with self.assertRaises(AssertionError):b.run()
            self.assertEqual(b.launches,[])
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            b=Boundary(Path(directory));b.full_label(2)
            command=b.commands[0]['argv'];command[command.index('--output')+1]=str(b.root/'drifted-run')
            b.dump(b.root/'argv.json',b.commands)
            b.release['argv_sha256']=b.digest(b.root/'argv.json')
            b.dump(b.root/'lead-release.json',b.release);b.release_sha=b.digest(b.root/'lead-release.json')
            with self.assertRaises(AssertionError):b.run()
            self.assertEqual(b.launches,[])

    def test_expiry_and_delayed_reap_bound_every_wait(self):
        for case in ('expiry', 'reap_timeout','cleanup_edge'):
            with self.subTest(case=case):
                b, code, terminal, final = self.exercise(case)
                self.assertLessEqual(next(x['timeout'] for x in b.waits if x['kind']=='child'),2670)
                self.bounded(b)
                self.assertEqual(code, 1)
                self.assertEqual(len(b.launches), 1)
                self.assertEqual(terminal['issued_stages'][0]['exit_code'], -9)
                self.assertEqual(len(terminal['skipped_stages']), 5)
                self.assertLessEqual(final['elapsed'],2700)

    def test_stale_prelaunch_and_postlaunch_wait_are_recomputed(self):
        b,code,terminal,_=self.exercise('stale_launch')
        self.assertEqual(code,1)
        self.assertEqual(len(b.launches),4)
        self.assertTrue(all(x['at']<2670 for x in b.launches))
        b,_,_,_=self.exercise('stale_wait')
        self.bounded(b)
        self.assertEqual(next(x['timeout'] for x in b.waits if x['kind']=='child'),1)

    def test_owned_survivor_and_unfinished_watcher_fail(self):
        for case in ('survivor','unfinished_watcher','unconfirmed'):
            with self.subTest(case=case):
                b,code,terminal,_=self.exercise(case)
                self.assertEqual(code,1)
                self.assertEqual(terminal['status'],'failed')
                self.assertTrue(terminal['owned_live_after_cleanup'] or terminal['watcher_unfinished'] or terminal['unconfirmed_owned'])
                self.assertTrue(all(pid in b.pids and pid!=1000 for pid,_ in b.signals))
                if case=='unconfirmed':self.assertEqual(len(b.launches),1)

    def test_overrun_including_finalization_keeps_uncapped_charge(self):
        for case in ('overrun','finalization_overrun'):
            with self.subTest(case=case):
                _,code,_,final=self.exercise(case)
                self.assertEqual(code,1)
                self.assertGreater(final['elapsed'],2700)
                self.assertGreater(final['gpu_hours'],6)
                self.assertEqual(final['gpu_hours'],final['elapsed']*8/3600)
                if case=='finalization_overrun':
                    self.assertLess(final['cleanup_completion_elapsed'],2700)
                    self.assertGreater(final['receipt_finalization_seconds'],0)

    def test_unresolved_issued_owner_is_not_skipped_or_invented_exit(self):
        b,code,terminal,_=self.exercise('unresolved')
        self.bounded(b)
        self.assertEqual(code,1)
        self.assertEqual(len(terminal['issued_stages']),1)
        self.assertEqual(terminal['issued_stages'][0]['status'],'issued')
        self.assertIsNone(terminal['issued_stages'][0]['exit_code'])
        self.assertEqual(len(terminal['unresolved_issued_stages']),1)
        self.assertEqual(len(terminal['skipped_stages']),5)

    def test_first_nonzero_and_success_bookkeeping(self):
        for case, expected, count in [('nonzero',1,1),('success',0,6)]:
            with self.subTest(case=case):
                b,code,terminal,final=self.exercise(case)
                self.bounded(b)
                self.assertEqual(code,expected)
                self.assertEqual(len(terminal['issued_stages']),count)
                self.assertEqual(len(terminal['completed_stages']),count)
                self.assertEqual(len(terminal['skipped_stages']),6-count)
                self.assertEqual(terminal['issued_stages'][0]['exit_code'],7 if case=='nonzero' else 0)
                self.assertEqual(final['gpu_hours'],final['elapsed']*8/3600)

    def test_watcher_wait_uses_execution_remainder(self):
        b,code,_,_=self.exercise('watch')
        self.bounded(b)
        self.assertEqual(code,1)
        self.assertEqual(len(b.launches),0)
        self.assertEqual(next(x['timeout'] for x in b.waits if x['kind']=='event'),1)

    def test_post_launch_cutoff_preserves_issued_stage(self):
        b,code,terminal,_=self.exercise('post_launch_expiry')
        self.bounded(b)
        self.assertEqual(code,1)
        self.assertEqual(len(b.launches),1)
        self.assertEqual(len(terminal['issued_stages']),1)
        self.assertEqual(len(terminal['skipped_stages']),5)

    def test_guard_expiry_does_not_launch_or_issue_a_stage(self):
        b,code,terminal,_=self.exercise('guard_expiry')
        self.bounded(b)
        self.assertEqual(code,1)
        self.assertEqual(len(b.launches),0)
        self.assertEqual(len(terminal['issued_stages']),0)
        self.assertEqual(len(terminal['skipped_stages']),6)

    def test_short_real_CPU_child_entry_receipts_and_owned_only_observation(self):
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            root=Path(directory);b=Boundary(root)
            commands=b.commands
            commands[0]['argv']=['python','-c','import time,sys;time.sleep(.2);sys.exit(7)']
            b.dump(root/'argv.json',commands)
            actual_popen=subprocess.Popen
            handles=[];observed=[]
            parent=os.getpid()

            class OwnProcDirectory:
                def __truediv__(self,name):
                    self_pids={parent,*[x.pid for x in handles]}
                    if str(name).isdigit():
                        if int(name) not in self_pids:raise AssertionError('unowned /proc observation')
                    return Path('/proc')/str(name)

                def iterdir(self):
                    # No machine inventory: observe only this test parent and exact Popen handles.
                    pids=[parent]+[x.pid for x in handles]
                    observed.extend(pids)
                    return [Path('/proc')/str(pid) for pid in pids]

            class FixturePath:
                def __new__(cls,value):
                    return OwnProcDirectory() if str(value)=='/proc' else Path(value)

                @staticmethod
                def cwd():
                    return CWD

            def popen(argv,**kwargs):
                child=actual_popen(argv,**kwargs);handles.append(child);return child

            original_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
            subreaper=ctypes.c_int()
            self.assertEqual(ctypes.CDLL(None).prctl(37,ctypes.byref(subreaper),0,0,0),0)
            output=io.StringIO()
            try:
                with patch.object(owner,'Path',FixturePath),patch.object(owner.subprocess,'Popen',popen),patch.object(owner.subprocess,'check_output',b.check_output),patch.object(owner.sys,'argv',['online-row-credit-owner','--root',str(root),'--release-sha256',b.release_sha]),contextlib.redirect_stdout(output):
                    code=owner.main()
            finally:
                for sig,handler in original_handlers.items():signal.signal(sig,handler)
                ctypes.CDLL(None).prctl(36,subreaper.value,0,0,0)
                for child in handles:
                    if child.poll() is None:
                        child.kill();child.wait(timeout=5)
            terminal=json.loads((root/'terminal.json').read_text())
            final=json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(code,1)
            self.assertEqual(len(handles),1)
            self.assertEqual(handles[0].returncode,7)
            self.assertEqual(terminal['issued_stages'][0]['exit_code'],7)
            self.assertEqual(len(terminal['skipped_stages']),5)
            self.assertEqual(terminal['owned_live_after_cleanup'],[])
            child=terminal['issued_stages'][0]['child']
            self.assertEqual(child['pid'],handles[0].pid)
            self.assertEqual(child['ppid'],parent)
            self.assertGreater(child['start_ticks'],0)
            self.assertTrue(child['boot_id'] and child['raw_nspid'])
            self.assertTrue(set(observed)<={parent,handles[0].pid})
            self.assertEqual(final['gpu_hours'],final['elapsed']*8/3600)
            if os.environ.get('OWNER_CPU_EVIDENCE'):
                evidence=dict(CPU_child_only=True,terminal=terminal,final=final,child_returncode=handles[0].returncode,
                              observed_PIDs=sorted(set(observed)),stdout=output.getvalue(),source_git_checks_mocked=True,
                              note='Actual maintained main/Popen/proc serialization/persistence/exit/cleanup; no broad process inventory or model/native command')
                (Path(os.environ['OWNER_CPU_EVIDENCE'])/'real-CPU-entry.json').write_text(json.dumps(evidence,indent=2,sort_keys=True))


if __name__=='__main__':
    unittest.main()
