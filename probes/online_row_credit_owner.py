"""Sequential online-row-credit owner; a separate exact lead release is required."""
import argparse
import os,sys,time,json,hashlib,subprocess,signal,threading,fcntl,ctypes,shutil
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version


def run(root, release_sha256):
    start=time.monotonic()
    root=Path(root);cwd=Path(__file__).resolve().parents[1]
    assert Path.cwd().resolve()==cwd==Path('/data/CoordExp/.worktrees/research-probes')
    assert root.is_absolute() and root.resolve().is_relative_to(cwd/'outputs')
    assert len(release_sha256)==64 and all(c in '0123456789abcdef' for c in release_sha256)
    def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write(name,value):
        with (root/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    def proc(pid):
        d=Path('/proc')/str(pid)
        try:
            s=(d/'stat').read_text().rsplit(')',1)[1].split()
            status={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in (d/'status').read_text().splitlines() if ':' in x}
            return dict(pid=pid,ppid=int(s[1]),state=s[0],start_ticks=int(s[19]),pgid=int(s[2]),
                        rss_kib=int(status.get('VmRSS','0 kB').split()[0]),hwm_kib=int(status.get('VmHWM','0 kB').split()[0]),
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        raw_nspid=next((x for x in (d/'status').read_text().splitlines() if x.startswith('NSpid:')),None),
                        argv=[x.decode(errors='replace') for x in (d/'cmdline').read_bytes().split(b'\0') if x])
        except (FileNotFoundError,PermissionError,ProcessLookupError):return None
    lock=(root/'owner.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0 # Linux subreaper keeps this job's orphaned children attached.
    assert shutil.which('python')==sys.executable,(shutil.which('python'),sys.executable)
    release=json.loads((root/'lead-release.json').read_text());qual=json.loads((root/'qualification.json').read_text())
    argv=json.loads((root/'argv.json').read_text())
    assert [(x['arm'],x['stage']) for x in argv]==[('control','run'),('control','readback'),('treatment','run'),('treatment','readback'),('control','offline'),('treatment','offline')]
    assert release['total_wall_ceiling_seconds']==2700
    total_deadline=start+2700;deadline=total_deadline-30
    def execution_remaining():
        seconds=deadline-time.monotonic()
        if seconds<=0:raise TimeoutError('execution_cutoff_2670_seconds')
        return seconds
    def cleanup_remaining(cap):
        return max(0,min(cap,total_deadline-time.monotonic()))
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    owner=proc(os.getpid());known={};peak_rss=0;peak_pid_rss={};why=[];finished=threading.Event()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name)==os.getpid():continue
        v=proc(int(d.name))
        if v and any(x in v['argv'] for x in ('probes.online_row_credit','probes.online_row_credit_owner','paired1-native-owner')) and str(root) in v['argv']:
            raise RuntimeError('matching live owner: '+str(v['pid']))
    for arm in ('control','treatment'):assert not (root/arm).exists()
    write('owner.json',dict(owner=owner,boot_id=boot,started_utc=datetime.now(timezone.utc).isoformat(),deadline_wall_seconds=2700,
        execution_cutoff_seconds=2670,cleanup_reserve_seconds=30,gpu_slots=8,
        lead_thread=release['lead_thread'],worker_thread=release['worker_thread'],source_commit=release['source_commit'],
        release_sha256=digest(root/'lead-release.json'),argv_sha256=digest(root/'argv.json'),python=sys.executable,
        owner_source_sha256=digest(__file__),
        direct_return=['python','/data/CoordExp/.codex/skills/lead-worker/scripts/worker_turn.py','--lead-thread',release['lead_thread'],
                       '--worker-thread',release['worker_thread'],'--cwd',str(cwd),'--to','lead','--send','--message','REPORT_FILE','--receipt','NEW_RECEIPT_FILE']))
    def children():
        nonlocal peak_rss
        allp={}
        for d in Path('/proc').iterdir():
            if d.name.isdigit():
                v=proc(int(d.name))
                if v:allp[v['pid']]=v
        ids={os.getpid()}
        while True:
            more={pid for pid,v in allp.items() if v['ppid'] in ids}
            if more<=ids:break
            ids|=more
        rows=[v for pid,v in allp.items() if pid in ids and pid!=os.getpid()]
        for v in rows:
            key=(v['pid'],v['start_ticks'])
            if key not in known:
                known[key]=v
                with (root/'process-identities.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
            peak_pid_rss[key]=max(peak_pid_rss.get(key,0),v['hwm_kib'])
        peak_rss=max(peak_rss,sum(v['rss_kib'] for v in rows))
        return rows
    def unconfirmed_owned():
        result=[]
        for (pid,ticks),row in list(known.items()):
            now=proc(pid)
            if now is None and (Path('/proc')/str(pid)).exists():result.append(row)
            elif now and now['start_ticks']==ticks and now['boot_id']!=row['boot_id']:result.append(row)
        return result
    def cleanup(hard=False):
        rows=children()
        for v in reversed(rows):
            now=proc(v['pid'])
            if now and now['start_ticks']==v['start_ticks'] and now['boot_id']==v['boot_id'] and now['state']!='Z':
                try:os.kill(v['pid'],signal.SIGKILL if hard else signal.SIGTERM)
                except ProcessLookupError:pass
        if not hard:
            finished.wait(cleanup_remaining(3))
            for v in children():
                now=proc(v['pid'])
                if now and now['start_ticks']==v['start_ticks'] and now['boot_id']==v['boot_id'] and now['state']!='Z':
                    try:os.kill(v['pid'],signal.SIGKILL)
                    except ProcessLookupError:pass
    def stop(signum,frame):
        why.append('owner_signal_'+str(signum));raise InterruptedError(why[-1])
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    offsets={}
    def watch():
        while not finished.is_set():
            rows=children()
            with (root/'rss-samples.jsonl').open('a') as f:
                f.write(json.dumps(dict(elapsed=time.monotonic()-start,aggregate_rss_kib=sum(x['rss_kib'] for x in rows),
                                       processes=[{k:x[k] for k in ('pid','ppid','start_ticks','state','rss_kib','hwm_kib')} for x in rows]))+'\n')
            logs=list(root.glob('stage-*.log'))+list((root).glob('*/rank-*/vllm.log'))
            for path in logs:
                try:
                    with path.open('rb') as f:f.seek(offsets.get(str(path),0));new=f.read();offsets[str(path)]=f.tell()
                    if any(s in new for s in (b'CUDA out of memory',b'OutOfMemoryError',b'Traceback (most recent call last):')):
                        why.append('observed_failure_log:'+str(path));cleanup();return
                except FileNotFoundError:pass
            if time.monotonic()>=deadline:
                why.append('execution_cutoff_2670_seconds');cleanup(hard=True);return
            finished.wait(max(0,min(15,deadline-time.monotonic())))
    def guard():
        assert digest(root/'lead-release.json')==release_sha256
        assert release['bindings']['probes/online_row_credit_owner.py']==digest(__file__)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True,timeout=execution_remaining()).strip()==release['source_commit']
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=cwd,timeout=execution_remaining())
        for path,sha in release['bindings'].items():assert digest(cwd/path if not Path(path).is_absolute() else path)==sha,path
        for row in qual['source']['files']:assert digest(cwd/row['path'])==row['sha256'],row['path']
        for path,sha in qual['sha256'].items():assert digest(path)==sha,path
        for path,sha in qual['test_source_sha256'].items():assert digest(cwd/path)==sha,path
        assert {k:version(k) for k in qual['runtime']}==qual['runtime']
    def costs():
        result=dict(requests=0,generated_tokens=0,HF_forwards=0,HF_input_tokens=0,HF_visual_tokens=0)
        for arm in ('control','treatment'):
            out=root/arm
            for path in out.glob('rollout-*/rank-*/*.json'):
                if path.name=='complete.json':continue
                try:r=json.loads(path.read_text())
                except json.JSONDecodeError:continue
                result['requests']+=1;result['generated_tokens']+=r['generated_tokens']
            for path in out.glob('rank-*/update-*.json'):
                for r in json.loads(path.read_text())['forwards']:
                    result['HF_forwards']+=1;result['HF_input_tokens']+=r['tokens'];result['HF_visual_tokens']+=r['visual_tokens']
        limits={'requests':72,'generated_tokens':222048,'HF_forwards':6210,'HF_input_tokens':99360000,'HF_visual_tokens':6359040}
        assert all(result[k]<=limit for k,limit in limits.items()),result
        return result
    print(json.dumps(dict(event='owner_started',pid=os.getpid(),start_ticks=owner['start_ticks'],root=str(root))),flush=True)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    receipts=[];current=None;status='failed';error=None
    try:
        for index,row in enumerate(argv):
            guard()
            if why:raise RuntimeError(why[-1])
            execution_remaining()
            if row['stage']=='offline':
                gate={}
                for arm,path in qual['pairs']['1'].items():
                    out=Path(path);read=json.loads((out/'readback.json').read_text())
                    assert [v['update'] for v in read]==[0,1]
                    assert all(v['requests']==18 for v in read)
                    freezes={}
                    for v in read:
                        frozen=out/f"rollout-{v['update']}"/'frozen.json'
                        assert digest(frozen)==v['frozen_sha256']
                        freezes[str(v['update'])]=digest(frozen)
                    gate[arm]=dict(readback_sha256=digest(out/'readback.json'),freezes=freezes)
                write(f'evaluator-gate-{index}.json',dict(verified_utc=datetime.now(timezone.utc).isoformat(),arms=gate))
            label=f"stage-{index}-{row['arm']}-{row['stage']}"
            before=time.monotonic()
            with (root/(label+'.log')).open('xb') as log:
                execution_remaining()
                current=subprocess.Popen(row['argv'],cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                receipt=dict(index=index,arm=row['arm'],stage=row['stage'],argv=row['argv'],
                             child=dict(pid=current.pid),status='issued',exit_code=None,
                             started_utc=datetime.now(timezone.utc).isoformat(),elapsed=time.monotonic()-start)
                receipts.append(receipt)
                write(label+'-issued.json',receipt)
                info=proc(current.pid);receipt['child']=info or receipt['child'];children()
                write(label+'-start.json',dict(index=index,arm=row['arm'],stage=row['stage'],argv=row['argv'],child=info,owner=owner,
                    started_utc=datetime.now(timezone.utc).isoformat(),elapsed=time.monotonic()-start))
                print(json.dumps(dict(event='stage_started',index=index,arm=row['arm'],stage=row['stage'],pid=current.pid)),flush=True)
                try:rc=current.wait(timeout=execution_remaining())
                except subprocess.TimeoutExpired:
                    why.append('execution_cutoff_2670_seconds');cleanup(hard=True)
                    current.wait(timeout=cleanup_remaining(5));rc=current.returncode
            receipt.update(status='exited',exit_code=rc,
                           seconds=time.monotonic()-before,elapsed=time.monotonic()-start,log_sha256=digest(root/(label+'.log')),stop_reasons=list(why))
            write(label+'-terminal.json',receipt)
            print(json.dumps(dict(event='stage_exited',index=index,arm=row['arm'],stage=row['stage'],exit_code=rc,seconds=receipt['seconds'])),flush=True)
            current=None
            if rc!=0 or why:raise RuntimeError('first_stage_failure:'+label)
            # Any surviving child is owned by the completed command and must not occupy the next arm.
            if any(x['state']!='Z' for x in children()):cleanup()
            if any(x['state']!='Z' for x in children()):raise RuntimeError('owned_child_survived_stage_cleanup')
            if unconfirmed_owned():raise RuntimeError('owned_child_cleanup_unconfirmed')
            guard()
            write(label+'-cost.json',costs())
        status='complete'
    except BaseException as e:
        error=repr(e);why.append(error);cleanup(hard=time.monotonic()>=deadline)
        if current is not None:
            try:rc=current.wait(timeout=cleanup_remaining(5))
            except subprocess.TimeoutExpired:rc=None
            receipt.update(status='exited' if rc is not None else 'issued',exit_code=rc,
                           seconds=time.monotonic()-before,elapsed=time.monotonic()-start,
                           log_sha256=digest(root/(label+'.log')),stop_reasons=list(why))
            if not (root/(label+'-terminal.json')).exists():write(label+'-terminal.json',receipt)
            if rc is not None:current=None
    finally:
        finished.set();watcher.join(timeout=cleanup_remaining(5))
        if current is None:
            while time.monotonic()<total_deadline:
                try:
                    pid,_=os.waitpid(-1,os.WNOHANG)
                    if pid==0:break
                except ChildProcessError:break
        alive=[x for x in children() if x['state']!='Z']
        if alive:cleanup(hard=True)
        alive=[x for x in children() if x['state']!='Z']
        unconfirmed=unconfirmed_owned()
        unresolved=[x for x in receipts if x['exit_code'] is None]
        cleanup_elapsed=time.monotonic()-start
        watcher_unfinished=watcher.is_alive()
        if cleanup_elapsed>2700 or alive or unconfirmed or unresolved or watcher_unfinished:
            status='failed';why.append('cleanup_unconfirmed_or_total_budget_overrun')
        terminal=dict(status=status,error=error,stop_reasons=why,issued_stages=receipts,completed_stages=[x for x in receipts if x['exit_code'] is not None],skipped_stages=argv[len(receipts):],
            elapsed=cleanup_elapsed,cleanup_completion_elapsed=cleanup_elapsed,owner=owner,boot_id=boot,owned_live_after_cleanup=alive,
            unconfirmed_owned=unconfirmed,unresolved_issued_stages=unresolved,watcher_unfinished=watcher_unfinished,
            gpu_hours=8*cleanup_elapsed/3600,charge_final_authority='full external owner invocation wall, including finalization; never subtract cleanup/finalization',
            peak_sampled_aggregate_rss_kib=peak_rss,per_process_hwm=[dict(pid=k[0],start_ticks=k[1],hwm_kib=v) for k,v in list(peak_pid_rss.items())],
            artifacts={str(x.relative_to(root)):dict(size_bytes=x.stat().st_size,sha256=digest(x)) for x in root.rglob('*') if x.is_file() and x.name not in ('owner.lock','terminal.json')})
        publication_elapsed=time.monotonic()-start
        terminal.update(elapsed=publication_elapsed,gpu_hours=8*publication_elapsed/3600,receipt_finalization_started_elapsed=cleanup_elapsed)
        if publication_elapsed>2700:
            status='failed';why.append('receipt_finalization_total_budget_overrun');terminal['status']=status
        write('terminal.json',terminal)
        lock.close()
        finalization_elapsed=time.monotonic()-start
        if finalization_elapsed>2700:
            status='failed';why.append('receipt_publication_total_budget_overrun')
        # The existing durable stdout event/exit receipt covers late publication; external full wall is authoritative.
        print(json.dumps(dict(event='owner_terminal',status=status,elapsed=finalization_elapsed,gpu_hours=8*finalization_elapsed/3600,
            cleanup_completion_elapsed=cleanup_elapsed,receipt_finalization_elapsed=finalization_elapsed,
            receipt_finalization_seconds=finalization_elapsed-cleanup_elapsed,error=error,stop_reasons=why,
            owned_live=len(alive),unconfirmed_owned=unconfirmed,unresolved_issued_stages=unresolved,watcher_unfinished=watcher_unfinished)),flush=True)
    return 0 if status=='complete' else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args()
    return run(args.root,args.release_sha256)


if __name__=='__main__':
    sys.exit(main())
