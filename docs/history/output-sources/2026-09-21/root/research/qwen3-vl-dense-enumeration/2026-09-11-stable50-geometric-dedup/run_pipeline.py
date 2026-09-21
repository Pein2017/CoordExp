"""One fixed pilot: existing eight-rank trainer, then eight native eval shards."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

OUT=Path(__file__).parent
CWD='/data/CoordExp/.worktrees/research-probes'


def write(name,value):
    with (OUT/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')


def main():
    start=time.monotonic();status='failed';error=None;phases=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    write('pipeline-launch.json',dict(pid=os.getpid(),started_unix=time.time(),cwd=CWD,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_sha256=hashlib.sha256((OUT/'inputs.json').read_bytes()).hexdigest()))
    try:
        for command in ('run','verify'):
            cmd=['python','-m','probes.dora_owner_learning.geometric_dedup_train',command,'--output',str(OUT)]
            result=subprocess.run(cmd,cwd=CWD,env=env,check=False)
            phases.append(dict(phase='training_'+command,exit_code=result.returncode))
            if result.returncode:raise RuntimeError('training '+command+' exit '+str(result.returncode))
        evaluation=OUT/'evaluation';evaluation.mkdir(exist_ok=False)
        processes=[];logs=[]
        for i in range(8):
            log=(evaluation/f'shard-{i}.log').open('x');logs.append(log)
            cmd=['python','-m','probes.dora_owner_learning.geometric_dedup_eval','execute','--shard',str(i)]
            proc=subprocess.Popen(cmd,cwd=CWD,env=dict(env,CUDA_VISIBLE_DEVICES=str(i)),stdout=log,stderr=subprocess.STDOUT)
            processes.append((i,proc,cmd))
        with (evaluation/'process-launch.json').open('x') as stream:
            json.dump({'processes':[dict(shard=i,pid=p.pid,command=cmd) for i,p,cmd in processes]},stream,indent=2)
        results=[dict(shard=i,pid=p.pid,exit_code=p.wait()) for i,p,_ in processes]
        for log in logs:log.close()
        with (evaluation/'process-exits.json').open('x') as stream:json.dump({'results':results},stream,indent=2)
        phases.append(dict(phase='evaluation',results=results))
        if any(r['exit_code'] for r in results):raise RuntimeError('evaluation worker failure')
        for command in ('merge','verify'):
            with (evaluation/f'{command}.log').open('x') as log:
                result=subprocess.run(['python','-m','probes.dora_owner_learning.geometric_dedup_eval',command],cwd=CWD,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
            phases.append(dict(phase='evaluation_'+command,exit_code=result.returncode))
            if result.returncode:raise RuntimeError('evaluation '+command+' exit '+str(result.returncode))
        status='completed'
    except BaseException as exc:
        error=repr(exc)
        raise
    finally:
        write('pipeline-terminal.json',dict(status=status,error=error,phases=phases,wall_seconds=time.monotonic()-start,pid=os.getpid()))
        print('PIPELINE COMPLETED' if status=='completed' else 'PIPELINE FAILED',flush=True)


if __name__=='__main__':main()
