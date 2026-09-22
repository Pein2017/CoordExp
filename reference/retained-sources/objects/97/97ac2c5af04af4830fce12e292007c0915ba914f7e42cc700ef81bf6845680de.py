import concurrent.futures, json, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT.parent/'second-fit-preparation-v1/manifest.json'
started=time.monotonic()
def publish(name,data):
    with (ROOT/name).open('x') as f: json.dump(data,f,indent=2);f.write('\n')
def execute(name,args,gpu,seconds):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    begin=time.monotonic()
    with (ROOT/(name+'.log')).open('x') as log:
        p=subprocess.Popen([sys.executable,'-m','probes.training_set_completion.training',*args],stdout=log,stderr=subprocess.STDOUT,env=env)
        publish(name+'-launch.json',{'pid':p.pid,'physical_gpu':gpu,'args':args,'timeout_seconds':seconds})
        try:code=p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            publish(name+'-exit.json',{'exit_code':p.returncode,'timed_out':True,'elapsed_seconds':time.monotonic()-begin})
            raise TimeoutError(name+' exceeded wall bound')
    publish(name+'-exit.json',{'exit_code':code,'elapsed_seconds':time.monotonic()-begin})
    if code:raise RuntimeError(name+' exit '+str(code))
    return name
publish('producer.json',{'pid':os.getpid(),'cwd':os.getcwd(),'manifest':str(MANIFEST),'scope':'all11 second fit from first-fit step16, fresh optimizer, targetv3, plus three cold native greedy readbacks'})
receipt={'schema':'training_set_completion.second_fit_phase.v1','phase':'training','status':'running'}
try:
    execute('training',['run','--manifest',str(MANIFEST),'--output',str(ROOT/'training'),'--device','cuda:0'],0,3660)
    terminal=json.loads((ROOT/'training/terminal.json').read_text())
    assert terminal['status']=='completed' and terminal['updates']==64 and terminal['model_forwards']==704
    receipt['phase']='cold_native_readbacks'
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(execute,'readback-step-'+str(step),['readback','--manifest',str(MANIFEST),'--adapter',str(ROOT/'training/checkpoints'/('step-'+str(step).zfill(5))/'adapter'),'--output',str(ROOT/('readback-step-'+str(step)+'.json')),'--device','cuda:0'],gpu,900) for step,gpu in [(16,1),(32,2),(64,3)]]
        for job in concurrent.futures.as_completed(jobs):job.result()
    receipt.update(status='completed_unscored',phase='terminal',readback_steps=[16,32,64])
except BaseException as exc:
    receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
finally:
    receipt['elapsed_seconds']=time.monotonic()-started
    publish('terminal.json',receipt)
    print('GPU_SECOND_FIT_SUCCESS' if receipt['status']=='completed_unscored' else 'GPU_SECOND_FIT_FAILED',flush=True)
sys.exit(0 if receipt['status']=='completed_unscored' else 1)
