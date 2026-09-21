"""One bounded missing-readback recovery; never executes training commands."""
import json, os, pathlib, subprocess, time, traceback
from producer import ROOT, WORKTREE, PACKET, binding, publish, spawn, wait_group
RECOVERY = ROOT / 'evaluation-recovery-v1'
LOG = RECOVERY / 'handoff.log'
def signal(s):
    with LOG.open('a') as f:
        f.write(s+'\n');f.flush();os.fsync(f.fileno())
def main():
    started=time.time()
    assert binding(PACKET)['sha256']=='fd17a003423e6a914df19f8406bad19cc148e71ad88acc0d44eb58d40f1f644d'
    packet=json.loads(PACKET.read_text())
    for arm in ['A','B']:
        t=json.loads((ROOT/arm/'training/terminal.json').read_text())
        assert t['status']=='completed' and t['updates']==64
    existing=[];missing=[]
    for ep in packet['endpoints']:
        for j in ep['jobs']:
            path=pathlib.Path(j['output'])
            if path.exists():
                v=json.loads(path.read_text());assert v['status']=='completed_unscored' and v['batch_size']==4
                existing.append(binding(path))
            else:missing.append((ep['label'],j))
    assert len(existing)==61 and len(missing)==19,(len(existing),len(missing))
    publish(RECOVERY/'launch.json',{'pid':os.getpid(),'packet':binding(PACKET),'retained':existing,'missing_jobs':[{'endpoint':e,**j} for e,j in missing],'interruption':'Previous controller and unfinished workers absent; no terminal/error receipt. Cause unestablished; retained execution is not rerun.'})
    signal('RECOVERY_STARTED')
    exits=[]
    for ep in packet['endpoints']:
        for split in ['train','dev']:
            jobs=[j for e,j in missing if e==ep['label'] and j['split']==split]
            if not jobs:continue
            running=[]
            for j in jobs:
                assert not pathlib.Path(j['output']).exists()
                log=RECOVERY/'logs'/ep['label']/split/f"shard-{j['shard']:02d}.log"
                proc,handle=spawn(j['command'],devices=str(j['visible_device']),log=log)
                running.append({'endpoint':ep['label'],'split':split,'shard':j['shard'],'pid':proc.pid,'output':j['output'],'log':str(log),'process':proc,'handle':handle})
            wave=wait_group(running);exits.extend(wave)
            publish(RECOVERY/f"{ep['label']}-{split}-exits.json",wave)
            assert all(x['exit_code']==0 for x in wave),wave
    for b in existing:assert binding(b['path'])==b,'Retained output changed'
    for ep in packet['endpoints']:
        for j in ep['jobs']:
            v=json.loads(pathlib.Path(j['output']).read_text());assert v['status']=='completed_unscored' and v['batch_size']==4
    with (RECOVERY/'reducer.log').open('xb') as f:
        code=subprocess.run(packet['reducer']['command'],cwd=WORKTREE,stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,f'reducer exit {code}'
    result=pathlib.Path(packet['reducer']['output']);v=json.loads(result.read_text());assert v['status']=='completed_saved_readback_evaluation'
    publish(RECOVERY/'terminal.json',{'status':'completed','elapsed_seconds':time.time()-started,'retained_shards':61,'recovered_shards':19,'total_shards':80,'result':binding(result),'exits':exits})
    signal('RECOVERY_COMPLETE')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        publish(RECOVERY/'failure.json',{'status':'failed','error':str(e),'traceback':traceback.format_exc()})
        signal('RECOVERY_FAILED');raise
