"""CPU/Gloo witness of fresh versus warmed DDP bucket state; no CUDA use."""
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path('/data/CoordExp/outputs/infra_base/optimization-20260912/verification/training')


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    records = []
    for find_unused in (False, True):
        torch.manual_seed(17)
        model = torch.nn.Sequential(*[torch.nn.Linear(64,64) for _ in range(3)])
        ddp = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=0.01, find_unused_parameters=find_unused)
        steps=[]
        for step in range(1,4):
            for micro in range(6):
                x=torch.ones(2,64)*(rank+1)
                if micro<5:
                    with ddp.no_sync(): ddp(x).square().mean().backward()
                else: ddp(x).square().mean().backward()
            data=ddp._get_ddp_logging_data()
            steps.append({'step':step, 'logging':{k:v for k,v in data.items() if 'bucket' in k or k in ('iteration','prev_iteration_grad_ready_order_indices')}})
            ddp.zero_grad(set_to_none=True)
        records.append({'find_unused_parameters':find_unused,'steps':steps})
        del ddp,model
        dist.barrier()
    if rank==0:
        Path(output).write_text(json.dumps({'schema':'coordexp-task-ddp-bucket-witness-v1','backend':'gloo','world_size':2,'grad_accum_steps':6,'bucket_cap_mb':0.01,'cuda_initialized':torch.cuda.is_initialized(),'records':records},indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker,args=(str(Path(tmp)/'rdzv'),str(ROOT/'ddp-bucket-witness.json')),nprocs=2,join=True)
    print(str(ROOT/'ddp-bucket-witness.json'))
