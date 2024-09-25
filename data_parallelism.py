import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

### Step 1: setup and cleanup setups
def setup(rank, world_size):
    ...
    # initialize the process group
    dist.init_process_group("tst", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
### Step 2: define DDP modeling
def dummy_init(rank, world_size):
    setup(rank, world_size)
    model = DummyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    ...
    cleanup()
### Step 3: Spawn to run
def run_dummy(dummy_fn, world_size):
    mp.spawn(dummy_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)