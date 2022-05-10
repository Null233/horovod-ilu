import torch
import numpy as np
import horovod.torch as hvd
from horovod.torch.mpi_ops import Average, Sum, synchronize
import time

hvd.init()
handles = {}

def start_allgather_scaler(tensor, name):
    handles[name] = hvd.allgather_async(tensor, name)

def get_allgather_scaler(name):
    output = synchronize(handles[name])
    return output

for i in range(10):
    tensor = torch.randn([3,7,5,6])
    name = f'scaler_{i%3}'
    start_allgather_scaler(torch.tensor([torch.max(tensor)]), name)
    time.sleep(0.1)
    if hvd.rank() == 0:
        print(get_allgather_scaler(name))
    