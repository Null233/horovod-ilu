import torch

import horovod.torch as hvd
from horovod.torch import allreduce_, allgather
from horovod.torch import size

hvd.init()

for i in range(10):
    tensor = torch.randn([3,4,5,6])
    scaler = tensor.max().abs().view(1)
    scalers = allgather(scaler, f'scaler_{i}')
    print(scalers)