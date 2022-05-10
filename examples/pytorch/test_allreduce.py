import torch
import horovod.torch as hvd


hvd.init()

val = torch.randn([1])

def printvals(broadcastfn):
    vals = broadcastfn()
    print(vals)

def broadcast_function():
    vals = hvd.allgather(val, 'scaler')
    return vals

printvals(broadcast_function)