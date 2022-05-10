import torch
import numpy as np
import horovod.torch as hvd
from horovod.torch.mpi_ops import Average, Adasum, Sum

_sto_factor = True
_shape = [50, 8, 30, 45, 7]
hvd.init()
#_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_device='cpu'
_num_workers = hvd.size()

def get_scaler(tensor):
    scaler = torch.max(tensor)
    scalers = []
    for i in range(hvd.size()):
        scaler_i = hvd.broadcast(scaler, root_rank=i, name='scaler')
        #print(scaler_i)
        scalers.append(scaler_i.item())
    return max(scalers)

def stochastical_binarize_tensor(tensor, scaler):
    shape = tensor.shape
    zeros = torch.zeros(shape, device=_device)
    abs_tensor = torch.abs(tensor)
    sign_tensor = torch.sign(tensor)
    rnd_sample = torch.rand(shape, device=_device) * scaler
    where_cond = torch.less(rnd_sample, abs_tensor)
    binarized_tensor = torch.where(where_cond, sign_tensor, zeros)
    return binarized_tensor


def ternary_decoder(encoded_data, shape, scaler):
    """Decoding the signs to float format """
    """ave_scaler = torch.gather(
        encoded_data, 0, torch.tensor(encoded_data.size(dim=0)-1, device=_device)).type(torch.float) / 10000
    encoded_data = torch.gather(encoded_data, 0, torch.tensor(
        np.arange(0, encoded_data.size(dim=0)-1), device=_device))"""
    a = torch.cat((encoded_data % 256, encoded_data // 256 % 256, encoded_data // 65536 %
                  256, encoded_data // 16777216 % 256), 0)
    a = torch.gather(a, 0, torch.tensor(
        np.arange(0, torch.prod(torch.tensor(shape))), device=_device))
    a = (torch.reshape(a, shape) - _num_workers).type(torch.float)
    #decoded = a * ave_scaler
    decoded = a
    return decoded


def ternary_encoder(tensor, scaler):
    if _sto_factor:
        tensor = stochastical_binarize_tensor(tensor, scaler)
    e = torch.flatten(torch.sign(tensor).type(torch.int)) + 1
    e = torch.cat((e, torch.zeros(4 - e.size(dim=0) % 4, dtype=torch.int, device=_device)), 0)
    e_split1, e_split2, e_split3, e_split4 = torch.chunk(e, 4)
    sum_all = e_split1 + e_split2 * 256 + e_split3 * 65536 + e_split4 * 16777216
    #sum_all = torch.cat((sum_all, scaler_pad), 0)
    return sum_all

for i in range(100):
    """tensors = []
    encoded_tensors = []
    for j in range(_num_workers):
        tensors.append(torch.randn(_shape, device=_device))
    for t in tensors:
        encoded_tensors.append(ternary_encoder(t))
    tensor_sum = torch.zeros(encoded_tensors[0].shape, device=_device).type(torch.int)
    for t in encoded_tensors:
        tensor_sum = torch.add(tensor_sum, t)
    tensor_avg = ternary_decoder(tensor_sum, _shape)

    signed_tensor = torch.zeros(_shape, device=_device)
    for t in tensors:
        signed_tensor = torch.add(signed_tensor, torch.sign(t))
    #correct_tensor = signed_tensor * avg_scaler / _num_workers
    print(torch.sum(tensor_avg == signed_tensor))
    print(torch.sum(tensor_avg == signed_tensor)
          == torch.prod(torch.tensor(_shape)))"""

    tensor = torch.randn(_shape, device=_device)
    scaler = get_scaler(tensor)
    encoded_tensor = ternary_encoder(tensor, scaler)
    sum_tensor = hvd.allreduce(encoded_tensor, name='random', op=Sum)
    decoded_tensor = ternary_decoder(sum_tensor, _shape, scaler)
    signed_tensor = torch.sign(tensor)
    print(torch.sum(decoded_tensor == signed_tensor)
          == torch.prod(torch.tensor(_shape)))