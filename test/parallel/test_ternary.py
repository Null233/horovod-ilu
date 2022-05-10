import torch
import numpy as np
import horovod.torch as hvd
from horovod.torch.mpi_ops import Average, Sum

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
    a = torch.cat((encoded_data % 256, encoded_data // 256 % 256, encoded_data // 65536 %
                  256, encoded_data // 16777216 % 256), 0)
    a = torch.gather(a, 0, torch.tensor(
        np.arange(0, torch.prod(torch.tensor(shape))), device=_device))
    a = (torch.reshape(a, shape) - _num_workers).type(torch.float)
    decoded = a * scaler
    return decoded


def ternary_encoder(tensor, scaler):
    if _sto_factor:
        tensor = stochastical_binarize_tensor(tensor, scaler)
    e = torch.flatten(torch.sign(tensor).type(torch.int)) + 1
    e = torch.cat((e, torch.zeros(4 - e.size(dim=0) % 4, dtype=torch.int, device=_device)), 0)
    e_split1, e_split2, e_split3, e_split4 = torch.chunk(e, 4)
    sum_all = e_split1 + e_split2 * 256 + e_split3 * 65536 + e_split4 * 16777216
    return sum_all


for i in range(100):
    tensor = torch.randn(_shape, device=_device, requires_grad=True)
    scaler = get_scaler(tensor)
    encoded_tensor = ternary_encoder(tensor, scaler)
    sum_tensor = hvd.allreduce(encoded_tensor, name='random_tern', op=Sum)
    avg_tern_tensor = ternary_decoder(sum_tensor, _shape, scaler) / hvd.size()
    avg_ori_tensor = hvd.allreduce(tensor, name = 'random_ori', op=Average)
    loss = torch.nn.MSELoss()
    output = loss(avg_tern_tensor, avg_ori_tensor)
    output.backward()
    if hvd.rank() == 0:
        print(output)
    