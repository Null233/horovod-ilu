import torch
import time

_sto_factor = True

handles = {}
numels = {}
is_compressed = {}
acc_compensate_cache = {}
compress_tensor_threshold = 147456
compress_layer_threshold = 0.2
_sto_factor = False
compress_rate = 4
shift_factors = []
print_ = 1
index_el = 0

def compress(tensor, name):
    shape = tensor.size()
    tensor = tensor.flatten()
    abs_gradient = tensor.abs()
    scalar = abs_gradient.max()
    sign_gradient = tensor.sign() * scalar
    rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
    sign_gradient[rnd_sample >= abs_gradient] = 0
    new_sign = sign_gradient.sign()  # -1, 0, 1
    tensor_compressed = new_sign.type(torch.int8), scalar.flatten()
    return shape

def stochastical_binarize_tensor(tensor, scaler, name):
    zeros = torch.zeros_like(tensor)
    abs_tensor = torch.abs(tensor)
    sign_tensor = torch.sign(tensor)
    rnd_sample = torch.rand(tensor.shape, device='cuda') * scaler
    #where_cond = torch.less(rnd_sample, abs_tensor)
    #binarized_tensor = torch.where(where_cond, sign_tensor, zeros).
    sign_tensor[rnd_sample >= abs_tensor] = 0
    #compensate_tensor = torch.where(~where_cond, tensor, zeros)
    #acc_compensate_cache[name] = compensate_tensor
    return sign_tensor

def ternary_encoder(tensor, scaler, name):
    tensor = tensor.flatten()
    stochastical_binarize_tensor(tensor, scaler, name)
    sum_all = 0
    e = torch.sign(tensor).type(torch.int) + 1
    """redundant_size = compress_rate - e.size(dim=0) % compress_rate
    e = torch.cat((e, torch.zeros(redundant_size, dtype=torch.int, device='cuda')), 0)"""
    """for split, shift_factor in zip(torch.chunk(e, compress_rate), shift_factors):
        sum_all += split * shift_factor"""
    return sum_all

for i in range(20):
    print('------------------------------------------------------------------------------------')
    tensor = torch.randn([3,7,23,8,11,8], device='cuda')
    scaler = torch.max(tensor)
    start = time.perf_counter()
    encoded = ternary_encoder(tensor, scaler, 'tensor')
    end = time.perf_counter()
    print(f"encoding time: {(end-start)*1000}")
    start = time.perf_counter()
    encoded = compress(tensor, 'tensor')
    end = time.perf_counter()
    print(f"encoding time: {(end-start)*1000}")