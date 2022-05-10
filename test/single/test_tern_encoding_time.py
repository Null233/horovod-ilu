import torch
import time
import numpy as np

compress_rate = 8
scaler = 0.01
shift = 32 // compress_rate
mul_factor = pow(2, shift)
shift_factors = [pow(mul_factor, i) for i in range(compress_rate)]

def stochastical_binarize_tensor(tensor, scaler):
    zeros = torch.zeros(tensor.shape, device='cuda')
    abs_tensor = torch.abs(tensor)
    sign_tensor = torch.sign(tensor)
    rnd_sample = torch.rand(tensor.shape, device='cuda') * scaler
    where_cond = torch.less(rnd_sample, abs_tensor)
    binarized_tensor = torch.where(where_cond, sign_tensor, zeros)
    return binarized_tensor

def ternary_encoder(tensor, scaler):
    print('')
    print('-------------------------NEW-------------------------')
    tensor = stochastical_binarize_tensor(tensor, scaler)
    e = torch.flatten(torch.sign(tensor).type(torch.int)) + 1
    redundant_size = compress_rate - e.size(dim=0) % compress_rate
    e = torch.cat((e, torch.zeros(redundant_size, dtype=torch.int, device='cuda')), 0)
    end_1 = time.perf_counter()

    sum_all = 0
    splits = [split * shift_factor for split, shift_factor in zip(torch.chunk(e, compress_rate), shift_factors)]
    end_2 = time.perf_counter()
    print(f'Splition  Time: {end_2-end_1}')
    sum_all = torch.stack(splits, dim=0).sum(dim=0).sum(dim=0)
    end_3 = time.perf_counter()
    print(f'Summation Time: {end_3-end_2}')
    print('-------------------------NEW-------------------------')
    print('')
    return sum_all

def ternary_encoder_ori(tensor, scaler):
    print('')
    print('-------------------------ORI-------------------------')
    tensor = stochastical_binarize_tensor(tensor, scaler)
    e = torch.flatten(torch.sign(tensor).type(torch.int)) + 1
    e = torch.cat((e, torch.zeros(4 - e.size(dim=0) %
                    4, dtype=torch.int, device='cuda')), 0)
    end_1 = time.perf_counter()

    e_split1, e_split2, e_split3, e_split4 = torch.chunk(e, 4)
    end_1 = time.perf_counter()
    sum_all = e_split1 + e_split2 * 256 + e_split3 * 65536 + e_split4 * 16777216
    end_2 = time.perf_counter()
    print(f'Summation Time: {end_2-end_1}')
    print('-------------------------ORI-------------------------')
    print('')
    return sum_all

def ternary_decoder(encoded_data, shape):
    """Decoding the signs to float format """
    index_original = torch.arange(0, torch.prod(torch.tensor(shape)), device='cuda')
    splits = [((encoded_data >> (compress_rate * i)) % mul_factor) \
        for i in range(compress_rate)]
    decoded_summed_data = torch.gather(torch.cat(splits, 0), 0, index_original)
    decoded_summed_data = torch.reshape(decoded_summed_data, shape).type(torch.float)
    return decoded_summed_data * scaler

tensors = [torch.randn([22,11,8,13,15], device='cuda') for i in range(10)]

for i in range(10):
    start = time.perf_counter()
    encoded_data = ternary_encoder(tensors[i], torch.max(tensors[i]))
    end_encode = time.perf_counter()
    encoded_data = ternary_encoder_ori(tensors[i], torch.max(tensors[i]))
    end_encode_ori = time.perf_counter()
    #ternary_decoder(encoded_data, tensors[i].shape)
    #end_decode = time.perf_counter()
    #print(f"Encoding time_new: {end_encode-start}")
    #print(f"Encoding time_ori: {end_encode_ori-end_encode}")