import torch
compress_rate = 8
scaler = 0.01
shift = 32 // compress_rate
mul_factor = pow(2, shift)
shift_factors = [pow(mul_factor, i) for i in range(compress_rate)]

def encode(tensor):
    shift = 32 // compress_rate
    mul_factor = pow(2, shift)
    e = torch.flatten(torch.sign(tensor).type(torch.int)) + 1
    e = torch.cat((e, torch.zeros(compress_rate - e.numel() % compress_rate, dtype=torch.int, device='cuda')), 0)
    splits = [split * shift_factor for split, shift_factor in zip(torch.chunk(e, compress_rate), shift_factors)]
    sum_all = torch.stack(splits, dim=0).sum(dim=0)
    return sum_all, tensor.shape
    
def decode(encoded, shape):
    shift = 32 // compress_rate
    mul_factor = pow(2, shift)
    splits = []
    index_original = torch.arange(0, torch.prod(torch.tensor(shape)), device='cuda')
    for i in range(compress_rate):
        div_ = torch.div(encoded, pow(mul_factor, i), rounding_mode='floor')
        split = div_ % mul_factor
        splits.append(split)
    decoded_data = torch.gather(torch.cat(splits, 0), 0, index_original)
    return decoded_data


tensor = torch.randn([2,5,7], device='cuda')
e = torch.flatten(torch.sign(tensor).type(torch.int)) + 1
encoded, shape = encode(tensor)
decoded = decode(encoded, shape)
print(e == decoded)