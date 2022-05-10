import torch
import numpy as np

shape = [50, 8, 30, 45, 7]
for i in range(100):
    tensor = torch.randn(shape)

    e = (torch.reshape(torch.sign(tensor), (-1,)) + 1).type(torch.int)
    e = torch.cat((e, torch.zeros(4 - e.size(dim=0) % 4).type(torch.int)), 0)
    e_split1, e_split2, e_split3, e_split4 = torch.chunk(e, 4)
    encoded_data = e_split1 + e_split2 * 256 + e_split3 * 65536 + e_split4 * 16777216

    a = torch.cat((a % 256, a // 256 % 256, a // 65536 % 256, a // 16777216 % 256), 0)
    a = torch.gather(a, 0, torch.tensor(np.arange(0, torch.prod(torch.tensor(shape)))))
    a = (torch.reshape(a, shape) - 1).type(torch.float)

    #print(a==torch.sign(tensor))
    print(torch.sum(a==torch.sign(tensor))
            == torch.prod(torch.tensor(shape)))