# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""

import torch
import numpy as np
import time

class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""
    @staticmethod
    def compress(tensor, size, broadcastfn, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx, size, shape, scaler,name,sync_func):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""
    @staticmethod
    def compress(tensor, size, broadcastfn, name):
        """Returns the tensor unmodified."""
        return tensor, None, None

    @staticmethod
    def decompress(tensor, ctx, size, shape, scaler,name,sync_func):
        """Returns the tensor unmodified."""
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    @staticmethod
    def compress(tensor, size, broadcastfn, name):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype, tensor.shape

    @staticmethod
    def decompress(tensor, ctx, size, shape, scaler,name,sync_func):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed


class TernCompressor(Compressor):
    """Quantize all floating point to ternary from."""
    
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
    compensate_factor = 0

    @staticmethod
    def get_attributes():
        compress_rate = TernCompressor.compress_rate
        shift = 32 // compress_rate
        mul_factor = pow(2, shift)
        shift_factors = TernCompressor.shift_factors
        return compress_rate, shift, mul_factor, shift_factors

    @staticmethod
    def get_max_scaler(tensor, size, broadcastfn, name):
        scaler = torch.abs(torch.max(tensor)).reshape([1])
        scaler_name = f'{name}.scaler'
        scalers = broadcastfn(scaler, scaler_name)
        scaler = torch.max(scalers)
        TernCompressor.handles[scaler_name] = scaler
        return scaler.item()

    @staticmethod
    def stochastical_binarize_tensor(tensor, scaler, name):
        #zeros = torch.zeros_like(tensor)
        abs_tensor = torch.abs(tensor)
        sign_tensor = torch.sign(tensor)
        rnd_sample = torch.rand(tensor.shape, device='cuda') * scaler
        #where_cond = torch.less(rnd_sample, abs_tensor)
        #binarized_tensor = torch.where(where_cond, sign_tensor, zeros)
        if TernCompressor.compensate_factor:
            compensate_tensor = sign_tensor
            compensate_tensor[rnd_sample < abs_tensor] = 0
            TernCompressor.acc_compensate_cache[name] = compensate_tensor
        sign_tensor[rnd_sample >= abs_tensor] = 0
        return sign_tensor

    @staticmethod
    def tensor_clamp(tensor):
        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        return gradient

    @staticmethod
    def ternary_decoder(encoded_data, shape, size, name, sync_func):
        """Decoding the signs to float format """
        scaler_name = f'{name}.scaler'
        #output = sync_func(TernCompressor.handles.pop(scaler_name))
        scaler = TernCompressor.handles[scaler_name]
        index_original = torch.arange(0, torch.prod(torch.tensor(shape)), device='cuda')
        _, __, mul_factor, shift_factors = TernCompressor.get_attributes()
        splits = [torch.div(encoded_data, shift_factor, rounding_mode='floor') % mul_factor \
                    for shift_factor in TernCompressor.shift_factors]
        decoded_summed_data = torch.gather(torch.cat(splits, 0), 0, index_original)
        decoded_summed_data = (torch.reshape(decoded_summed_data, shape) - size).type(torch.float)
        return decoded_summed_data * scaler

    @staticmethod
    def ternary_encoder(tensor, scaler, name):
        if TernCompressor._sto_factor:
            tensor = TernCompressor.stochastical_binarize_tensor(tensor, scaler, name)
        compress_rate, _, __, shift_factors = TernCompressor.get_attributes()
        sum_all = 0
        e = torch.sign(tensor).type(torch.int) + 1
        redundant_size = compress_rate - e.size(dim=0) % compress_rate
        e = torch.cat((e, torch.zeros(redundant_size, dtype=torch.int, device='cuda')), 0)
        for split, shift_factor in zip(torch.chunk(e, compress_rate), shift_factors):
            sum_all += split * shift_factor
        return sum_all

    @staticmethod
    def compress(tensor, size, broadcastfn, name):
        shape = tensor.shape
        ctx = tensor.dtype
        if not TernCompressor.numels.get(name):
            TernCompressor.numels[name] = (tensor.numel(), TernCompressor.index_el)
            TernCompressor.index_el += 1
            TernCompressor.is_compressed[name] = 0
        else:
            TernCompressor.is_compressed[name] = 1 if \
                TernCompressor.numels[name][0] > TernCompressor.compress_tensor_threshold and \
                TernCompressor.numels[name][1] > len(TernCompressor.numels)*(1-TernCompressor.compress_layer_threshold) \
                else 0
        if TernCompressor.is_compressed[name]:
            if not TernCompressor.shift_factors:
                compress_rate, _, mul_factor, shift_factors = TernCompressor.get_attributes()
                TernCompressor.shift_factors = [pow(mul_factor, i) for i in range(TernCompressor.compress_rate)]
            tensor_compressed = tensor.flatten()
            tensor_compressed.requires_grad = False
            if TernCompressor.compensate_factor:
                if name in TernCompressor.acc_compensate_cache:
                    tensor_compressed.add_(TernCompressor.acc_compensate_cache[name])
                    #tensor_compressed = tensor_compressed + TernCompressor.acc_compensate_cache[name]
            tensor_compressed = TernCompressor.tensor_clamp(tensor_compressed)
            unified_scaler = TernCompressor.get_max_scaler(
                tensor_compressed, size, broadcastfn, name)
            if tensor.dtype.is_floating_point:
                # Only allow compression from other floating point types
                tensor_compressed = TernCompressor.ternary_encoder(
                    tensor_compressed, unified_scaler, name)
            return tensor_compressed, ctx, shape
        else:
            return tensor, ctx, shape

    @staticmethod
    def decompress(tensor, ctx, size, shape, scaler, name, sync_func):
        if TernCompressor.is_compressed[name]:
            if not TernCompressor.shift_factors:
                compress_rate, _, mul_factor, shift_factors = TernCompressor.get_attributes()
                TernCompressor.shift_factors = [pow(mul_factor, i) for i in range(TernCompressor.compress_rate)]
            tensor_decompressed = tensor
            dtype = ctx
            if dtype.is_floating_point:
                tensor_decompressed = TernCompressor.ternary_decoder(
                    tensor, shape, size, name, sync_func) / size
            return tensor_decompressed
        else:
            return tensor


class TegrCompressor(Compressor):
    @staticmethod
    def testde(tensor_compressed, shape):
        #print(f"3. received data:{tensor_compressed}")
        ave_scaler = torch.gather(tensor_compressed, 0, torch.tensor(
            tensor_compressed.size(dim=0)-1, device='cuda'))
        ave_scaler = ave_scaler.type(torch.float)
        ave_scaler /= 20000
        tensor_compressed = torch.gather(tensor_compressed, 0, torch.tensor(
            np.arange(0, tensor_compressed.size(dim=0)-1), device='cuda'))

        # next, decompress tensor from 1/4
        #tensor_compressed = TegrCompressor.decompress_from_025(tensor_compressed,shape)

        sign = tensor_compressed.type(torch.float32)
        tensor_compressed = tensor_compressed*0.5
        tensor_decompressed = sign * ave_scaler
        return tensor_decompressed.view(shape)

    @staticmethod
    def testen(tensor):
        #print(f"1. init data:{tensor}")
        shape = tensor.size()
        tensor = torch.reshape(tensor, (-1,))

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()

        sign_gradient = gradient.sign() * scalar
        rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
        sign_gradient[rnd_sample >= abs_gradient] = 0
        new_sign = sign_gradient.sign()  # -1, 0, 1

        # next, compress tensor to 1/4
        #new_sign = TegrCompressor.compress_to_025(new_sign)

        # add scalar to the tensor end
        scaler_cat = torch.tensor(
            [scalar.flatten().item()*10000], device='cuda')
        tensor_compressed = torch.cat((new_sign, scaler_cat), 0)
        tensor_compressed = tensor_compressed.type(torch.int32)

        tensor_compressed *= 2
        #print(f"2. encode data:{tensor_compressed}")
        return tensor_compressed

    @staticmethod
    def compress_to_025(a):  # a is the tensor need to be compressed
        pad_size = 4 - a.size(dim=0) % 4
        if pad_size > 0:
            pad = torch.zeros(pad_size, device='cuda')
            a = torch.cat((a, pad), 0)
        a_split1, a_split2, a_split3, a_split4 = torch.chunk(a, 4)
        sum_1 = torch.add(a_split1, a_split2 * 256)
        sum_2 = torch.add(a_split3 * 65536, a_split4 * 16777216)
        sum_all = torch.add(sum_1, sum_2)
        return sum_all

    @staticmethod
    def decompress_from_025(encoded_data, shape):
        a = encoded_data.type(torch.int32)
        a_split1 = (a % 5)
        a_split2 = (a // 256 % 5)
        a_split3 = a // 65536 % 5
        a_split4 = a // 16777216 % 5
        a = torch.cat([a_split1, a_split2, a_split3, a_split4], 0)
        real_size = torch.prod(torch.tensor(shape))
        #a = a.type(torch.float)
        a = torch.gather(a, 0, torch.tensor(
            np.arange(0, real_size), device='cuda'))
        a = torch.reshape(a, shape)
        return a

    @staticmethod
    def compress(tensor, size, broadcastfn, name):
        tensor_compressed = tensor
        tensor_compressed = TegrCompressor.testen(tensor_compressed)
        return tensor_compressed, tensor.dtype, tensor.shape

    @staticmethod
    def decompress(tensor, ctx, size, shape, scaler, name,sync_func):
        tensor_decompressed = tensor
        dtype = ctx
        tensor_decompressed = TegrCompressor.testde(tensor_decompressed, shape)
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor

    """Quantize all floating point to ternary from."""
    tern = TernCompressor

    tegr = TegrCompressor
