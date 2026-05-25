import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_tensor_int8(W):
    scale = W.abs().amax(dim=1) / 127
    scale = scale.clamp(min=1e-8)
    W_int8 = torch.clamp(torch.round(W / scale[:, None]), -127, 127).to(torch.int8)
    return W_int8, scale


def dequantize_tensor_int8(W_int8, scale):
    return W_int8 * scale[:, None]


class QuantizedLinearInt8(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        W_int8, scale = quantize_tensor_int8(linear.weight)
        self.register_buffer('W_int8', W_int8)
        self.register_buffer('scale', scale.to(torch.bfloat16))
        self.bias = linear.bias

    def forward(self, x):
        W_deq = dequantize_tensor_int8(self.W_int8, self.scale)
        return F.linear(x, W_deq, self.bias)


def quantize_model_int8(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantizedLinearInt8(child))
        else:
            quantize_model_int8(child)
    return module
