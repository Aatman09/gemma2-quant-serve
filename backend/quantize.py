"""From-scratch weight-only quantization. THE CORE OF THIS PROJECT.

No bitsandbytes. You implement the math.

Implement, in order:

1. quantize_tensor(W) -> (W_int8, scale)
   Per-channel symmetric int8. For a weight [out_features, in_features],
   compute one scale per output row:
       scale = W.abs().amax(dim=1) / 127
       W_int8 = round(W / scale[:, None]).clamp(-127, 127).to(int8)

2. dequantize_tensor(W_int8, scale) -> W_fp16
       W_int8.to(fp16) * scale[:, None]

3. class QuantizedLinear(nn.Module)
   Stores W_int8 (int8 buffer) + scale (fp16 buffer) + bias.
   forward(x): dequantize, then F.linear(x, W_deq, bias).
   Memory win is real ONLY if you store int8 and dequantize on the fly
   (do not keep the fp16 weight around).

4. quantize_model(model) -> model
   Recursively walk modules; replace every nn.Linear with a QuantizedLinear
   built from its weight. (Skip the lm_head if it hurts perplexity too much.)

Later (harder, stretch): int4 by packing two 4-bit values per byte.
"""
import torch 
import torch.nn as nn

def quantize_tensor(W): 
   W_fp32 = W.float()
   scale = W_fp32.abs().amax(dim=1) / 127
   scale = scale.clamp(min=1e-8)

   W_int8 = torch.clamp(torch.round(W_fp32 / scale[:,None]) , -127 , 127).to(torch.int8)
   return W_int8,scale


def dequantize_tensor(W_int8, scale):
   W_float32 = W_int8 * scale[:,None] # as scale is itself float32 the W_float32 is automatically promoted from int8 -> float32
   return W_float32


x,scale = quantize_tensor(torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]], dtype=torch.float32))
print(x)
print(dequantize_tensor(x,scale))


class QunartizedLinear(nn.Module):
   def __init__(self, Linear : nn.Linear):
      super().__init__()

      W_int8 = quantize_tensor(Linear.weight)
      
