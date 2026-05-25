"""Compare the base bf16 model vs YOUR quantized model.

For each of {bf16 base, your-int8}:
  - memory:     sum of parameter bytes (int8 weight = 1 byte, bf16 = 2)
  - throughput: warmup, then torch.cuda.synchronize() around timed generate()
  - quality:    compute_perplexity() from perplexity.py

Output: the Memory / Tokens-s / Perplexity table from the README.
"""
import torch
from model import load_model
from quantize import quantize_model_int8
from perplexity import perplexity_calculator

# ----- bf16 baseline -----

def model_size_gb(model):
    total_bytes = 0 
    for p in model.parameters():
        total_bytes+= p.numel() * p.element_size()
    for p in model.buffers():
        total_bytes+= p.numel() *p.element_size()

    return total_bytes /1024**3


model, tokenizer = load_model()
model_size_normal = model_size_gb(model)
ppl_bf16 = perplexity_calculator(model, tokenizer)

del model
torch.cuda.empty_cache()

# ----- int8 quantized -----
model, _ = load_model()
quantize_model_int8(model)
model_size_int8 = model_size_gb(model)
ppl_int8 = perplexity_calculator(model, tokenizer)

del model
torch.cuda.empty_cache()
#------- int8 Bitsandbyte config -----
model , _ = load_model(quantize_config=True)
model_size_int8_bits_and_bytes = model_size_gb(model)
ppl_int8_bits_and_bytes = perplexity_calculator(model , tokenizer)
print(f"Perplexity of bf16 is {ppl_bf16:.2f}\nPerplexity of int8 is {ppl_int8:.2f}\nPerplexity of int8 Bits and bytes {ppl_int8_bits_and_bytes:.2f}")
print(f"Model size of bf16 is {model_size_normal:.2f}\nModel Size of int8 is {model_size_int8:.2f}\nModel _size of int8 Bits and bytes {model_size_int8_bits_and_bytes:.2f}")


