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
model, tokenizer = load_model()
ppl_bf16 = perplexity_calculator(model, tokenizer)

del model
torch.cuda.empty_cache()

# ----- int8 quantized -----
model, _ = load_model()
quantize_model_int8(model)
ppl_int8 = perplexity_calculator(model, tokenizer)

print(f"Perplexity of bf16 is {ppl_bf16:.2f}\nPerplexity of int8 is {ppl_int8:.2f}")
