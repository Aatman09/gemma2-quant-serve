"""Compare the base fp16 model vs YOUR quantized model.

For each of {fp16 base, your-int8 (, your-int4)}:
  - memory:     sum of parameter bytes (int8 weight = 1 byte, fp16 = 2)
  - throughput: warmup, then torch.cuda.synchronize() around timed generate()
  - quality:    compute_perplexity() from perplexity.py

Optional reference row: bitsandbytes load_in_8bit — only to validate that
your from-scratch perplexity is in the same ballpark. Not the point.

Output: the Memory / Tokens-s / Perplexity table from the README.
"""
