"""Perplexity evaluation on wikitext-2 — the core quality metric.

Implement compute_perplexity(model, tokenizer):
  1. load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
  2. join some rows of text, tokenize
  3. forward with labels=input_ids (model shifts internally and returns loss)
  4. return torch.exp(loss)

Sanity check: a healthy fp16 Gemma-2-2B lands in the high single digits to
low teens. Hundreds+ means something is wrong.
"""
