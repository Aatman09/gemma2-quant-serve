"""Perplexity evaluation on wikitext-2 — the core quality metric.

Implement compute_perplexity(model, tokenizer):
  1. load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
  2. join some rows of text, tokenize
  3. forward with labels=input_ids (model shifts internally and returns loss)
  4. return torch.exp(loss)

Sanity check: a healthy fp16 Gemma-2-2B lands in the high single digits to
low teens. Hundreds+ means something is wrong.
"""
import torch 
from model import load_model
from datasets import load_dataset


@torch.no_grad()
def perplexity_calculator(model, tokenizer):
    model.to('cuda')
    model.eval()

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test"
    )
    text = '\n'.join(dataset["text"][:10])
    input_ids = tokenizer(text , return_tensors = 'pt').input_ids.to('cuda')
    out = model(input_ids , labels = input_ids)
    loss = out.loss

    ppl = torch.exp(loss)
    return ppl.item()

if __name__ == "__main__":
   
    model, tokenizer = load_model()
    ppl = perplexity_calculator(model, tokenizer)
    print(f"Perplexity: {ppl:.2f}")