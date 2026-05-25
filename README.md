# gemma2-quant

**Quantization implemented from scratch** for Gemma 2 2B — no bitsandbytes for the
core. Write the int8 quant/dequant math and the layer that uses it, then measure
what it costs in quality.

## The thesis

Quantization trades accuracy for memory. The question this project answers:
*how much accuracy does my own int8 scheme actually lose?* — measured by
**perplexity on wikitext-2**, against the bf16 baseline.

## Structure

```
backend/
  model.py        # load base Gemma 2 in bf16 (or bnb int8 reference)
  quantize.py     # FROM SCRATCH: quant/dequant + QuantizedLinear + quantize_model
  perplexity.py   # perplexity on wikitext-2 (the quality metric)
  benchmark.py    # memory + perplexity: bf16 vs my-int8 vs bitsandbytes int8
  server.py       # FastAPI /generate endpoint
frontend/
  app.py          # Streamlit UI calling the FastAPI backend
```

## What "from scratch" means here

- **Per-channel symmetric int8**, one scale per output row:
  `scale = W.abs().amax(dim=1) / 127`
- Zero-row guard via `scale.clamp(min=1e-8)` to avoid div-by-zero.
- `QuantizedLinear` stores `W_int8` and `scale` as buffers; `forward` dequantizes
  on the fly and runs `F.linear` — no fp weight kept in memory.
- `quantize_model` is a recursive walk that replaces every `nn.Linear` with a
  `QuantizedLinear` via `setattr`.
- All math runs in bf16; scale is stored in bf16 so the dequantized weight matches
  the model's activation dtype with no implicit casts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login          # Gemma 2 requires accepting Google's license
```

## Reproduce

```bash
python backend/benchmark.py
```

## Results — Gemma 2 2B, wikitext-2 test

| Model                  | Memory   | Δ Memory | Perplexity | Δ Quality |
|------------------------|----------|----------|------------|-----------|
| bf16 (baseline)        | 4.87 GB  | —        | 12.46      | —         |
| **int8 (from scratch)**| **3.54 GB** | **-27%** | **12.53** | **+0.56%** |
| bitsandbytes int8      | 2.98 GB  | -39%     | 12.70      | +1.93%    |

### Reading the table

- **My int8 beats bitsandbytes on quality** (12.53 vs 12.70). bitsandbytes casts
  bf16 activations to fp16 internally before its int8 matmul; the cast costs a
  little accuracy. The from-scratch path stays in bf16 end-to-end.
- **bitsandbytes saves more memory** (-39% vs -27%). The gap is because my
  current `quantize_model` only replaces `nn.Linear` modules. Gemma's
  `embed_tokens` is `nn.Embedding` (not Linear) and stays in bf16 — ~1.18 GB
  untouched. Quantizing embeddings is a separate piece of work (different
  forward: a lookup, not a matmul).
- The trade-off: my pipeline is on a more accuracy-favoring point of the
  memory/quality Pareto curve. Same ballpark on size, ~3× less perplexity damage.

## Limitations / Next steps

- `nn.Embedding` is not quantized — biggest remaining memory chunk.
- Weight tying between `embed_tokens` and `lm_head` is broken by the current
  Linear-replacement walk; skipping `lm_head` recovers the tie and drops memory
  further.
- Throughput (tokens/sec with `model.generate`) not yet measured.
- int4 packed (two 4-bit values per byte) is a stretch goal.
