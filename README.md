# gemma2-quant

**Quantization implemented from scratch** for Gemma 2 2B — no bitsandbytes for the
core. Write the int8 quant/dequant math and the layer that uses it, then measure
what it costs in quality.

## The thesis

Quantization trades accuracy for memory. The question this project answers:
*how much accuracy does my own int8 scheme actually lose?* — measured by
**perplexity on wikitext-2**, against the fp16 baseline.

## Structure

```
backend/
  model.py        # load base Gemma 2 in fp16 (the model you quantize)
  quantize.py     # FROM SCRATCH: quant/dequant + QuantizedLinear + quantize_model
  perplexity.py   # perplexity on wikitext-2 (the quality metric)
  benchmark.py    # memory + throughput + perplexity: fp16 vs your-int8
  server.py       # FastAPI /generate endpoint
frontend/
  app.py          # Streamlit UI calling the FastAPI backend
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login          # Gemma 2 requires accepting Google's license
```

## Results

| Model              | Memory | Tokens/s | Perplexity |
|--------------------|--------|----------|------------|
| fp32 (baseline)    | TBD    | TBD      | TBD        |
| int8 (from scratch)| TBD    | TBD      | TBD        |
| int4 (from scratch)| TBD    | TBD      | TBD        |
| bitsandbytes int8  | TBD    | TBD      | TBD        | (optional reference)
