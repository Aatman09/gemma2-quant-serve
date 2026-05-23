"""FastAPI server exposing the quantized model.

Implement:
  - load one chosen precision at startup (not per-request)
  - POST /generate  { "prompt": str, "max_new_tokens": int } -> { "text": str }
  - GET  /health    -> ok

Run: uvicorn backend.server:app --reload
"""
