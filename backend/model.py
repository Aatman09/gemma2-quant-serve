"""Loads the base Gemma 2 2B model in fp16/bfloat16.

This is the UNQUANTIZED model — the thing you quantize yourself in quantize.py.
No bitsandbytes, no quantization_config here.

Implement:
  - load_model()   -> AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16)
  - a shared tokenizer
"""
from transformers import AutoModelForCausalLM , AutoTokenizer , BitsAndBytesConfig
import torch

qunatize_int8_config = BitsAndBytesConfig(load_in_8bit=True)

def load_model(quantize_config : bool = False):
  if quantize_config:
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b-it' , quantization_config = qunatize_int8_config , device_map = 'auto')
  else:
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b-it' , torch_dtype = torch.bfloat16)
  tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')
  return model , tokenizer