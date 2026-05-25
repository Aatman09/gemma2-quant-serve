import torch
import torch.nn as nn
from backend.quantize import quantize_model, QuantizedLinear  # use whatever name you saved
from backend.model import load_model

m ,_ = load_model()
number_of_linear_before = sum(1 for m in m.modules() if isinstance(m , nn.Linear))
print(f"Number of Linear before qunatization: {number_of_linear_before}")

total = sum(p.numel() for p in m.parameters())
print(total / 1e9, "B parameters")
def model_size(model):

    total_bytes = 0

    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()

    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()

    return total_bytes / 1024**3
print(model_size(m), "GB")

with torch.no_grad():
    quantize_model(m)

print(model_size(m), "GB after quntization")

number_of_linear_after = sum(1 for m in m.modules() if isinstance(m , nn.Linear))
number_of_Qunatized_linear = sum(1 for m in m.modules() if isinstance(m , QuantizedLinear))

print(f"Linears after: {number_of_linear_after}, QuantizedLinear: {number_of_Qunatized_linear}")

