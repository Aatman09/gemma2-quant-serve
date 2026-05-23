import torch.nn as nn


class QunatizedLinear(nn.Module):
    def __int__(self, linear :nn.Linear):
        super().__init__()
        self.linear = linear

    def forward()