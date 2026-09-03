#BatchNorm Folding
"""
Since conv itself is also just weight * x + bias, and BN is scale * x + shift applied right after — you can algebraically merge BN's scale/shift directly into the conv's weight and bias. Result: one conv layer that does the exact same math, and the separate BN layer disappears entirely.
"""


import torch
from aimet_torch.batch_norm_fold import fold_all_batch_norms
import torch.nn as nn

class ConvBN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x

model = ConvBN()
model.eval()

dummy_input = torch.randn(1,3,8,8)
with torch.no_grad():
    before_output = model(dummy_input)

print("Before folding:")
print(model)

fold_all_batch_norms(model, input_shapes=(1, 3, 8, 8))

print("\nAfter folding:")
print(model)

with torch.no_grad():
    after_output = model(dummy_input)

print("\nMax difference in output:", (before_output - after_output).abs().max().item())