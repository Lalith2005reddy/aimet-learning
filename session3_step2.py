import torch
from aimet_torch.cross_layer_equalization import equalize_model
import torch.nn as nn

class TwoConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,4,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

model = TwoConv()
model.eval()


# Deliberately unbalance conv1's filters: make filter 0 tiny, filter 3 huge
with torch.no_grad():
    model.conv1.weight[0] *= 0.01
    model.conv1.weight[3] *= 50.0

def print_channel_ranges(model, label):
    print(f"\n{label}")
    for i in range(4):
        w = model.conv1.weight[i]
        print(f"  filter {i}: min={w.min().item():.4f}, max={w.max().item():.4f}")

print_channel_ranges(model, "Before CLE")

dummy_input = torch.randn(1, 3, 8, 8)
with torch.no_grad():
    before_output = model(dummy_input)

equalize_model(model, input_shapes=(1, 3, 8, 8))

print_channel_ranges(model, "After CLE")

with torch.no_grad():
    after_output = model(dummy_input)

print("\nMax difference in final output:", (before_output - after_output).abs().max().item())