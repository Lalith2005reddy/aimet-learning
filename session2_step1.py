import torch
import torchvision

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Look at just the first conv layer's weight shape
print("conv1 weight shape:", model.conv1.weight.shape)