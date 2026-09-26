import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from aimet_torch.compress import ModelCompressor
from aimet_torch.common.defs import CostMetric, CompressionScheme
from aimet_torch.defs import SpatialSvdParameters
from aimet_torch.defs import GreedySelectionParameters  # fallback import path if aimet_torch.common lacks it

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10("/tmp/cifar10", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("/tmp/cifar10", train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.shape[0], -1)
        return self.fc(x)

def evaluate_full(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total

# Compression's expected callback signature: evaluate(model, iterations, use_cuda)
def eval_for_compression(model, iterations, use_cuda):
    d = "cuda" if use_cuda else "cpu"
    model = model.to(d)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            if iterations is not None and idx >= iterations:
                break
            x, y = x.to(d), y.to(d)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total

model = SimpleCNN().to(device)
model.load_state_dict(torch.load(os.path.expanduser("~/aimet-learning/checkpoints/simplecnn_fp32.pth"), map_location=device, weights_only=True))
print(f"FP32 accuracy (full test set): {evaluate_full(model, test_loader)*100:.2f}%")

greedy_params = GreedySelectionParameters(
    target_comp_ratio=0.5,       # aim to keep ~50% of original MACs
    num_comp_ratio_candidates=4, # try 4 ratios per layer (kept small — this model is tiny, and each candidate re-evaluates)
)
auto_params = SpatialSvdParameters.AutoModeParams(greedy_select_params=greedy_params)
params = SpatialSvdParameters(mode=SpatialSvdParameters.Mode.auto, params=auto_params)

compressed_model, stats = ModelCompressor.compress_model(
    model=model,
    eval_callback=eval_for_compression,
    eval_iterations=5,  # batches per candidate evaluation, kept small for speed
    input_shape=(1, 3, 32, 32),
    compress_scheme=CompressionScheme.spatial_svd,
    cost_metric=CostMetric.mac,
    parameters=params,
)

print("\n", stats)
print(f"\nCompressed model accuracy (full test set): {evaluate_full(compressed_model, test_loader)*100:.2f}%")
print("\nCompressed model structure:")
print(compressed_model)

optimizer = torch.optim.Adam(compressed_model.parameters(), lr=1e-3)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

compressed_model.train()
for epoch in range(3):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(compressed_model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Fine-tune epoch {epoch+1} done")

print(f"Compressed + fine-tuned accuracy: {evaluate_full(compressed_model, test_loader)*100:.2f}%")

model_control = SimpleCNN().to(device)
model_control.load_state_dict(torch.load(os.path.expanduser("~/aimet-learning/checkpoints/simplecnn_fp32.pth"), map_location=device, weights_only=True))

optimizer = torch.optim.Adam(model_control.parameters(), lr=1e-3)
model_control.train()
for epoch in range(3):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model_control(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"FP32 control epoch {epoch+1} done")

print(f"FP32 with 3 extra epochs (control): {evaluate_full(model_control, test_loader)*100:.2f}%")
