import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from aimet_torch import model_preparer, batch_norm_fold
from aimet_torch.quantsim import QuantizationSimModel
import aimet_torch as aimet

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10("/tmp/cifar10", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("/tmp/cifar10", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
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

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

# Load the trained FP32 model from the previous step
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("/tmp/simplecnn_fp32.pth", map_location=device))
fp32_acc = evaluate(model, test_loader)
print(f"FP32 accuracy:                {fp32_acc:.2f}%")

# Session 1 lesson: convert functional F.relu into a real nn.Module
prepared = model_preparer.prepare_model(model)

# Session 3 lesson: fold BatchNorm before simulating quantization
batch_norm_fold.fold_all_batch_norms(prepared, input_shapes=(1, 3, 32, 32))
folded_acc = evaluate(prepared, test_loader)
print(f"After prepare + BN fold:      {folded_acc:.2f}%  (should match FP32 closely)")

# Session 2 lesson: 4-bit weights hurt a lot, 8-bit activations
dummy_input = torch.randn(1, 3, 32, 32).to(device)
sim = QuantizationSimModel(prepared, dummy_input=dummy_input, default_output_bw=8, default_param_bw=4)

# Calibrate with REAL training data this time
with aimet.nn.compute_encodings(sim.model):
    for idx, (x, _) in enumerate(train_loader):
        sim.model(x.to(device))
        if idx >= 10:
            break

ptq_acc = evaluate(sim.model, test_loader)
print(f"4-bit PTQ (before QAT):       {ptq_acc:.2f}%")

# QAT: train through the fake-quant nodes (STE handles the round() gradient)
optimizer = torch.optim.Adam(sim.model.parameters(), lr=1e-4)
sim.model.train()
for epoch in range(2):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(sim.model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"QAT epoch {epoch+1} done")

qat_acc = evaluate(sim.model, test_loader)
print(f"4-bit after QAT:              {qat_acc:.2f}%")
