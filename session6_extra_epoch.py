import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from aimet_torch import model_preparer, batch_norm_fold

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

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("/tmp/simplecnn_fp32.pth", map_location=device, weights_only=True))
print(f"FP32 accuracy (start):        {evaluate(model, test_loader):.2f}%")

# Same preparation as the QAT run, so the comparison is apples-to-apples
prepared = model_preparer.prepare_model(model)
batch_norm_fold.fold_all_batch_norms(prepared, input_shapes=(1, 3, 32, 32))

# Same optimizer, same lr, same 2 epochs as QAT — but NO quantization
optimizer = torch.optim.Adam(prepared.parameters(), lr=1e-4)
prepared.train()
for epoch in range(2):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(prepared(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"FP32 extra epoch {epoch+1} done")

print(f"FP32 after 2 extra epochs:    {evaluate(prepared, test_loader):.2f}%")
