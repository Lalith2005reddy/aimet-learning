import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import os

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
for epoch in range(3):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} done")

print(f"FP32 accuracy: {evaluate(model, test_loader):.2f}%")

os.makedirs(os.path.expanduser("~/aimet-learning/checkpoints"), exist_ok=True)
torch.save(model.state_dict(), os.path.expanduser("~/aimet-learning/checkpoints/simplecnn_fp32.pth"))
print("Saved to ~/aimet-learning/checkpoints/simplecnn_fp32.pth (durable, survives reboot)")
