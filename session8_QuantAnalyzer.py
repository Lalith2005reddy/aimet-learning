import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from aimet_torch import model_preparer, batch_norm_fold
from aimet_torch.quant_analyzer import QuantAnalyzer
from aimet_torch.common.defs import QuantScheme, CallbackFunc

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
    return correct / total  # fraction, matches eval_callback's expected scalar

model = SimpleCNN().to(device)
model.load_state_dict(torch.load(os.path.expanduser("~/aimet-learning/checkpoints/simplecnn_fp32.pth"), map_location=device, weights_only=True))
prepared = model_preparer.prepare_model(model)
batch_norm_fold.fold_all_batch_norms(prepared, input_shapes=(1, 3, 32, 32))

dummy_input = torch.randn(1, 3, 32, 32).to(device)

def calibrate(m, _args):
    with torch.no_grad():
        for idx, (x, _) in enumerate(train_loader):
            m(x.to(device))
            if idx >= 10:
                break

def eval_fn(m):
    return evaluate(m, test_loader)

analyzer = QuantAnalyzer(
    model=prepared,
    dummy_input=dummy_input,
    forward_pass_callback=CallbackFunc(calibrate, None),
    eval_callback=eval_fn,
)

sim = analyzer._create_quantsim_and_encodings(
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    default_param_bw=4,
    default_output_bw=8,
    config_file=None,
)

fp32_score, weight_only_score, act_only_score = analyzer.check_model_sensitivity_to_quantization(sim)
print(f"FP32:                          {fp32_score*100:.2f}%")
print(f"Weight-only 4-bit (W4A32):     {weight_only_score*100:.2f}%")
print(f"Activation-only 8-bit (W32A8): {act_only_score*100:.2f}%")

per_layer = analyzer.perform_per_layer_analysis_by_enabling_quant_wrappers(sim, results_dir="/tmp/quant_analyzer_results")
print("\nPer-layer sensitivity (only this layer quantized, eval score):")
for name, score in per_layer.items():
    print(f"  {name}: {score*100:.2f}%")
