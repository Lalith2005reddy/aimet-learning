import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from aimet_torch import model_preparer, batch_norm_fold
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.mixed_precision import choose_mixed_precision
from aimet_torch.common.defs import QuantizationDataType, CallbackFunc

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10("/tmp/cifar10", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("/tmp/cifar10", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
amp_loader = DataLoader(Subset(test_set, range(0, 2000)), batch_size=256)       # AMP chooses with this
final_loader = DataLoader(Subset(test_set, range(2000, 10000)), batch_size=256)  # we report on this

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

# Callbacks must have signature func(model, args)
def eval_fn(model, loader):
    return evaluate(model, loader) / 100.0   # fraction, so allowed_accuracy_drop=0.02 means 2 points

def calibrate(model, _args):
    with torch.no_grad():
        for idx, (x, _) in enumerate(train_loader):
            model(x.to(device))
            if idx >= 10:
                break

# Same preparation as Session 6
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("/tmp/simplecnn_fp32.pth", map_location=device, weights_only=True))
prepared = model_preparer.prepare_model(model)
batch_norm_fold.fold_all_batch_norms(prepared, input_shapes=(1, 3, 32, 32))
print(f"FP32 (final split):            {evaluate(prepared, final_loader):.2f}%")

dummy_input = torch.randn(1, 3, 32, 32).to(device)

def make_sim(param_bw):
    sim = QuantizationSimModel(copy.deepcopy(prepared), dummy_input=dummy_input,
                               default_output_bw=8, default_param_bw=param_bw)
    sim.compute_encodings(calibrate, None)
    return sim

sim8 = make_sim(8)
print(f"Uniform W8/A8:                 {evaluate(sim8.model, final_loader):.2f}%")
sim4 = make_sim(4)
print(f"Uniform W4/A8:                 {evaluate(sim4.model, final_loader):.2f}%")

# AMP: starts at the highest candidate, tries lowering weight bits per layer
sim = make_sim(8)
candidates = [
    ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),  # highest
    ((8, QuantizationDataType.int), (4, QuantizationDataType.int)),  # lowest
]
pareto = choose_mixed_precision(
    sim, dummy_input, candidates,
    eval_callback_for_phase1=CallbackFunc(eval_fn, amp_loader),
    eval_callback_for_phase2=CallbackFunc(eval_fn, amp_loader),
    allowed_accuracy_drop=0.02,
    results_dir="/tmp/amp_results",
    clean_start=True,
    forward_pass_callback=CallbackFunc(calibrate, None),
)

print(f"AMP mixed precision:           {evaluate(sim.model, final_loader):.2f}%")
print("\nWeight bitwidth chosen per layer:")
for name, m in sim.model.named_modules():
    pq = getattr(m, "param_quantizers", None)
    if pq is not None and "weight" in pq and pq["weight"] is not None:
        print(f"  {name}: {getattr(pq['weight'], 'bitwidth', '?')}-bit")
