import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.arch_checker.arch_checker import ArchChecker
from aimet_torch import model_preparer, batch_norm_fold

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # reuses self.pool below too
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.shape[0], -1)
        return self.fc(x)

model = SimpleCNN().to(device)
model.load_state_dict(torch.load(os.path.expanduser("~/aimet-learning/checkpoints/simplecnn_fp32.pth"), map_location=device, weights_only=True))
model.eval()

dummy_input = torch.randn(1, 3, 32, 32).to(device)

print("=== ModelValidator on the RAW model (before prepare/fold) ===")
is_valid = ModelValidator.validate_model(model, dummy_input)
print("Model valid:", is_valid)

print("\n=== ModelValidator on the PREPARED + BN-folded model ===")
prepared = model_preparer.prepare_model(model)
batch_norm_fold.fold_all_batch_norms(prepared, input_shapes=(1, 3, 32, 32))
is_valid_prepared = ModelValidator.validate_model(prepared, dummy_input)
print("Model valid:", is_valid_prepared)

print("\n=== ArchChecker on the prepared model ===")
report = ArchChecker.check_model_arch(prepared, dummy_input, result_dir="/tmp/arch_checker_results")
print(report)
