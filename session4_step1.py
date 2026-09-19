#Adaround

import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.quantsim import QuantizationSimModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.eval()
model = model.to(device)

torch.manual_seed(0)
fixed_image = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    fp32_class = model(fixed_image).argmax(dim=1).item()
print("FP32 predicted class:", fp32_class)

# Small unlabeled calibration set for AdaRound to optimize rounding against
calibration_data = torch.randn(16, 3, 224, 224)
dummy_labels = torch.zeros(16)  # never actually used, just satisfies AdaRound's expected (input, label) format
data_loader = DataLoader(TensorDataset(calibration_data, dummy_labels), batch_size=4)

params = AdaroundParameters(data_loader=data_loader, num_batches=4)

ada_model = Adaround.apply_adaround(
    model, fixed_image, params,
    path="/tmp", filename_prefix="resnet18_adaround",
    default_param_bw=4,
)

sim = QuantizationSimModel(ada_model, dummy_input=fixed_image, default_output_bw=8, default_param_bw=4)

import aimet_torch.v2 as aimet
with aimet.nn.compute_encodings(sim.model):
    for _ in range(5):
        sim.model(torch.randn(1, 3, 224, 224).to(device))

with torch.no_grad():
    adaround_class = sim.model(fixed_image).argmax(dim=1).item()
print("4-bit AdaRound predicted class:", adaround_class)
