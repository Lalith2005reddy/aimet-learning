import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.seq_mse import apply_seq_mse, SeqMseParams
from aimet_torch.bn_reestimation import reestimate_bn_stats
import aimet_torch as aimet

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.eval()

torch.manual_seed(0)
fixed_image = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    fp32_class = model(fixed_image).argmax(dim=1).item()
print("FP32 predicted class:", fp32_class)

# Labeled loader — used later for BN Reestimation, which has its own explicit forward_fn
calibration_data = torch.randn(16, 3, 224, 224)
dummy_labels = torch.zeros(16)
data_loader = DataLoader(TensorDataset(calibration_data, dummy_labels), batch_size=4)

# Unlabeled loader — SeqMSE's internal default_forward_fn expects bare tensors, not (input, label) pairs
seq_mse_calibration_data = torch.randn(16, 3, 224, 224)
seq_mse_loader = DataLoader(TensorDataset(seq_mse_calibration_data), batch_size=4)

sim = QuantizationSimModel(model, dummy_input=fixed_image, default_output_bw=8, default_param_bw=4)

# Step 1: SeqMSE picks better weight ranges before we calibrate activations
seq_mse_params = SeqMseParams(num_batches=4)
apply_seq_mse(model, sim, seq_mse_loader, seq_mse_params)

# Step 2: Calibrate activation quantizers as usual
with aimet.nn.compute_encodings(sim.model):
    for _ in range(5):
        sim.model(torch.randn(1, 3, 224, 224))

with torch.no_grad():
    seqmse_class = sim.model(fixed_image).argmax(dim=1).item()
print("4-bit SeqMSE predicted class:", seqmse_class)

# Step 3: BN Reestimation on the original FP32 model, before it's folded elsewhere
def forward_fn(m, batch):
    inputs, _ = batch
    return m(inputs)

handle = reestimate_bn_stats(model, data_loader, num_batches=4, forward_fn=forward_fn)
print("BN reestimation done — running_mean/var updated in place")
handle.remove()  # commits the reestimated stats permanently
