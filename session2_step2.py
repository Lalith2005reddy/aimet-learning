import torch
import torchvision
from aimet_torch.quantsim import QuantizationSimModel
import aimet_torch as aimet

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.eval()

torch.manual_seed(0)
fixed_image = torch.randn(1,3,224,224)  # stand-in for a real photo, same shape ResNet18 expects

with torch.no_grad():
    fp32_output = model(fixed_image)
fp32_class = fp32_output.argmax(dim=1).item()
print("FP32 predicted class:", fp32_class)


sim = QuantizationSimModel(model,dummy_input=fixed_image,default_output_bw=8,default_param_bw=4)

with aimet.nn.compute_encodings(sim.model):
    for _ in range(5):
        sim.model(torch.randn(1, 3, 224, 224))  # calibration data (also random, just for demo)


with torch.no_grad():
    quant_output = sim.model(fixed_image)
quant_class = quant_output.argmax(dim=1).item()
print("8-bit quantized predicted class:", quant_class)