import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,4,kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

net = TinyNet()
print(net)

# for name, param in net.named_parameters():
#     print(name, param.shape)

from aimet_torch.quantsim import QuantizationSimModel

dummy_input = torch.randn(1,1,8,8)

sim = QuantizationSimModel(net, dummy_input=dummy_input)
print(sim.model)

import aimet_torch.v2 as aimet

with aimet.nn.compute_encodings(sim.model):
    for _ in range(5):
        dummy_input = torch.randn(1, 1, 8, 8)
        sim.model(dummy_input)

# Now inspect the actual learned scale for the conv weight quantizer
wq = sim.model.conv.param_quantizers['weight']
print("weight quantizer min:", wq.get_min())
print("weight quantizer max:", wq.get_max())


oq = sim.model.relu.output_quantizers[0]
print("relu output quantizer min:", oq.get_min())
print("relu output quantizer max:", oq.get_max())
