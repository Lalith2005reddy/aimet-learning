import torch
import torchvision

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
weight = model.conv1.weight.detach()  # shape (64, 3, 7, 7)

def quantize_dequantize(tensor, bitwidth, per_channel):
    qmax = 2 ** (bitwidth - 1) - 1   # e.g. 7 for 4-bit signed
    if per_channel:
        # one scale per output filter (dim 0), matching what you saw AIMET do
        scale = tensor.abs().amax(dim=[1, 2, 3], keepdim=True) / qmax
    else:
        # one scale for the whole tensor
        scale = tensor.abs().max() / qmax
    q = torch.clamp(torch.round(tensor / scale), -qmax - 1, qmax)
    return q * scale

for bw in [8, 4]:
    per_tensor_result = quantize_dequantize(weight, bw, per_channel=False)
    per_channel_result = quantize_dequantize(weight, bw, per_channel=True)

    err_tensor = (weight - per_tensor_result).pow(2).mean().item()
    err_channel = (weight - per_channel_result).pow(2).mean().item()

    print(f"{bw}-bit  |  per-tensor MSE: {err_tensor:.8f}  |  per-channel MSE: {err_channel:.8f}")