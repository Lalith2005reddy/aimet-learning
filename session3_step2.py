import torch
from aimet_torch.cross_layer_equalization import equalize_model
import torch.nn as nn

class TwoConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,4,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

model = TwoConv()
model.eval()



