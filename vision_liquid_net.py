import torch
from liquidnet.vision_liquidnet import VisionLiquidNet

# Random Input Image
x = torch.randn(4, 3, 32, 32)

# Create a VisionLiquidNet with a specified number of units
model = VisionLiquidNet(64, 10)

# Forward pass through the VisionLiquidNet
print(model(x).shape)
