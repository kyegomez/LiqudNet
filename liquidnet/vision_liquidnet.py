import torch
from torch import nn
from liquidnet.main import LiquidNet


class VisionLiquidNet(nn.Module):
    def __init__(self, num_units, num_classes):
        super(VisionLiquidNet, self).__init__()
        # Convolutional layers to process the image data
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)

        # Number of features the last conv will output
        num_features_after_conv = 32 * 8 * 8

        # LiquidNet cell
        self.liquid_net = LiquidNet(num_units)

        # Linear layer
        self.fc = nn.Linear(num_units, num_classes)

        self.hidden_state = None

    def forward(self, x):
        # Apply convolutional layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten the output for the liquidnet
        x = x.view(x.size(0), -1)

        # Initialize the hidden state if it is the first pass
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(x.size(0), self.liquid_net.state_size).to(
                x.device
            )

        # Forward pass through the liquidnet
        x, self.hidden_state = self.liquid_net(x, self.hidden_state)

        # Classification layer
        x = self.fc(x)
        return x
