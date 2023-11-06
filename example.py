import torch
from liquidnet.main import LiquidNet

# Create an LiquidNet with a specified number of units
num_units = 64
ltc_cell = LiquidNet(num_units)

# Generate random input data with batch size 4 and input size 32
batch_size = 4
input_size = 32
inputs = torch.randn(batch_size, input_size)

# Initialize the cell state (hidden state)
initial_state = torch.zeros(batch_size, num_units)

# Forward pass through the LiquidNet
outputs, final_state = ltc_cell(inputs, initial_state)

# Print the shape of outputs and final_state
print("Outputs shape:", outputs.shape)
print("Final state shape:", final_state.shape)
