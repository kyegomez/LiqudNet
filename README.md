[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# LiquidNet
This is a simple implementation of the Liquid net official repo translated into pytorch for simplicity. [Find the original repo here:](https://github.com/raminmh/liquid_time_constant_networks)

## Install
`pip install liquid-net`

## Usage
```python
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

```


# Citation
```bibtex
@article{DBLP:journals/corr/abs-2006-04439,
  author       = {Ramin M. Hasani and
                  Mathias Lechner and
                  Alexander Amini and
                  Daniela Rus and
                  Radu Grosu},
  title        = {Liquid Time-constant Networks},
  journal      = {CoRR},
  volume       = {abs/2006.04439},
  year         = {2020},
  url          = {https://arxiv.org/abs/2006.04439},
  eprinttype    = {arXiv},
  eprint       = {2006.04439},
  timestamp    = {Fri, 12 Jun 2020 14:02:57 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2006-04439.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

```


# License
MIT


# Todo:
- [ ] Implement LiquidNet for vision and train on CIFAR
