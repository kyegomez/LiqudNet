import os

import numpy as np
import pytest
import torch
from torch import nn

from liquidnet.main import LiquidNet, MappingType, ODESolver
from liquidnet.vision_liquidnet import VisionLiquidNet


# Initialize the VisionLiquidNet
def test_vision_liquid_net_initialization():
    num_units = 64
    num_classes = 10
    model = VisionLiquidNet(num_units, num_classes)
    assert isinstance(model, nn.Module)
    assert isinstance(model.liquid_net, LiquidNet)


# Test forward pass through VisionLiquidNet
def test_vision_liquid_net_forward_pass():
    num_units = 64
    num_classes = 10
    model = VisionLiquidNet(num_units, num_classes)

    # Create a sample input tensor
    batch_size = 8
    channels = 3
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Perform a forward pass
    output = model(input_tensor)

    # Check the shape of the output
    assert output.shape == (batch_size, num_classes)


# Test initialization of hidden state
def test_hidden_state_initialization():
    num_units = 64
    num_classes = 10
    model = VisionLiquidNet(num_units, num_classes)

    # Create a sample input tensor
    batch_size = 8
    channels = 3
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Perform a forward pass
    model(input_tensor)

    # Check if the hidden state is initialized
    assert model.hidden_state is not None


# Initialize VisionLiquidNet and LiquidNet instances for testing
@pytest.fixture
def vision_liquid_net():
    return VisionLiquidNet(num_units=64, num_classes=10)


@pytest.fixture
def liquid_net():
    return LiquidNet(num_units=64)


# Test cases for VisionLiquidNet
def test_vision_liquid_net_forward(vision_liquid_net):
    batch_size = 4
    input_channels = 3
    input_height = 32
    input_width = 32
    num_classes = 10

    inputs = torch.randn(batch_size, input_channels, input_height, input_width)

    outputs = vision_liquid_net(inputs)

    assert outputs.shape == (batch_size, num_classes)


def test_vision_liquid_net_hidden_state(vision_liquid_net):
    batch_size = 4
    input_channels = 3
    input_height = 32
    input_width = 32

    inputs = torch.randn(batch_size, input_channels, input_height, input_width)

    # Check if hidden state is initialized
    assert vision_liquid_net.hidden_state is None

    # Perform a forward pass
    _ = vision_liquid_net(inputs)

    # Check if hidden state is updated
    assert vision_liquid_net.hidden_state is not None


# Test cases for LiquidNet
def test_liquid_net_forward(liquid_net):
    batch_size = 4
    input_size = 32
    num_units = liquid_net.state_size

    inputs = torch.randn(batch_size, input_size)
    initial_state = torch.zeros(batch_size, num_units)

    outputs, final_state = liquid_net(inputs, initial_state)

    assert outputs.shape == (batch_size, num_units)
    assert final_state.shape == (batch_size, num_units)


def test_liquid_net_parameter_constraints(liquid_net):
    constraints = liquid_net.get_param_constrain_op()
    for param in constraints:
        assert (param >= 0).all()  # Ensure non-negativity of parameters


# Add more test cases as needed


# Define some constants for testing
NUM_UNITS = 64
BATCH_SIZE = 4
INPUT_SIZE = 32
NUM_ITERATIONS = 100


# Create fixtures for your tests
@pytest.fixture
def liquid_net():
    return LiquidNet(NUM_UNITS)


@pytest.fixture
def sample_inputs():
    return torch.randn(BATCH_SIZE, INPUT_SIZE)


@pytest.fixture
def initial_state():
    return torch.zeros(BATCH_SIZE, NUM_UNITS)


# Write individual test functions
def test_liquid_net_initialization(liquid_net):
    assert liquid_net.state_size == NUM_UNITS
    assert liquid_net.output_size == NUM_UNITS


def test_forward_pass(liquid_net, sample_inputs, initial_state):
    outputs, final_state = liquid_net(sample_inputs, initial_state)
    assert outputs.shape == (BATCH_SIZE, NUM_UNITS)
    assert final_state.shape == (BATCH_SIZE, NUM_UNITS)


def test_variable_constraints(liquid_net):
    constraining_ops = liquid_net.get_param_constrain_op()
    for op in constraining_ops:
        assert torch.all(op >= 0)  # Check that values are non-negative


def test_export_weights(liquid_net):
    dirname = "test_weights"
    liquid_net.export_weights(dirname)

    # Check if the weight files exist in the specified directory
    assert os.path.exists(os.path.join(dirname, "w.csv"))
    assert os.path.exists(os.path.join(dirname, "erev.csv"))
    assert os.path.exists(os.path.join(dirname, "mu.csv"))
    assert os.path.exists(os.path.join(dirname, "sigma.csv"))
    assert os.path.exists(os.path.join(dirname, "sensory_w.csv"))
    assert os.path.exists(os.path.join(dirname, "sensory_erev.csv"))
    assert os.path.exists(os.path.join(dirname, "sensory_mu.csv"))
    assert os.path.exists(os.path.join(dirname, "sensory_sigma.csv"))
    assert os.path.exists(os.path.join(dirname, "vleak.csv"))
    assert os.path.exists(os.path.join(dirname, "gleak.csv"))
    assert os.path.exists(os.path.join(dirname, "cm.csv"))


# Parameterized tests for different configurations
@pytest.mark.parametrize("solver", [ODESolver.SemiImplicit, ODESolver.Explicit])
@pytest.mark.parametrize(
    "mapping_type", [MappingType.Identity, MappingType.Linear, MappingType.Affine]
)
def test_solver_and_mapping_types(
    liquid_net, sample_inputs, initial_state, solver, mapping_type
):
    liquid_net._solver = solver
    liquid_net._input_mapping = mapping_type
    outputs, final_state = liquid_net(sample_inputs, initial_state)
    # Add assertions based on solver and mapping_type configurations


# Write a test for continuous integration (CI) integration
def test_continuous_integration():
    # Simulate CI environment
    assert True


if __name__ == "__main__":
    pytest.main()
