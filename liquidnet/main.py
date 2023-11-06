import os
from enum import Enum

import numpy as np
import torch
import torch.nn as nn


class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2


class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2


class LiquidNet(nn.Module):
    """
    Long short-term chaos cell (LiquidNet) as described in https://arxiv.org/abs/1905.12374

    Args:
        num_units: Number of units in the cell

    Attributes:
        state_size: Integer, the number of units in the cell
        output_size: Integer, the number of units in the cell

    Call arguments:
        inputs: A 2D tensor with shape [batch_size, input_size]
        state: A 2D tensor with shape [batch_size, state_size]

    Constants:
        _ode_solver_unfolds: Number of ODE solver steps in one RNN step
        _solver: ODE solver type
        _input_mapping: Input mapping type
        _erev_init_factor: Factor for the initial value of the reversal potential
        _w_init_max: Upper bound for the initial value of the synaptic weights
        _w_init_min: Lower bound for the initial value of the synaptic weights
        _cm_init_min: Lower bound for the initial value of the membrane capacitance
        _cm_init_max: Upper bound for the initial value of the membrane capacitance
        _gleak_init_min: Lower bound for the initial value of the leak conductance
        _gleak_init_max: Upper bound for the initial value of the leak conductance
        _w_min_value: Lower bound for the synaptic weights
        _w_max_value: Upper bound for the synaptic weights
        _gleak_min_value: Lower bound for the leak conductance
        _gleak_max_value: Upper bound for the leak conductance
        _cm_t_min_value: Lower bound for the membrane capacitance
        _cm_t_max_value: Upper bound for the membrane capacitance
        _fix_cm: Fix the membrane capacitance to a specific value
        _fix_gleak: Fix the leak conductance to a specific value
        _fix_vleak: Fix the leak reversal potential to a specific value

    Variables:
        sensory_mu: A 2D tensor with shape [input_size, state_size]
        sensory_sigma: A 2D tensor with shape [input_size, state_size]
        sensory_W: A 2D tensor with shape [input_size, state_size]
        sensory_erev: A 2D tensor with shape [input_size, state_size]
        mu: A 2D tensor with shape [state_size, state_size]
        sigma: A 2D tensor with shape [state_size, state_size]
        W: A 2D tensor with shape [state_size, state_size]
        erev: A 2D tensor with shape [state_size, state_size]
        vleak: A 1D tensor with shape [state_size]
        gleak: A 1D tensor with shape [state_size]
        cm_t: A 1D tensor with shape [state_size]

    Methods:
        _map_inputs: Maps the inputs to the correct dimensionality
        build: Builds the cell
        forward: Performs a forward pass through the cell
        _get_variables: Creates the torch parameters
        _ode_step: Performs a forward pass through the cell using the semi-implicit euler method
        _f_prime: Calculates the derivative of the cell
        _ode_step_runge_kutta: Performs a forward pass through the cell using the Runge-Kutta method
        _ode_step_explicit: Performs a forward pass through the cell using the explicit euler method
        _sigmoid: Calculates the sigmoid function
        get_param_constrain_op: Returns the operations to constrain the parameters to the specified bounds
        export_weights: Exports the weights of the cell to a specified directory


    Examples:
    >>> ltc_cell = LiquidNet(64)
    >>> batch_size = 4
    >>> input_size = 32
    >>> inputs = torch.randn(batch_size, input_size)
    >>> initial_state = torch.zeros(batch_size, num_units)
    >>> outputs, final_state = ltc_cell(inputs, initial_state)
    >>> print("Outputs shape:", outputs.shape)
    >>> print("Final state shape:", final_state.shape)
    Outputs shape: torch.Size([4, 64])
    Final state shape: torch.Size([4, 64])




    """

    def __init__(self, num_units):
        super(LiquidNet, self).__init__()

        self._input_size = -1
        self._num_units = num_units
        self._is_built = False

        # Number of ODE solver steps in one RNN step
        self._ode_solver_unfolds = 6
        self._solver = ODESolver.SemiImplicit

        self._input_mapping = MappingType.Affine

        self._erev_init_factor = 1

        self._w_init_max = 1.0
        self._w_init_min = 0.01
        self._cm_init_min = 0.5
        self._cm_init_max = 0.5
        self._gleak_init_min = 1
        self._gleak_init_max = 1

        self._w_min_value = 0.00001
        self._w_max_value = 1000
        self._gleak_min_value = 0.00001
        self._gleak_max_value = 1000
        self._cm_t_min_value = 0.000001
        self._cm_t_max_value = 1000

        self._fix_cm = None
        self._fix_gleak = None
        self._fix_vleak = None

    @property
    def state_size(self):
        """State size of the cell."""
        return self._num_units

    @property
    def output_size(self):
        """Output size of the cell."""
        return self._num_units

    def _map_inputs(self, inputs, resuse_scope=False):
        """Maps the inputs to the correct dimensionality"""
        if (
            self._input_mapping == MappingType.Affine
            or self._input_mapping == MappingType.Linear
        ):
            w = nn.Parameter(torch.ones(self._input_size))
            inputs = inputs * w
        if self._input_mapping == MappingType.Affine:
            b = nn.Parameter(torch.zeros(self._input_size))
            inputs = inputs + b
        return inputs

    def forward(self, inputs, state):
        """Forward pass through the cell"""
        if not self._is_built:
            # TODO: Move this part into the build method inherited form nn.Module
            self._is_built = True
            self._input_size = int(inputs.shape[-1])

            self._get_variables()

        elif self._input_size != int(inputs.shape[-1]):
            raise ValueError(
                "You first feed an input with {} features and now one with {} features, that is not possible".format(
                    self._input_size, int(inputs[-1])
                )
            )

        inputs = self._map_inputs(inputs)

        if self._solver == ODESolver.Explicit:
            next_state = self._ode_step_explicit(
                inputs, state, _ode_solver_unfolds=self._ode_solver_unfolds
            )
        elif self._solver == ODESolver.SemiImplicit:
            next_state = self._ode_step(inputs, state)
        elif self._solver == ODESolver.RungeKutta:
            next_state = self._ode_step_runge_kutta(inputs, state)
        else:
            raise ValueError("Unknown ODE solver '{}'".format(str(self._solver)))

        outputs = next_state

        return outputs, next_state

    # Create torch parameters
    def _get_variables(self):
        """Creates the torch parameters"""
        self.sensory_mu = nn.Parameter(
            torch.rand(self._input_size, self._num_units) * 0.5 + 0.3
        )
        self.sensory_sigma = nn.Parameter(
            torch.rand(self._input_size, self._num_units) * 5.0 + 3.0
        )
        self.sensory_W = nn.Parameter(
            torch.Tensor(
                np.random.uniform(
                    low=self._w_init_min,
                    high=self._w_init_max,
                    size=[self._input_size, self._num_units],
                )
            )
        )
        sensory_erev_init = (
            2
            * np.random.randint(low=0, high=2, size=[self._input_size, self._num_units])
            - 1
        )
        self.sensory_erev = nn.Parameter(
            torch.Tensor(sensory_erev_init * self._erev_init_factor)
        )

        self.mu = nn.Parameter(torch.rand(self._num_units, self._num_units) * 0.5 + 0.3)
        self.sigma = nn.Parameter(
            torch.rand(self._num_units, self._num_units) * 5.0 + 3.0
        )
        self.W = nn.Parameter(
            torch.Tensor(
                np.random.uniform(
                    low=self._w_init_min,
                    high=self._w_init_max,
                    size=[self._num_units, self._num_units],
                )
            )
        )

        erev_init = (
            2
            * np.random.randint(low=0, high=2, size=[self._num_units, self._num_units])
            - 1
        )
        self.erev = nn.Parameter(torch.Tensor(erev_init * self._erev_init_factor))

        if self._fix_vleak is None:
            self.vleak = nn.Parameter(torch.rand(self._num_units) * 0.4 - 0.2)
        else:
            self.vleak = nn.Parameter(torch.Tensor(self._fix_vleak))

        if self._fix_gleak is None:
            if self._gleak_init_max > self._gleak_init_min:
                self.gleak = nn.Parameter(
                    torch.rand(self._num_units)
                    * (self._gleak_init_max - self._gleak_init_min)
                    + self._gleak_init_min
                )
            else:
                self.gleak = nn.Parameter(
                    torch.Tensor([self._gleak_init_min] * self._num_units)
                )
        else:
            self.gleak = nn.Parameter(torch.Tensor(self._fix_gleak))

        if self._fix_cm is None:
            if self._cm_init_max > self._cm_init_min:
                self.cm_t = nn.Parameter(
                    torch.rand(self._num_units)
                    * (self._cm_init_max - self._cm_init_min)
                    + self._cm_init_min
                )
            else:
                self.cm_t = nn.Parameter(
                    torch.Tensor([self._cm_init_min] * self._num_units)
                )
        else:
            self.cm_t = nn.Parameter(torch.Tensor(self._fix_cm))

    # Hybrid euler method
    def _ode_step(self, inputs, state):
        """ODE solver step"""
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

            rev_activation = w_activation * self.erev

            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator / denominator

        return v_pre

    def _f_prime(self, inputs, state):
        """Calculates the derivative of the cell"""
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

            w_reduced_synapse = torch.sum(w_activation, dim=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = (
                torch.sum(sensory_in, dim=1)
                - v_pre * w_reduced_synapse
                + torch.sum(synapse_in, dim=1)
                - v_pre * w_reduced_sensory
            )

            f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

            v_pre = v_pre + 0.1 * f_prime

        return f_prime

    def _ode_step_runge_kutta(self, inputs, state):
        """ODE solver step"""
        h = 0.1
        for i in range(self._ode_solver_unfolds):
            k1 = h * self._f_prime(inputs, state)
            k2 = h * self._f_prime(inputs, state + k1 * 0.5)
            k3 = h * self._f_prime(inputs, state + k2 * 0.5)
            k4 = h * self._f_prime(inputs, state + k3)

            state = state + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return state

    def _ode_step_explicit(self, inputs, state, _ode_solver_unfolds):
        """ODE solver step"""
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(_ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

            w_reduced_synapse = torch.sum(w_activation, dim=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = (
                torch.sum(sensory_in, dim=1)
                - v_pre * w_reduced_synapse
                + torch.sum(synapse_in, dim=1)
                - v_pre * w_reduced_sensory
            )

            f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

            v_pre = v_pre + 0.1 * f_prime

        return v_pre

    def _sigmoid(self, v_pre, mu, sigma):
        """Calculates the sigmoid function"""
        v_pre = v_pre.view(-1, v_pre.shape[-1], 1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def get_param_constrain_op(self):
        """Returns the operations to constrain the parameters to the specified bounds"""

        cm_clipping_op = torch.clamp(
            self.cm_t, self._cm_t_min_value, self._cm_t_max_value
        )
        gleak_clipping_op = torch.clamp(
            self.gleak, self._gleak_min_value, self._gleak_max_value
        )
        w_clipping_op = torch.clamp(self.W, self._w_min_value, self._w_max_value)
        sensory_w_clipping_op = torch.clamp(
            self.sensory_W, self._w_min_value, self._w_max_value
        )

        return [cm_clipping_op, gleak_clipping_op, w_clipping_op, sensory_w_clipping_op]

    def export_weights(self, dirname, output_weights=None):
        """Exports the weights of the cell to a specified directory"""
        os.makedirs(dirname, exist_ok=True)
        w, erev, mu, sigma = (
            self.W.data.cpu().numpy(),
            self.erev.data.cpu().numpy(),
            self.mu.data.cpu().numpy(),
            self.sigma.data.cpu().numpy(),
        )
        sensory_w, sensory_erev, sensory_mu, sensory_sigma = (
            self.sensory_W.data.cpu().numpy(),
            self.sensory_erev.data.cpu().numpy(),
            self.sensory_mu.data.cpu().numpy(),
            self.sensory_sigma.data.cpu().numpy(),
        )
        vleak, gleak, cm = (
            self.vleak.data.cpu().numpy(),
            self.gleak.data.cpu().numpy(),
            self.cm_t.data.cpu().numpy(),
        )

        if output_weights is not None:
            output_w, output_b = output_weights
            np.savetxt(
                os.path.join(dirname, "output_w.csv"), output_w.data.cpu().numpy()
            )
            np.savetxt(
                os.path.join(dirname, "output_b.csv"), output_b.data.cpu().numpy()
            )
        np.savetxt(os.path.join(dirname, "w.csv"), w)
        np.savetxt(os.path.join(dirname, "erev.csv"), erev)
        np.savetxt(os.path.join(dirname, "mu.csv"), mu)
        np.savetxt(os.path.join(dirname, "sigma.csv"), sigma)
        np.savetxt(os.path.join(dirname, "sensory_w.csv"), sensory_w)
        np.savetxt(os.path.join(dirname, "sensory_erev.csv"), sensory_erev)
        np.savetxt(os.path.join(dirname, "sensory_mu.csv"), sensory_mu)
        np.savetxt(os.path.join(dirname, "sensory_sigma.csv"), sensory_sigma)
        np.savetxt(os.path.join(dirname, "vleak.csv"), vleak)
        np.savetxt(os.path.join(dirname, "gleak.csv"), gleak)
        np.savetxt(os.path.join(dirname, "cm.csv"), cm)