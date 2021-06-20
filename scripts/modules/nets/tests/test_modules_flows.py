#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test layers in modules.flows.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch

from ..modules.flows import (AffineTransformer, SOSPolynomialTransformer,
                             NeuralSplineTransformer, MobiusTransformer)
from ..modules.flows import InvertibleBatchNorm1d, MAF, NormalizingFlow
from ..utils import generate_block_sizes


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_random_input(batch_size, dimension, n_parameters=0, gate=None,
                        dtype=None, seed=0, x_func=torch.randn, par_func=torch.randn):
    """Create input, parameters and gates.

    Parameters
    ----------
    gate : bool, optional
        If False, the returned gate will be None. Otherwise a random
        gate is generated. If not passed, the gate parameter is not
        returned.

    """
    # Make sure the input arguments are deterministic.
    generator = torch.Generator()
    generator.manual_seed(seed)

    if dtype is None:
        dtype = torch.get_default_dtype()

    x = x_func(batch_size, dimension, dtype=dtype, generator=generator, requires_grad=True)
    returned_values = [x]

    if n_parameters > 0:
        parameters = par_func(batch_size, n_parameters, dimension, dtype=dtype,
                              generator=generator, requires_grad=True)
        returned_values.append(parameters)

    if gate is not None:
        # Then we need to return a gating parameter.
        if gate is True:
            gate = torch.rand(batch_size, dtype=dtype, generator=generator, requires_grad=True)

            # Set first and second batch to gate None (i.e., 1) and 0.0.
            gate.data[0] = 1.0
            gate.data[1] = 0.0
        else:
            gate = None

        returned_values.append(gate)

    if len(returned_values) == 1:
        return returned_values[0]
    return returned_values

# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('n_polynomials', [2, 3])
def test_sos_affine_transformer_equivalence(n_polynomials):
    """The SOS polynomial is a generalization of the affine transformer."""
    batch_size = 2
    dimension = 5

    # Make sure the input arguments are deterministic.
    generator = torch.Generator()
    generator.manual_seed(0)

    # Create random input.
    x, affine_parameters = create_random_input(batch_size, dimension, n_parameters=2, dtype=torch.double)

    # Create coefficients for an SOS polynomial that translate into an affine transformer.
    sos_coefficients = torch.zeros(
        size=(batch_size, 1 + n_polynomials*2, dimension), dtype=torch.double)
    sos_coefficients[:, 0] = affine_parameters[:, 0].clone()

    # Divide the scale coefficient equally among all polynomials.
    # The affine transformer takes the log scale as input parameter.
    scale = torch.sqrt(torch.exp(affine_parameters[:, 1]) / n_polynomials)
    for poly_idx in range(n_polynomials):
        sos_coefficients[:, 1 + poly_idx*2] = scale.clone()

    # Check that they are equivalent.
    affine_y, affine_log_det_J = AffineTransformer()(x, affine_parameters)
    sos_y, sos_log_det_J = SOSPolynomialTransformer(n_polynomials=n_polynomials)(x, sos_coefficients)

    assert torch.allclose(affine_y, sos_y)
    assert torch.allclose(affine_log_det_J, sos_log_det_J)


@pytest.mark.parametrize('transformer', [
    AffineTransformer(),
    SOSPolynomialTransformer(n_polynomials=2),
    MobiusTransformer(blocks=[3, 2])
])
def test_gate_parameter(transformer):
    """Test that turning on the gate parameter will return the identity function."""
    batch_size = 2
    dimension = 5

    # Make sure the input arguments are deterministic.
    generator = torch.Generator()
    generator.manual_seed(0)

    # Create random input.
    x, parameters = create_random_input(batch_size, dimension, dtype=torch.double,
                                        n_parameters=transformer.n_parameters_per_input)

    for func in [transformer, transformer.inv]:
        gate0 = torch.zeros(batch_size, dtype=x.dtype)
        gate1 = torch.ones_like(gate0)

        # Check that when gate is off the transformation is
        # not the identity or the test has no meaning.
        try:
            y, log_det_J = func(x, parameters, gate=gate1)
            assert not torch.allclose(x, y)
            y, log_det_J = func(x, parameters, gate=gate0)
            assert torch.allclose(x, y)
        except NotImplementedError:
            pass


def test_identity_initialization_batch_norm():
    """Test the initialization of batch norm to the identity function."""
    batch_size = 4
    dimension = 5

    # Create random input.
    x = create_random_input(batch_size, dimension)

    batch_norm = InvertibleBatchNorm1d(dimension, initialize_identity=True)
    y, log_det_J = batch_norm(x)

    assert torch.allclose(x, y)
    assert torch.allclose(log_det_J, torch.zeros(batch_size), atol=1e-6)


def test_identity_batch_norm():
    """Test that inv(forward()) is the identity function."""
    batch_size = 4
    dimension = 5

    # Create random input.
    x = create_random_input(batch_size, dimension)

    # Initialize weight and bias to random numbers.
    batch_norm = InvertibleBatchNorm1d(dimension, initialize_identity=False)
    y, log_det_J = batch_norm(x)

    # We don't support training with inversion.
    batch_norm.eval()

    # Set the running mean and variance to the values used for forward.
    batch_norm.running_mean.data = torch.mean(x, dim=0).detach()
    batch_norm.running_var.data = torch.var(x, dim=0, unbiased=False).detach()
    x_inv, log_det_J_inv = batch_norm.inv(y)

    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros(batch_size))


@pytest.mark.parametrize('dimensions_hidden', [1, 4])
@pytest.mark.parametrize('dimension_conditioning', [0, 2])
@pytest.mark.parametrize('degrees_in', ['input', 'reversed'])
@pytest.mark.parametrize('weight_norm', [False, True])
@pytest.mark.parametrize('split_conditioner', [True, False])
@pytest.mark.parametrize('transformer', [
    AffineTransformer(),
    SOSPolynomialTransformer(2),
    SOSPolynomialTransformer(3),
    NeuralSplineTransformer(x0=torch.tensor(-2), xf=torch.tensor(2), n_bins=3),
    MobiusTransformer(blocks=3, shorten_last_block=True)
])
@pytest.mark.parametrize('gate', [False, True])
def test_identity_initialization_MAF(dimensions_hidden, dimension_conditioning, degrees_in,
                                     weight_norm, split_conditioner, transformer, gate):
    """Test that the identity initialization of MAF works.

    This tests both that the flow layers can be initialized to perform
    the identity function and that the gate parameter is used to force
    the layer to be skipped.

    """
    dimension = 5
    batch_size = 2

    if gate:
        gate = torch.zeros(batch_size)
        initialize_identity = False
    else:
        gate = None
        initialize_identity = True

    # We don't initialize as the identity function to make the test meaningful.
    maf = MAF(
        dimension,
        dimensions_hidden,
        dimension_conditioning=dimension_conditioning,
        degrees_in=degrees_in,
        weight_norm=weight_norm,
        split_conditioner=split_conditioner,
        transformer=transformer,
        initialize_identity=initialize_identity
    )

    # Create random input.
    if isinstance(transformer, NeuralSplineTransformer):
        x = create_random_input(batch_size, dimension, x_func=torch.rand)
        x = x * (transformer.xf - transformer.x0) + transformer.x0
    else:
        x = create_random_input(batch_size, dimension)

    try:
        y, log_det_J = maf.forward(x, gate=gate)
    except NotImplementedError:
        # Some transformer may not support "gate".
        if gate is not None:
            return
        raise

    assert torch.allclose(x, y)
    assert torch.allclose(log_det_J, torch.zeros(batch_size), atol=1e-6)


@pytest.mark.parametrize('dimension_conditioning', [0, 2])
@pytest.mark.parametrize('degrees_in', ['input', 'reversed', 'random'])
@pytest.mark.parametrize('split_conditioner', [True, False])
@pytest.mark.parametrize('transformer', [
    AffineTransformer(),
    MobiusTransformer(blocks=3, shorten_last_block=True)
])
@pytest.mark.parametrize('weight_norm', [False, True])
@pytest.mark.parametrize('gate', [False, True])
def test_round_trip_MAF(dimension_conditioning, degrees_in, weight_norm, split_conditioner, transformer, gate):
    """Test that the MAF.inv(MAF.forward(x)) equals the identity."""
    dimension = 5
    dimensions_hidden = 2
    batch_size = 2

    # Temporarily set default precision to double to improve comparisons.
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)

    # With the Mobius transformer, we need block dependencies.
    if isinstance(transformer, MobiusTransformer):
        blocks = generate_block_sizes(dimension-dimension_conditioning, transformer.blocks,
                                      transformer.shorten_last_block)
        shorten_last_block = transformer.shorten_last_block
        n_blocks = len(blocks)
    else:
        blocks = 1
        shorten_last_block = False
        n_blocks = dimension - dimension_conditioning

    # Make sure the permutation is reproducible.
    if degrees_in == 'random':
        random_state = np.random.RandomState(0)
        degrees_in = random_state.permutation(range(n_blocks))

    # We don't initialize as the identity function to make the test meaningful.
    maf = MAF(
        dimension, dimensions_hidden,
        dimension_conditioning=dimension_conditioning,
        degrees_in=degrees_in,
        weight_norm=weight_norm,
        blocks=blocks,
        shorten_last_block=shorten_last_block,
        split_conditioner=split_conditioner,
        transformer=transformer,
        initialize_identity=False
    )

    # Create random input.
    x, gate = create_random_input(batch_size, dimension, gate=gate)

    # The conditioning features are always left unchanged.
    y, log_det_J = maf.forward(x, gate=gate)
    assert torch.allclose(x[:, :dimension_conditioning], y[:, :dimension_conditioning])

    # Inverting the transformation produces the input vector.
    x_inv, log_det_J_inv = maf.inv(y, gate=gate)
    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros(batch_size), atol=1e-04)

    # Restore default dtype.
    torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('dimension', [5, 7])
@pytest.mark.parametrize('constant_input_indices', [None, [1]])
@pytest.mark.parametrize('dimension_gate', [1, 2])
@pytest.mark.parametrize('weight_norm', [False, True])
def test_normalizing_flow_gate_network(batch_size, dimension, constant_input_indices, dimension_gate, weight_norm):
    """Test that the dense network correctly generates the gate parameters.

    The test creates a normalizing flows with a gate, and it checks that
    setting the parameters of the dense network so that gate is 0 recovers
    the identity function.

    """
    if constant_input_indices is None:
        n_constant_input_indices = 0
    else:
        n_constant_input_indices = len(constant_input_indices)

    # Add a stack of three MAF layers
    flows = []
    for degrees_in in ['input', 'reversed', 'input']:
        # We don't initialize as the identity function to make the test meaningful.
        flows.append(MAF(
            dimension=dimension - n_constant_input_indices,
            dimensions_hidden=9,
            dimension_conditioning=dimension_gate,
            degrees_in=degrees_in,
            weight_norm=weight_norm,
            initialize_identity=False
        ))
    flow = NormalizingFlow(
        *flows,
        constant_indices=constant_input_indices,
        dimension_gate=dimension_gate,
        weight_norm=weight_norm,
    )

    # The gate network has the correct size.
    assert flow.gate_net[0].in_features == dimension_gate
    assert flow.gate_net[-2].out_features == 3

    # First check that by default the function implemented
    # by the flow is not the identity.
    x = create_random_input(batch_size, dimension)
    y, log_det_J = flow.forward(x)
    assert not torch.allclose(x, y)

    # Now set the parameters of the last layer so that the
    # output before the final sigmoid is very negative.
    if weight_norm:
        flow.gate_net[-2].weight_g.data.fill_(0.0)
    else:
        flow.gate_net[-2].weight.data.fill_(0.0)
    flow.gate_net[-2].bias.data.fill_(-10000)

    # Now the function should be very close to identity
    y, log_det_J = flow.forward(x)
    assert torch.allclose(x, y)


@pytest.mark.parametrize('constant_input_indices', [None, [1, 4]])
@pytest.mark.parametrize('dimension_gate', [0, 1, 2])
@pytest.mark.parametrize('weight_norm', [False, True])
def test_round_trip_NormalizingFlow(constant_input_indices, dimension_gate, weight_norm):
    """Test that the NormalizingFlow.inv(NormalizingFlow.forward(x)) equals the identity."""
    dimension = 7
    dimensions_hidden = 2
    batch_size = 2

    if constant_input_indices is None:
        n_constant_input_indices = 0
    else:
        n_constant_input_indices = len(constant_input_indices)

    # Add a stack of three MAF layers
    flows = []
    for degrees_in in ['input', 'reversed', 'input']:
        # We don't initialize as the identity function to make the test meaningful.
        flows.append(MAF(
            dimension=dimension - n_constant_input_indices,
            dimensions_hidden=dimensions_hidden,
            dimension_conditioning=dimension_gate,
            degrees_in=degrees_in,
            weight_norm=weight_norm,
            initialize_identity=False
        ))
    flow = NormalizingFlow(
        *flows,
        constant_indices=constant_input_indices,
        dimension_gate=dimension_gate,
        weight_norm=weight_norm,
    )

    # Create random input.
    x = create_random_input(batch_size, dimension)
    y, log_det_J = flow.forward(x)
    x_inv, log_det_J_inv = flow.inv(y)

    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros(batch_size), atol=1e-04)


def test_constant_input_multiscale_architect():
    """Test the constant_input_indices multiscale architecture."""
    dimension = 10
    dimensions_hidden = 2
    batch_size = 5
    constant_input_indices = [0, 1, 4, 7]

    # Create a three-layer MAF flow.
    flows = []
    for degrees_in in ['input', 'reversed', 'input']:
        # We don't initialize as the identity function to make the test meaningful.
        flows.append(MAF(
            dimension=dimension - len(constant_input_indices),
            dimensions_hidden=dimensions_hidden,
            degrees_in=degrees_in,
            initialize_identity=False
        ))
    flow = NormalizingFlow(
        *flows,
        constant_indices=constant_input_indices
    )

    # Create random input.
    x = create_random_input(batch_size, dimension)

    # Make sure the flow is not the identity
    # function or the test doesn't make sense.
    y, log_det_J = flow.forward(x)
    assert not torch.allclose(x, y)

    # The gradient of the constant input should always be 0.0.
    loss = torch.sum(y)
    loss.backward()
    assert torch.all(x.grad[:, constant_input_indices] == 1.0)

    # Make sure the inverse also works.
    x_inv, log_det_J_inv = flow.inv(y)
    assert torch.allclose(x, x_inv, atol=1e-04)
    assert torch.allclose(log_det_J + log_det_J_inv, torch.zeros(batch_size), atol=1e-04)
