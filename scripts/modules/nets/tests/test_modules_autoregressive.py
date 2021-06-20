#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test layers in modules.autoregressive.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from unittest import mock

import numpy as np
import pytest
import torch

from ..modules.autoregressive import MADE


# =============================================================================
# FIXTURES
# =============================================================================

# Each test case is a tuple:
# (dimension_in, dimensions_hidden, out_per_dimension, dimension_conditioning,
#       expected_dimensions_hidden, expected_out_dimensions)
@pytest.fixture(
    params=[
        (3, 2, 1, 0,
            [2, 2], 3),
        (3, [5], 1, 0,
            [5], 3),
        (3, 4, 1, 1,
            [2]*4, 2),
        (5, 7, 2, 0,
            [8]*7, 10),
        (5, 7, 2, 2,
            [8]*7, 6),
        (5, [4, 7, 9], 2, 2,
            [4, 7, 9], 6)
    ]
)
def dimensions(request):
    return request.param

# Each test case is a tuple:
# (blocks, degrees_in,
#       dimension_in, dimensions_hidden, out_per_dimension, dimension_conditioning,
#       expected_dimensions_hidden, expected_out_dimensions)
@pytest.fixture(
    params=[
        (2, 'input',
            3, 2, 1, 0,
            [2, 2], 3),
        (2, 'reversed',
            3, 2, 1, 0,
            [1, 1], 3),
        ([1, 2], 'input',
            3, 2, 1, 0,
            [1, 1], 3),
        ([1, 2], 'reversed',
            3, 2, 1, 0,
            [2, 2], 3),
        (3, 'input',
            7, 3, 2, 2,
            [10]*3, 10),
        (3, 'reversed',
            7, 3, 2, 2,
            [8]*3, 10),
        ([1, 2, 2], 'input',
            7, 3, 2, 2,
            [10]*3, 10),
        ([1, 2, 2], 'reversed',
            7, 3, 2, 2,
            [12]*3, 10),
        ([2, 3], 'input',
            7, 3, 2, 2,
            [8]*3, 10),
        ([2, 3], 'reversed',
            7, 3, 2, 2,
            [10]*3, 10),
        (2, 'input',
            7, [6, 9, 11], 2, 2,
            [6, 9, 11], 10),
        (2, 'reversed',
            7, [6, 9, 11], 2, 2,
            [6, 9, 11], 10),
        ([1, 2, 3], np.array([0, 1, 2]),
            8, 3, 2, 2,
            [10]*3, 12),
        ([1, 2, 3], np.array([0, 2, 1]),
            8, 3, 2, 2,
            [12]*3, 12),
        ([1, 2, 3], np.array([2, 0, 1]),
            8, 3, 2, 2,
            [14]*3, 12),
    ]
)
def blocked_dimensions(request):
    return request.param


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_random_degrees_in(dimension_in, dimension_conditioning):
    # Make sure the test is reproducible with a random state.
    random_state = np.random.RandomState(dimension_in)
    return random_state.permutation(list(range(dimension_in-dimension_conditioning)))


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('in_out_dimension,n_layers,dimension_conditioning', [
    (5, 3, 0),
    (7, 2, 2),
    (7, 4, 1),
    (10, 3, 3),
])
def test_MADE_create_mask(in_out_dimension, n_layers, dimension_conditioning):
    """Test the method MADE.create_mask().

    Simulate a 3-layer network with sequential degree assignment and
    check that all the masks have the appropriate shape and are lower
    triangular.

    """
    first_layer_dim = in_out_dimension
    inner_layer_dim = in_out_dimension - 1
    output_layer_dim = in_out_dimension - dimension_conditioning

    # Assign degrees sequentially for the simulated 3-layer network.
    degrees = []
    for layer_idx in range(n_layers+1):
        # The first and last layers have an extra unit.
        if layer_idx == 0:
            degrees.append(np.arange(first_layer_dim))
        elif layer_idx == n_layers:
            degrees.append(np.arange(dimension_conditioning, dimension_conditioning+output_layer_dim))
        else:
            degrees.append(np.arange(inner_layer_dim))

    # Build masks for all 3 layers.
    masks = [MADE.create_mask(degrees[i], degrees[i+1], is_output_layer=(i==n_layers-1))
             for i in range(n_layers)]

    for layer_idx, mask in enumerate(masks):
        is_output_layer = (layer_idx == n_layers-1)

        # Check that they are all lower triangular.
        if is_output_layer:
            assert torch.all(mask == torch.tril(mask, diagonal=dimension_conditioning-1))
        else:
            assert torch.all(mask == torch.tril(mask))

        # In the first layer, the last input unit must have no
        # connection with the first hidden layer.
        if layer_idx == 0:
            assert torch.all(mask[:,-1] == False)
            assert mask.shape == (inner_layer_dim, first_layer_dim)

        # In the last layer, the first output unit must be attached
        # only to conditioning node or be constant.
        elif is_output_layer:
            assert torch.all(mask[0, dimension_conditioning:] == False)
            assert mask.shape == (output_layer_dim, inner_layer_dim)
        else:
            assert mask.shape == (inner_layer_dim, inner_layer_dim)


@pytest.mark.parametrize('degrees_in', ['input', 'reversed', 'random'])
def test_MADE_get_dimensions(degrees_in, dimensions):
    """Test the method MADE._get_dimensions without blocks.

    The dimensions should be independent of the degrees_in option.
    """
    if degrees_in == 'random':
        degrees_in = generate_random_degrees_in(dimensions[0], dimensions[3])
    check_MADE_get_dimensions(1, degrees_in, *dimensions)


def test_MADE_get_dimensions_blocks(blocked_dimensions):
    """Test the method MADE._get_dimensions with blocks."""
    check_MADE_get_dimensions(*blocked_dimensions)


def check_MADE_get_dimensions(
        blocks, degrees_in, dimension_in, dimensions_hidden, out_per_dimension, dimension_conditioning,
        expected_dimensions_hidden, expected_out_dimensions
):
    """Used by test_MADE_get_dimensions and test_MADE_get_dimensions_blocks."""
    n_hidden_layers, dimensions_hidden, out_dimension, expanded_blocks = MADE._get_dimensions(
        dimension_in, dimensions_hidden, out_per_dimension, dimension_conditioning,
        degrees_in, blocks, shorten_last_block=True)

    assert n_hidden_layers == len(expected_dimensions_hidden)
    assert dimensions_hidden == expected_dimensions_hidden
    assert out_dimension == expected_out_dimensions


@pytest.mark.parametrize(('dimension_in,dimension_conditioning,degrees_in,degrees_hidden_motif,blocks,'
                                'expected_degrees_in,expected_degrees_hidden_motif'), [
    (5, 0, 'input', None, [3, 2],
        [0, 0, 0, 1, 1], [0, 0, 0]),
    (7, 2, 'input', None, [3, 2],
        [-1, -1, 0, 0, 0, 1, 1], [-1, -1, 0, 0, 0]),
    (5, 0, 'reversed', None, [3, 2],
        [1, 1, 1, 0, 0], [0, 0]),
    (7, 2, 'reversed', None, [3, 2],
        [-1, -1, 1, 1, 1, 0, 0], [-1, -1, 0, 0]),
    (6, 0, [2, 0, 1], None, [1, 3, 2],
        [2, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1]),
    (7, 1, [2, 0, 1], None, [1, 3, 2],
        [-1, 2, 0, 0, 0, 1, 1], [-1, 0, 0, 0, 1, 1]),
])
def test_MADE_generate_degrees(dimension_in, dimension_conditioning, degrees_in, degrees_hidden_motif, blocks,
                               expected_degrees_in, expected_degrees_hidden_motif):
    """Test that the input degrees and the motif for the hidden nodes are correct."""
    # Create a mock MADE class with the blocks attribute.
    mock_made = mock.Mock(blocks=blocks)
    mock_made.degrees_in = MADE._assign_degrees_in(mock_made, dimension_in, dimension_conditioning, degrees_in)
    motif = MADE._generate_degrees_hidden_motif(mock_made, degrees_hidden_motif)

    assert np.all(mock_made.degrees_in == np.array(expected_degrees_in))
    assert np.all(motif == np.array(expected_degrees_hidden_motif))


@pytest.mark.parametrize('weight_norm', [False, True])
def test_MADE_mask_dimensions(weight_norm, dimensions):
    """Test that the dimension of the hidden layers without blocks follow the init options correctly."""
    check_MADE_mask_dimensions(1, 'input', *dimensions[:-2], weight_norm=weight_norm)


@pytest.mark.parametrize('weight_norm', [False, True])
def test_MADE_mask_dimensions_blocks(weight_norm, blocked_dimensions):
    """Test that the dimension of the hidden layers with blocks follow the init options correctly."""
    check_MADE_mask_dimensions(*blocked_dimensions[:-2], weight_norm=weight_norm)


def check_MADE_mask_dimensions(blocks, degrees_in, dimension_in, dimensions_hidden,
                               out_per_dimension, dimension_conditioning, weight_norm):
    """Used by test_MADE_mask_dimensions and test_MADE_mask_dimensions_blocks."""
    made = MADE(
        dimension_in=dimension_in,
        dimensions_hidden=dimensions_hidden,
        out_per_dimension=out_per_dimension,
        dimension_conditioning=dimension_conditioning,
        degrees_in=degrees_in,
        weight_norm=weight_norm,
        blocks=blocks,
        shorten_last_block=True
    )

    # Compute the expected dimensions.
    n_hidden_layers, dimensions_hidden, out_dimension, expanded_blocks = made._get_dimensions(
        dimension_in, dimensions_hidden, out_per_dimension, dimension_conditioning,
        degrees_in, blocks, shorten_last_block=True)

    # Masked linear layers are alternated with nonlinearities.
    masked_linear_modules = made.layers[::2]

    # Check all dimensions.
    assert len(masked_linear_modules) == n_hidden_layers + 1
    assert masked_linear_modules[0].in_features == dimension_in
    for layer_idx in range(n_hidden_layers):
        masked_linear_modules[layer_idx].out_features == dimensions_hidden[layer_idx]
        masked_linear_modules[layer_idx+1].in_features == dimensions_hidden[layer_idx]
    assert masked_linear_modules[-1].out_features == out_dimension

    # Test correct implementation of the Python properties.
    assert made.dimension_in == dimension_in
    assert made.n_layers == n_hidden_layers + 1
    assert made.dimensions_hidden == dimensions_hidden
    assert made.dimension_conditioning == dimension_conditioning


@pytest.mark.parametrize('weight_norm', [False, True])
@pytest.mark.parametrize('degrees_in', ['input', 'reversed', 'random'])
def test_MADE_autoregressive_property(weight_norm, degrees_in, dimensions):
    """Test that MADE without blocks satisfies the autoregressive property.

    The test creates a random input for a MADE network and then perturbs
    it one a time, making sure that output k changes if and only if
    input with a smaller degrees have changed.

    """
    # Generate a random permutation if requested.
    if degrees_in == 'random':
        degrees_in = generate_random_degrees_in(dimensions[0], dimensions[3])
    check_MADE_autoregressive_property(1, degrees_in, *dimensions, weight_norm=weight_norm)


@pytest.mark.parametrize('weight_norm', [False, True])
def test_MADE_autoregressive_property_blocks(weight_norm, blocked_dimensions):
    """Test that MADE with blocks satisfies the autoregressive property.

    The test creates a random input for a MADE network and then perturbs
    it one a time, making sure that output k changes if and only if
    input with a smaller degrees have changed.

    """
    check_MADE_autoregressive_property(*blocked_dimensions, weight_norm=weight_norm)


def check_MADE_autoregressive_property(blocks, degrees_in, dimension_in, dimensions_hidden,
                                       out_per_dimension, dimension_conditioning, _, out_dimension, weight_norm):
    """Used by test_MADE_autoregressive_property and test_MADE_autoregressive_property_blocks."""
    made = MADE(
        dimension_in=dimension_in,
        dimensions_hidden=dimensions_hidden,
        out_per_dimension=out_per_dimension,
        dimension_conditioning=dimension_conditioning,
        degrees_in=degrees_in,
        blocks=blocks,
        shorten_last_block=True
    )

    # Create a random input and make it go through the net.
    x = np.random.randn(1, dimension_in)
    input = torch.tensor(x, dtype=torch.float, requires_grad=True)
    output = made.forward(input)
    assert output.shape == (1, out_dimension)

    # Make sure that there are no duplicate degrees in the input/output.
    assert len(set(made.degrees_in)) == len(made.blocks) + int(dimension_conditioning > 0)

    for out_idx in range(out_dimension // out_per_dimension):
        # Compute the gradient of the out_idx-th dimension of the
        # output with respect to the gradient vector.
        loss = torch.sum(output[0, out_idx:out_dimension:out_dimension//out_per_dimension])
        loss.backward(retain_graph=True)

        # In all cases, the conditioning features should affect the whole output.
        grad = input.grad[0]
        assert torch.all(grad[:dimension_conditioning] != 0.0)

        # Now consider the non-conditioning features only.
        grad = grad[dimension_conditioning:]
        degrees = made.degrees_in[dimension_conditioning:]

        # For the autoregressive property to hold, the k-th output should
        # have non-zero gradient only for the inputs with a smaller degree.
        degree_out = degrees[out_idx]
        for in_idx in range(len(degrees)):
            if degrees[in_idx] < degree_out:
                assert grad[in_idx] != 0
            else:
                assert grad[in_idx] == 0

        # Reset gradients for next iteration.
        made.zero_grad()
        input.grad.data.zero_()
