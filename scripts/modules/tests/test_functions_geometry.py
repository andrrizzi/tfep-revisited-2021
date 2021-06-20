#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module functions.geometry.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pint
import pytest
import torch

from ..functions.geometry import (to_batch_atom_3_shape, to_atom_batch_3_shape,
                                  batchwise_atom_dist)


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

_ureg = pint.UnitRegistry()


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_positions,batch_atom_size', [
    # Numpy arrays.
    [np.zeros((2, 5, 3)), None],
    [np.zeros((4, 6*3)), (4, 6)],
    [np.zeros((5, 3)), (1, 5)],
    # Pint Quantities.
    [np.zeros((2, 5, 3)) * _ureg.angstrom, None],
    [np.zeros((4, 6*3)) * _ureg.angstrom, (4, 6)],
    [np.zeros((5, 3)) * _ureg.angstrom, (1, 5)],
    # Tensors.
    [torch.zeros((2, 5, 3)), None],
    [torch.zeros((4, 6*3)), (4, 6)],
    [torch.zeros((5, 3)), (1, 5)],
])
@pytest.mark.parametrize('reshape_function', [
    # to_batch_atom_3_shape,
    to_atom_batch_3_shape
])
def test_position_reshape(batch_positions, batch_atom_size, reshape_function):
    """Test to_batch_atom_3_shape and to_atom_batch_3_shape functions."""
    reshaped_positions = reshape_function(batch_positions)

    # We use batch_atom_size = None to flag cases in which we expect
    # the to_batch_atom_3_shape function to return the same object.
    if batch_atom_size is None:
        if reshape_function is to_batch_atom_3_shape:
            assert batch_positions is reshaped_positions
            return
        else:
            assert reshape_function is to_atom_batch_3_shape
            expected_shape = (batch_positions.shape[1], batch_positions.shape[0], 3)
    else:
        if reshape_function is to_batch_atom_3_shape:
            expected_shape = (batch_atom_size[0], batch_atom_size[1], 3)
        else:
            assert reshape_function is to_atom_batch_3_shape
            expected_shape = (batch_atom_size[1], batch_atom_size[0], 3)
    assert reshaped_positions.shape == expected_shape

    # The reshaped positions should have the same type as the input.
    assert type(batch_positions) == type(batch_positions)

    # The reshaped positions should be a view.
    if isinstance(batch_positions, torch.Tensor):
        assert batch_positions.data_ptr() == reshaped_positions.data_ptr()
    elif isinstance(batch_positions, pint.Quantity):
        assert reshaped_positions.magnitude.base is batch_positions.magnitude
    else:
        # Numpy array.
        assert reshaped_positions.base is batch_positions


def test_batchwise_atom_dist():
    """Test the batchwise_atom_dist function."""
    random_state = np.random.RandomState(0)
    batch_size = 5
    pos_atom_a = random_state.randn(batch_size, 3)
    pos_atom_b = random_state.randn(batch_size, 3)

    # Make sure we can compute the gradient.
    torch_pos_atom_a = torch.tensor(pos_atom_a, requires_grad=True, dtype=torch.double)
    torch_pos_atom_b = torch.tensor(pos_atom_b, requires_grad=True, dtype=torch.double)

    # Check gradient.
    assert torch.autograd.gradcheck(
        func=batchwise_atom_dist,
        inputs=[torch_pos_atom_a, torch_pos_atom_b]
    )

    # Compute a reference with numpy.
    dist_numpy = np.linalg.norm(pos_atom_a - pos_atom_b, axis=1)
    dist_torch = batchwise_atom_dist(torch_pos_atom_a, torch_pos_atom_b)
    assert np.allclose(dist_numpy, dist_torch.detach().numpy())
