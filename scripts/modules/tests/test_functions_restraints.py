#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module functions.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch

from ..functions.restraints import upper_walls_plumed, lower_walls_plumed


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _ref_potential_walls_plumed(cv, at, kappa, offset=0.0, exp=2.0, eps=1.0, upper_wall=True):
    """A reference implementation of PLUMED UPPER/LOWER_WALLS restraint for testing."""
    if upper_wall and cv <= at - offset:
        return 0.0
    elif not upper_wall and cv >= at - offset:
        return 0.0
    dist = cv - at + offset
    if not upper_wall:
        dist = -dist
    return kappa * (dist / eps)**exp


@pytest.mark.parametrize('wall_func', [upper_walls_plumed, lower_walls_plumed])
@pytest.mark.parametrize('at,kappa,offset,exp,eps', [
    (0.5, 100,   0, 2, 1),
    (1.5, 150, 0.5, 3, 1),
    (  2,  10,   0, 1, 2),
])
def test_walls_plumed(wall_func, at, kappa, offset, exp, eps):
    cvs = np.linspace(0, 5, num=25)

    # Build a profile along the reference and PyTorch implementation.
    potential = np.empty(cvs.shape)
    upper_wall = True if wall_func is upper_walls_plumed else False
    for i, cv in enumerate(cvs):
        potential[i] = _ref_potential_walls_plumed(cv, at, kappa, offset, exp, eps, upper_wall)

    cvs_torch = torch.tensor(cvs, dtype=torch.double, requires_grad=True)
    potential_torch = wall_func(cvs_torch, at, kappa, offset, exp, eps)
    potential_torch_arr = potential_torch.detach().numpy()
    assert np.allclose(potential, potential_torch_arr)

    # Check that it's zero in the right portion of the cv.
    cutoff_idx = np.argmax(cvs > at - offset)
    if upper_wall:
        assert np.all(potential_torch_arr[:cutoff_idx] == 0.0)
    else:
        assert np.all(potential_torch_arr[cutoff_idx:] == 0.0)

    # Check gradient with autograd
    # Check gradient.
    assert torch.autograd.gradcheck(
        func=wall_func,
        inputs=[cvs_torch, at, kappa, offset, exp, eps]
    )
