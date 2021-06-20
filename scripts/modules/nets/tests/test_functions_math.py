#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in functions.math.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch

from ..functions.math import cov


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('ddof', [0, 1])
@pytest.mark.parametrize('dim_n', [0, 1])
def test_cov(ddof, dim_n):
    """Test the covariance matrix against the numpy implementation."""
    random_state = np.random.RandomState(0)
    x = random_state.randn(10, 15)

    if dim_n == 0:
        cov_np = np.cov(x.T, ddof=ddof)
    else:
        cov_np = np.cov(x, ddof=ddof)

    cov_torch = cov(torch.tensor(x), dim_n=dim_n, ddof=ddof, inplace=True).numpy()

    assert np.allclose(cov_np, cov_torch)
