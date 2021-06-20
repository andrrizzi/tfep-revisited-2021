#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in functions.linear.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch
import torch.autograd

from ..functions.linear import masked_linear


# =============================================================================
# TESTS
# =============================================================================

def test_masked_linear_gradcheck():
    """Run autograd.gradcheck on the masked_linear function."""
    batch_size = 2
    in_features = 3
    out_features = 5

    # Normal linear arguments.
    input = torch.randn(batch_size, in_features, dtype=torch.double, requires_grad=True)
    weight = torch.randn(out_features, in_features, dtype=torch.double, requires_grad=True)
    bias = torch.randn(out_features, dtype=torch.double, requires_grad=True)

    # Lower triangular mask.
    mask = torch.tril(torch.ones(out_features, in_features, dtype=torch.double, requires_grad=False))

    # With a None mask, the module should fall back to the native implementation.
    for m in [mask, None]:
        result = torch.autograd.gradcheck(
            func=masked_linear,
            inputs=[input, weight, bias, m]
        )
        assert result
