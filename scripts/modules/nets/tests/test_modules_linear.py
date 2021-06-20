#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test layers in modules.linear.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from ..modules.linear import MaskedLinear, masked_weight_norm


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_weight_vectors(w):
    w = w.detach()
    vector_norms = torch.tensor([torch.norm(x) for x in w])
    for v_idx, v_norm in enumerate(vector_norms):
        if v_norm != 0:
            w[v_idx] /= v_norm
    return w


def check_wnorm_components(layer, mask):
    masked_weights = layer.weight.detach() * mask.detach()
    expected_g = torch.tensor([[torch.norm(x)] for x in masked_weights])
    expected_normalized_v = normalize_weight_vectors(masked_weights)

    # Compute the normalized v.
    normalized_weight_v = normalize_weight_vectors(layer.weight_v)

    assert torch.allclose(layer.weight_g, expected_g)
    assert torch.allclose(normalized_weight_v, expected_normalized_v)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('diagonal', [0, -1, -2])
@pytest.mark.parametrize('wnorm', [False, True])
def test_masked_linear_wnorm_compatibility(diagonal, wnorm):
    """Check that training of the masked linear layer is compatible with weight normalization."""
    batch_size = 2
    in_features = 4
    out_features = 5

    # Generate random input. Make sure the test is reproducible.
    generator = torch.Generator()
    generator.manual_seed(0)
    x = torch.randn(batch_size, in_features, generator=generator, requires_grad=True)

    # Lower triangular mask.
    mask = torch.tril(torch.ones(out_features, in_features, requires_grad=False),
                      diagonal=diagonal)

    # Create a weight-normalized masked linear layer.
    masked_linear = MaskedLinear(in_features, out_features, bias=True, mask=mask)
    if wnorm:
        masked_linear = masked_weight_norm(masked_linear, name='weight')

        # The norm and direction vectors are also masked.
        check_wnorm_components(masked_linear, mask)

    # The gradient of the masked parameters should be zero.
    y = masked_linear(x)
    loss = torch.sum(y)
    loss.backward()

    if wnorm:
        assert (masked_linear.weight_g.grad[:abs(diagonal)] == 0).detach().byte().all()
        assert (masked_linear.weight_v.grad * (1 - mask) == 0).detach().byte().all()
    else:
        assert (masked_linear.weight.grad * (1 - mask) == 0).detach().byte().all()

    if wnorm:
        # Simulate one batch update.
        sgd = torch.optim.SGD(masked_linear.parameters(), lr=0.01, momentum=0.9)
        sgd.step()

        # Make a forward pass so that the wnorm wrapper will update masked_linear.weight
        masked_linear(x)

        # Check that g and v are still those expected.
        check_wnorm_components(masked_linear, mask)
