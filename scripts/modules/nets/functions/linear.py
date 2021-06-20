#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Function objects for PyTorch.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch.autograd
import torch.nn.functional

# =============================================================================
# Functions
# =============================================================================

class MaskedLinearFunction(torch.autograd.Function):
    r"""Implement the masked linear transformation: :math:`y = x \cdot (M \circ A)^T + b`.

    This is based on :func:`torch.nn.functional.linear`, but with an extra
    keyword argument ``mask`` having the same shape as ``weight``.

    Note that the function does not perform a sparse multiplication, but
    simply implements the mask with an element-wise multiplication of the
    weight matrix before evaluating the linear transformation.

    A functional shortcut to ``MaskedLinearFunction`` is available in
    this same module with ``masked_linear``.

    The return value is a ``Tensor`` of shape ``(batch_size, *, n_out_features)``,
    where ``*`` correspond to the same number of additional dimensions
    in the `input` argument.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor x of shape ``(batch_size, *, n_in_features)``, where
        ``*`` means any number of additional dimensions.
    weight : torch.Tensor
        Scaling tensor A of shape ``(n_out_features, n_in_features)``.
    bias : torch.Tensor, optional
        Shifting tensor b of shape ``(n_out_features)``.
    mask : torch.Tensor, optional
        Mask of A of shape ``(n_out_features, n_in_features)``.

    Examples
    --------

    >>> batch_size = 2
    >>> in_features = 3
    >>> out_features = 5
    >>> input = torch.randn(batch_size, in_features, dtype=torch.double)
    >>> weight = torch.randn(out_features, in_features, dtype=torch.double)
    >>> bias = torch.randn(out_features, dtype=torch.double)
    >>> # Lower triangular mask.
    >>> mask = torch.tril(torch.ones(out_features, in_features, dtype=torch.bool))
    >>> output = masked_linear(input, weight, bias, mask)

    """

    @staticmethod
    def forward(ctx, input, weight, bias=None, mask=None):
        # Check if we need to mask the weights.
        if mask is not None:
            # Mask weight matrix.
            weight = weight * mask

        # We save the MASKED weights for backward propagation so that
        # we don't need to perform the element-wise multiplication.
        ctx.save_for_backward(input, weight, bias, mask)

        # Compute the linear transformation.
        return torch.nn.functional.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # Unpack previously stored tensors.
        input, masked_weight, bias, mask = ctx.saved_tensors

        # We still need to return None for grad_mask even if we don't
        # compute its gradient.
        grad_input = grad_weight = grad_bias = grad_mask = None

        # Compute gradients w.r.t. input parameters.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(masked_weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

            # Mask the gradients.
            if mask is not None:
                grad_weight.mul_(mask)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, grad_mask

# Functional notation.
masked_linear = MaskedLinearFunction.apply
