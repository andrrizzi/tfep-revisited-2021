#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility functions for algebra in PyTorch.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# FUNCTIONS
# =============================================================================

def batchwise_dot(x1, x2):
    """Batchwise dot product between two 2D tensors.

    Takes two tensors of shape ``(batch_size, N)`` and returns the batchwise
    dot product of shape ``(batch_size, 1)``.

    """
    return (x1 * x2).sum(dim=-1, keepdim=True)


def batchwise_outer(x1, x2):
    """Batchwise outer product between two 2D tensors.

    Takes two tensors of shape ``(batch_size, N)`` and returns the outer
    product of shape ``(batch_size, N, N)``.

    """
    return torch.matmul(x1[:, :, None], x2[:, None, :])


def cov(x, ddof=1, dim_n=1, inplace=False):
    """Return the covariance matrix of the data.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape ``(m, n)``, where ``n`` is the number of samples
        used to estimate the covariance, and ``m`` is the dimension of
        the multivariate variable. If ``dim_n`` is 0, then the expected
        shape is ``(n, m)``.
    ddof : int, optional
        The number of dependent degrees of freedom. The covariance will
        be estimated dividing by ``n - ddof``. Default is 1.
    dim_n : int, optional
        The dimension used to collect the samples. Default is 1.
    inplace : bool, optional
        If ``True``, the input argument ``x`` is modified to be centered
        on its mean. Default is ``False``.

    Returns
    -------
    cov : torch.Tensor
        A tensor of shape ``(m, m)``.

    """
    if len(x.shape) != 2:
        raise ValueError('The function supports only 2D matrices')
    if dim_n not in {0, 1}:
        raise ValueError('dim_n must be either 0 or 1')

    # Center the data on the mean.
    if dim_n == 1:
        keepdim = True
    else:
        keepdim = False
    mean = torch.mean(x, dim_n, keepdim=keepdim)
    if inplace:
        x -= mean
    else:
        x = x - mean

    # Average normalization factor.
    n = x.shape[dim_n] - ddof

    # Compute the covariance matrix
    if dim_n == 0:
        c = x.t().matmul(x) / n
    else:
        c = x.matmul(x.t()) / n

    return c
