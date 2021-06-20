#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Function objects implementing normalizing flow transformers for PyTorch.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch
import torch.autograd
import torch.nn.functional

from .math import batchwise_dot, batchwise_outer
from ..utils import generate_block_sizes


# =============================================================================
# AFFINE
# =============================================================================

def affine_transformer(x, shift, log_scale, gate=None):
    r"""Implement a gated affine transformer for triangular maps.

    This is an implementation of the transformer

    :math:`y_i = gamma * x_i + (1 - gamma) * exp(a_i) * x_i + b_i`

    The function returns the transformed feature as a ``Tensor`` of shape
    ``(batch_size, n_features)`` and the log absolute determinant of its
    Jacobian as a ``Tensor`` of shape ``(batch_size,)``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor x of shape ``(batch_size, n_features)``.
    shift : torch.Tensor
        The shift coefficients of shape ``(batch_size, n_features)``
        (i.e. the ``b`` coefficients).
    log_scale : torch.Tensor
        The logarithm of the scale coefficients of shape ``(batch_size, n_features)``
        (i.e. the ``a`` coefficients).
    gate : torch.Tensor, optional
        The gamma parameter, which is a tensor of shape ``(batch_size,)``
        between 0.0 and 1.0. If 0.0, the transformer is forced to the
        identity function. If 1.0, it is a standard affine transformer.
        Default is 1.0.

    """
    scale = torch.exp(log_scale)

    if gate is None:
        y =  x * scale + shift
        log_det_J = torch.sum(log_scale, dim=1)
    else:
        # Add dimension for broadcasting.
        gate = gate[:, None]
        scale = 1 - gate + gate*scale
        y = x * scale + gate * shift
        log_det_J = torch.sum(torch.log(scale), dim=1)

    return y, log_det_J


def affine_transformer_inv(y, shift, log_scale, gate=None):
    r"""Inverse function of ``gated_affine_transformer``.

    This is the inverse of the transformer

    :math:`y_i = gamma * x_i + (1 - gamma) * exp(a_i) * x_i + b_i`

    The function returns the transformed feature as a ``Tensor`` of shape
    ``(batch_size, n_features)`` and the log absolute determinant of its
    Jacobian as a ``Tensor`` of shape ``(batch_size,)``.

    Parameters
    ----------
    y : torch.Tensor
        Input tensor x of shape ``(batch_size, n_features)``.
    shift : torch.Tensor
        The shift coefficients of shape ``(batch_size, n_features)``
        (i.e. the ``b`` coefficients).
    log_scale : torch.Tensor
        The logarithm of the scale coefficients of shape ``(batch_size, n_features)``
        (i.e. the ``a`` coefficients).
    gate : torch.Tensor, optional
        The gamma parameter, which is a tensor of shape ``(batch_size,)``
        between 0.0 and 1.0. If 0.0, the transformer is forced to the
        identity function. If 1.0, it is a standard affine transformer.
        Default is 1.0.

    """
    if gate is None:
        x = (y - shift) * torch.exp(-log_scale)
        log_det_J = -torch.sum(log_scale, dim=1)
    else:
        # Add dimension for broadcasting.
        gate = gate[:, None]
        scale = 1 - gate + gate*torch.exp(log_scale)
        x = (y - shift*gate) / scale
        log_det_J = -torch.sum(torch.log(scale), dim=1)
    return x, log_det_J


# =============================================================================
# SUM-OF-SQUARES POLYNOMIAL TRANSFORMER
# =============================================================================

class SOSPolynomialTransformer(torch.autograd.Function):
    r"""Implement the sum-of-squares polynomial transformer for triangular maps.

    This is an implementation of the polynomial transformer proposed in [1].

    :math:`y_i = a_0 + \int_0^{x_i} \sum_{k=1}^K \left( \sum_{l=0}^L a_{kl} z^l \right)^2 dz`

    With the addition of a gating parameter gamma so that the final output is

    :math:`y'_i = \gamma * y_i = (1 - \gamma) * x_i`

    The function returns the transformed feature as a ``Tensor`` of shape
    ``(batch_size, n_features)`` and the log absolute determinant of its
    Jacobian as a ``Tensor`` of shape ``(batch_size,)``.

    Only sums of squared first-degree polynomials (i.e., L=1) are currently
    supported as they are the only one with an analytic inverse and sum of
    zeroth degree polynomials (i.e., L=0) are equivalent to affine transformer.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor x of shape ``(batch_size, n_features)``.
    coefficients : torch.Tensor
        The coefficients of the squared polynomials obtained from the
        conditioner. Each ``Tensor`` has shape ``(batch_size, 1+K*L, n_features)``.
        The coefficients are ordered by polynomial so that ``coefficients[:,0]``
        is :math:`a_0` followed by :math:`a_{10}, a_{11}, ..., a_{K0}, a_{K1}`.
    gate : torch.Tensor, optional
        The gamma parameter, which is a tensor of shape ``(batch_size,)``
        between 0.0 and 1.0. If 0.0, the transformer is forced to the
        identity function. If 1.0, it is a standard affine transformer.
        Default is 1.0.

    References
    ----------
    [1] Jaini P, Selby KA, Yu Y. Sum-of-Squares Polynomial Flow. arXiv
        preprint arXiv:1905.02325. 2019 May 7.

    """

    @staticmethod
    def forward(ctx, x, coefficients, gate=None):
        # Compute the parameters of the sos polynomial.
        sos_degree_coefficients = SOSPolynomialTransformer.get_sos_poly_coefficients(coefficients)

        # Compute the power of x.
        x_powers = [x, x*x]

        # Compute y and the gradient of y w.r.t. x.
        y = sos_degree_coefficients[1].clone()
        grad_x = sos_degree_coefficients[1].clone()

        for degree, coef in enumerate(sos_degree_coefficients[2:]):
            term = coef * x_powers[degree]
            y += term
            grad_x += (degree+2) * term

        y *= x
        y += sos_degree_coefficients[0]

        # Add gating.
        if gate is not None:
            # We already compute the gradient of gate since we have
            # already done most of the computational effort.
            grad_gate = y - x

            # Add fake dimension to enable broadcasting. We still need the
            # passed gate to save it for backward.
            gate_unsqueezed = gate.unsqueeze(1)
            one_m_gate = 1 - gate_unsqueezed
            y = gate_unsqueezed * y + one_m_gate * x
            grad_x = gate_unsqueezed * grad_x + one_m_gate

        log_det_J = torch.sum(torch.log(grad_x), dim=1)

        # Save tensor used for backward() before returning.
        if gate is None:
            ctx.save_for_backward(grad_x, coefficients, *x_powers)
        else:
            ctx.save_for_backward(grad_x, coefficients, gate, grad_gate, *x_powers)

        # We don't need to compute gradients of log_det_J.
        ctx.mark_non_differentiable(log_det_J)
        return y, log_det_J

    @staticmethod
    def backward(ctx, grad_y, grad_log_det_J):
        try:
            saved_grad_x, coefficients, gate, saved_grad_gate, x, x2 = ctx.saved_tensors
            gate = gate.unsqueeze(1)
        except ValueError:
            saved_grad_x, coefficients, x, x2 = ctx.saved_tensors
            gate = None
        grad_x = grad_coefficients = grad_gate = None
        batch_size, n_features = saved_grad_x.shape

        # Compute gradients w.r.t. input parameters.
        if ctx.needs_input_grad[0]:
            grad_x = saved_grad_x * grad_y

        if ctx.needs_input_grad[1]:
            grad_coefficients = torch.empty_like(coefficients)

            # The first coefficient is the constant term.
            grad_coefficients[:, 0] = torch.ones(
                size=(batch_size, n_features), dtype=saved_grad_x.dtype)

            # Zeroth and first degree terms of the inner polynomials.
            zeroth_degree_terms = coefficients[:, 1::2]
            first_degree_terms = coefficients[:, 2::2]

            # We need to add a dimension corresponding to the number of
            # coefficients in the power of x for them to be broadcastable.
            x = x.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            x3 = x2 * x

            grad_coefficients[:, 1::2] = first_degree_terms*x2 + 2*zeroth_degree_terms*x
            grad_coefficients[:, 2::2] = 2/3*first_degree_terms*x3 + zeroth_degree_terms*x2

            if gate is not None:
                grad_coefficients *= gate.unsqueeze(2)

            grad_coefficients = grad_coefficients * grad_y.unsqueeze(1)

        if ctx.needs_input_grad[2]:
            # batchwise_dot returns shape (batch_size, 1)
            grad_gate = batchwise_dot(grad_y, saved_grad_gate).squeeze(1)

        return grad_x, grad_coefficients, grad_gate

    @staticmethod
    def get_sos_poly_coefficients(coefficients):
        """Compute the coefficient of the SOS polynomial.

        Parameters
        ----------
        coefficients : torch.Tensor
            The coefficients of the squared polynomials obtained from the
            conditioner. Each ``Tensor`` has shape ``(batch_size, 1+K*L, n_features)``.
            The coefficients are ordered by polynomial so that ``coefficients[:,0]``
            is :math:`a_0` followed by :math:`a_{10}, a_{11}, ..., a_{K0}, a_{K1}`.

        Returns
        -------
        sos_poly_coefficients : List[torch.Tensor]
            ``sos_poly_coefficients[i]`` is a tensor of shape ``(batch_size, n_features)``
            with the coefficients of the term of the SOS polynomial of degree ``i``.

        """
        # We support only L=1 for now. Number of coefficients in
        # each summed polynomials include also the constant term.
        coeff_per_inner_poly = 2
        batch_size, _, n_features = coefficients.shape

        # inner_degree_parameters[d][b][p] is the parameter for the term of
        # the p-th inner polynomial of degree d for the b-th batch sample.
        inner_degree_coefficients = []
        for degree in range(coeff_per_inner_poly):
            inner_degree_coefficients.append(coefficients[:, 1+degree::coeff_per_inner_poly])

        # Find the coefficients of the integrated polynomial.
        sos_degree_coefficients = [coefficients[:, 0]]
        sos_degree_coefficients.append(torch.sum(inner_degree_coefficients[0]**2, dim=1))
        sos_degree_coefficients.append(torch.sum(inner_degree_coefficients[0]*inner_degree_coefficients[1], dim=1))
        sos_degree_coefficients.append(torch.sum(inner_degree_coefficients[1]**2, dim=1) / 3)

        return sos_degree_coefficients


# Functional notation.
sos_polynomial_transformer = SOSPolynomialTransformer.apply


# =============================================================================
# NEURAL SPLINE
# =============================================================================

def neural_spline_transformer(x, x0, y0, widths, heights, slopes):
    r"""Implement the circular spline transformer.

    This is an implementation of the neural spline transformer proposed
    in [1]. Using the therminology in [1], the spline function is defined
    from K+1 knots (x, y) that give rise to K bins. Currently, this
    implementation doesn't support gating.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor x of shape ``(batch_size, n_features)``. Currently, this
        must hold: ``x0[i] <= x[i] <= x0[i] + cumsum(widths)`` for all ``i``.
    x0 : torch.Tensor
        The input dimension for the first of the K+1 knots determining the
        positions of the K bins as a tensor of shape ``(n_features,)``. Inputs
        that are equal or below this (in any dimension) are mapped to itself.
    y0 : torch.Tensor
        The output dimension for the first of the K+1 knots determining the
        positions of the K bins as a tensor of shape ``(n_features,)``.
    widths : torch.Tensor
        ``widths[b, k, i]`` is the width of the k-th bin between the k-th and (k+1)-th
         knot for the i-th feature and b-th batch. The tensor has shape
         ``(batch_size, K, n_features)``.
    heights : torch.Tensor
        ``heights[b, k, i]`` is the height of the k-th bin between the k-th and (k+1)-th
         knot for the i-th feature and b-th batch. The tensor has shape
         ``(batch_size, K, n_features)``.
    slopes : torch.Tensor
        ``slopes[b, k, i]`` is the slope at the (k+1)-th knot (the slope of the
        first and last knots are always 1. The tensor has shape
        ``(batch_size, K-1, n_features)``.

    References
    ----------
    [1] Durkan C, Bekasov A, Murray I, Papamakarios G. Neural spline flows.
        arXiv preprint arXiv:1906.04032. 2019 Jun 10.

    """
    dtype = x0.dtype
    batch_size, n_bins, n_features = widths.shape
    n_knots = n_bins + 1

    # knots_x has shape (n_features, K+1).
    knots_x = torch.empty(batch_size, n_knots, n_features, dtype=dtype)
    knots_x[:, 0] = x0
    knots_x[:, 1:] = x0 + torch.cumsum(widths, dim=1)
    knots_y = torch.empty(batch_size, n_knots, n_features, dtype=dtype)
    knots_y[:, 0] = y0
    knots_y[:, 1:] = y0 + torch.cumsum(heights, dim=1)

    # The 0-th and last knots have always slope 1 to avoid discontinuities.
    # After this, slopes has shape (batch_size, n_features, K+1).
    ones = torch.ones(batch_size, 1, n_features)
    slopes = torch.cat((ones, slopes, ones), dim=1)

    # For an idea about how the indexing is working in this function, see
    # https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor.
    batch_indices = torch.arange(batch_size).unsqueeze(-1)  # Shape: (batch_size, 1).
    feat_indices = torch.arange(n_features).repeat(batch_size, 1)  # Shape: (batch_size, n_features).

    # bin_indices[i][j] is the index of the bin assigned to x[i][j].
    bin_indices = torch.sum((x.unsqueeze(1) > knots_x), dim=1) - 1

    # All the following arrays have shape (batch_size, n_features).
    # widths_b_f[i][j] is the width of the bin assigned to x[i][j].
    widths_b_f = widths[batch_indices, bin_indices, feat_indices]
    heights_b_f = heights[batch_indices, bin_indices, feat_indices]

    # lower_knot_x_b_f[i][j] is the lower bound of the bin assigned to x[i][j].
    lower_knot_x_b_f = knots_x[batch_indices, bin_indices, feat_indices]
    lower_knot_y_b_f = knots_y[batch_indices, bin_indices, feat_indices]

    # slopes_k_b_f[i][j] is the slope of the lower-bound knot of the bin assigned to x[i][j].
    # slopes_k1_b_f[i][j] is the slope of the upper-bound knot of the bin assigned to x[i][j].
    slopes_k_b_f = slopes[batch_indices, bin_indices, feat_indices]
    slopes_k1_b_f = slopes[batch_indices, bin_indices+1, feat_indices]

    # This is s_k = (y^k+1 - y^k)/(x^k+1 - x^k) and epsilon in the
    # paper, both with shape (batch_size, n_features).
    s_b_f = heights_b_f / widths_b_f
    epsilon_b_f = (x - lower_knot_x_b_f) / widths_b_f

    # epsilon * (1 - epsilon)
    epsilon_1mepsilon_b_f = epsilon_b_f * (1 - epsilon_b_f)
    epsilon2_b_f = epsilon_b_f**2

    # Compute the output.
    numerator = heights_b_f * (s_b_f * epsilon2_b_f + slopes_k_b_f * epsilon_1mepsilon_b_f)
    denominator = s_b_f + (slopes_k1_b_f + slopes_k_b_f - 2*s_b_f) * epsilon_1mepsilon_b_f
    y = lower_knot_y_b_f + numerator/denominator

    # Compute the derivative
    numerator = s_b_f**2 * (slopes_k1_b_f*epsilon2_b_f + 2*s_b_f*epsilon_1mepsilon_b_f + slopes_k_b_f*(1 - epsilon_b_f)**2)
    denominator = (s_b_f + (slopes_k1_b_f + slopes_k_b_f - 2 * s_b_f) * epsilon_1mepsilon_b_f)**2
    dy_dx = numerator / denominator

    # Compute the log det J.
    log_det_J = torch.sum(torch.log(dy_dx), dim=1)

    return y, log_det_J


# =============================================================================
# MOBIUS TRANSFORMER
# =============================================================================

def unit_cube_to_inscribed_sphere(w, blocks, shorten_last_block=False):
    r"""Utility function mapping vectors from the cube to its inscribed sphere.

    The mapping is supported only for dimensions of blocks up to three.

    Parameters
    ----------
    w : torch.Tensor
        The vectors within the hypercubes to be mapped to the hyperspheres.
        This has shape ``(batch_size, n_features)``.
    blocks : int or List[int]
        The size of the blocks. If an integer, ``w`` is divided into
        blocks of equal size. Otherwise, it is divided into ``len(blocks)``
        blocks, with the i-th block having size ``blocks[i]``.
    shorten_last_block : bool, optional
        If ``True`` and ``blocks`` is an integer that is not a divisor of
        the number of features, the last block is shortened automatically.
        Otherwise, an exception is raised if ``blocks`` is an integer
        that does not divide the number of features.

    Returns
    -------
    mapped_w : torch.Tensor
        The mapped vectors of the same shape of ``w``.

    """
    batch_size, n_features = w.shape

    # Eventually convert a constant block size to a list of block sizes.
    blocks = generate_block_sizes(n_features, blocks, shorten_last_block)

    # Initialized the returned value.
    mapped_w = torch.empty_like(w)

    # The pointer to the index where the current block starts.
    block_pointer = 0

    for block_size in blocks:
        w_block = w[:, block_pointer:block_pointer+block_size]

        if block_size == 3:
            squared = w_block**2
            squared_norms = squared.sum(dim=1, keepdim=True)
            yxx = torch.index_select(squared, dim=1, index=torch.tensor([1, 0, 0]))
            zzy = torch.index_select(squared, dim=1, index=torch.tensor([2, 2, 1]))
            mapped_w_block = w_block * torch.sqrt(1 - (squared_norms - squared) / 2 + yxx * zzy / 3)
        elif block_size == 2:
            swapped = torch.index_select(w_block, dim=1, index=torch.tensor([1, 0]))
            mapped_w_block = w_block * torch.sqrt(1 - swapped**2/2)
        elif block_size == 1:
            mapped_w_block = w_block
        else:
            raise NotImplementedError('Hypercube to hypersphere mapping is not implemented')

        mapped_w[:, block_pointer:block_pointer+block_size] = mapped_w_block

        block_pointer += block_size

    return mapped_w


class MobiusTransformer(torch.autograd.Function):
    r"""Implement the Mobius transformation.

    This is a variant of the transformation proposed in [1], and used
    in [2] to create flows on tori and spheres manifolds. The difference
    with [1], is that we always project the input vector on the unit
    sphere before applying the transformation so that it always preserve
    the distance with the center (which in the current implementation
    is hardcoded to be the origin). The difference with [2] is that the
    transformation contracts points close to the parameter vector rather
    than expanding it.

    The transformer applies the Mobius transformation in "blocks". Blocks
    of size up to 3 are supported. The function returns the transformed
    feature as a ``Tensor`` of shape ``(batch_size, n_features)`` and
    the log absolute determinant of its Jacobian as a ``Tensor`` of shape
    ``(batch_size,)``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor x of shape ``(batch_size, n_features)``.
    w : torch.Tensor
        The vectors determining the point to be contracted/expanded.
        These vectors has shape ``batch_size, n_features)`` and they
        must be within the unit sphere for their block.
    blocks : int or List[int]
        The size of the blocks. If an integer, ``x`` and ``w`` are
        divided into blocks of equal size. Otherwise, it is divided
        into ``len(blocks)`` blocks, with the i-th block having size
        ``blocks[i]``.
    gate : torch.Tensor, optional
        The gamma parameter, which is a tensor of shape ``(batch_size,)``
        between 0.0 and 1.0. If 0.0, the transformer is forced to the
        identity function. If 1.0, it is a standard affine transformer.
        Default is 1.0.
    shorten_last_block : bool, optional
        If ``True`` and ``blocks`` is an integer that is not a divisor of
        the number of features, the last block is shortened automatically.
        Otherwise, an exception is raised if ``blocks`` is an integer
        that does not divide the number of features.

    References
    ----------
    [1] Kato S, McCullagh P. Mobius transformation and a Cauchy family
        on the sphere. arXiv preprint arXiv:1510.07679. 2015 Oct 26.
    [2] Rezende DJ, Papamakarios G, Racanière S, Albergo MS, Kanwar G,
        Shanahan PE, Cranmer K. Normalizing Flows on Tori and Spheres.
        arXiv preprint arXiv:2002.02428. 2020 Feb 6.

    """

    @staticmethod
    def forward(ctx, x, w, blocks, gate=None, shorten_last_block=False):
        batch_size, n_features = x.shape

        # Eventually convert a constant block size to a list of block sizes.
        blocks = generate_block_sizes(n_features, blocks, shorten_last_block)

        # If gate is passed, we'll work with gate * w, but we still need
        # to remember the original w for the gradient.
        w_passed = w
        if gate is not None:
            # Add a dimension for broadcasting.
            w = gate[:, None] * w

        # The input vectors normalized by their own norm.
        x_normalized = torch.empty_like(x)

        # x_normalized + w.
        x_normalized_plus_w = torch.empty_like(x)

        # Compute the norms of the x vectors for each block.
        # norm_squared has shape (batch_size, n_features).
        x_norm = torch.empty_like(w)
        w_norm_squared = torch.empty_like(w)
        xw_norm_squared = torch.empty_like(x)

        # The pointer to the index where the current block starts.
        block_pointer = 0

        for block_size in blocks:
            x_block = x[:, block_pointer:block_pointer+block_size]
            w_block = w[:, block_pointer:block_pointer+block_size]

            # Compute normalized input vector for this block.
            x_norm_block = torch.sqrt(torch.sum(x_block**2, dim=1, keepdim=True))
            x_normalized_block = x_block / x_norm_block
            x_norm[:, block_pointer:block_pointer+block_size] = x_norm_block
            x_normalized[:, block_pointer:block_pointer+block_size] = x_normalized_block

            x_normalized_plus_w_block = x_normalized_block + w_block
            x_normalized_plus_w[:, block_pointer:block_pointer+block_size] = x_normalized_plus_w_block

            # Compute norms.
            w_norm_squared_block = torch.sum(w_block**2, dim=1, keepdim=True)
            xw_norm_squared_block = torch.sum(x_normalized_plus_w_block**2, dim=1, keepdim=True)
            w_norm_squared[:, block_pointer:block_pointer+block_size] = w_norm_squared_block
            xw_norm_squared[:, block_pointer:block_pointer+block_size] = xw_norm_squared_block

            block_pointer += block_size

        # Compute the transformation.
        y_normalized = (1 - w_norm_squared) / xw_norm_squared * (x_normalized + w) + w
        y = x_norm * y_normalized

        # Compute the gradient for backprop and the Jacobian determinant.
        grad_x = torch.zeros(batch_size, n_features, n_features, dtype=x.dtype)
        log_det_J = torch.zeros(batch_size, dtype=x.dtype)

        block_pointer = 0
        for block_size in blocks:
            x_block = x[:, block_pointer:block_pointer+block_size]
            x_normalized_plus_w_block = x_normalized_plus_w[:, block_pointer:block_pointer+block_size]
            y_normalized_block = y_normalized[:, block_pointer:block_pointer+block_size]

            # Add two fake dimensions to the norms so that they can be broadcasted correctly.
            x_norm_block = x_norm[:, block_pointer, None, None]
            w_norm_squared_block = w_norm_squared[:, block_pointer, None, None]
            xw_norm_squared_block = xw_norm_squared[:, block_pointer, None, None]

            # d||x||/dx_j = x_j / ||x||
            dxnorm_dx = x_normalized[:, block_pointer:block_pointer+block_size]

            # Compute dx_normalized/dx = I/x_norm - (x_block X x_block)/x_norm**3
            # where "I" is the identity matrix and "X" is the outer product.
            # dxnormalized_dx[i][j] = dx_normalized_i/dx_j.
            x_block_outer_x_block = batchwise_outer(x_block, x_block)
            batch_eye = torch.diag_embed(torch.ones(batch_size, block_size, dtype=x.dtype))
            dxnormalized_dx = (batch_eye - x_block_outer_x_block / x_norm_block**2) / x_norm_block

            # Compute the block Jacobian.
            grad_x_block = torch.matmul(dxnormalized_dx, x_normalized_plus_w_block[:, :, None])[:, :, 0]
            grad_x_block = 2 / xw_norm_squared_block * batchwise_outer(x_normalized_plus_w_block, grad_x_block)
            grad_x_block = x_norm_block * (1 - w_norm_squared_block) / xw_norm_squared_block * (dxnormalized_dx - grad_x_block)
            grad_x_block += batchwise_outer(y_normalized_block, dxnorm_dx)
            grad_x[:, block_pointer:block_pointer+block_size, block_pointer:block_pointer+block_size] = grad_x_block

            # Compute the determinant.
            log_det_J += torch.log(torch.abs(_det(grad_x_block)))

            block_pointer += block_size

        # Save tensors used for backward() before returning.
        if gate is None:
            ctx.save_for_backward(grad_x, w_passed, x_normalized_plus_w, x_norm, w_norm_squared, xw_norm_squared)
        else:
            ctx.save_for_backward(grad_x, w_passed, x_normalized_plus_w, x_norm, w_norm_squared, xw_norm_squared, gate)
        ctx.blocks = blocks

        # We don't need to compute gradients of log_det_J.
        ctx.mark_non_differentiable(log_det_J)
        return y, log_det_J

    @staticmethod
    def backward(ctx, grad_y, grad_log_det_J):
        grad_x = grad_w = grad_blocks = grad_gate = grad_shorten_last_block = None

        # Read the saved tensors.
        try:
            # gate was passed.
            saved_grad_x, w_passed, x_normalized_plus_w, x_norm, w_norm_squared, xw_norm_squared, gate = ctx.saved_tensors
            # Add a dimension for broadcasting.
            gate = gate[:, None]
            w = gate * w_passed
        except ValueError:
            # gate is None.
            saved_grad_x, w_passed, x_normalized_plus_w, x_norm, w_norm_squared, xw_norm_squared = ctx.saved_tensors
            gate = None
            w = w_passed
        batch_size, n_features = w.shape

        # Check which gradients we need.
        compute_x = ctx.needs_input_grad[0]
        compute_w = ctx.needs_input_grad[1]
        try:
            compute_gate = ctx.needs_input_grad[3]
        except:
            compute_gate = False

        # Compute gradients w.r.t. input parameters.
        if compute_x:
            grad_x = grad_y[:, None, :].matmul(saved_grad_x)[:, 0, :]

        # Initialize gradient tensors.
        if compute_w:
            # grad_w is block-diagonal so most of the entries are zero.
            grad_w = torch.zeros_like(saved_grad_x)

        if compute_gate:
            grad_gate = torch.empty_like(grad_y)

        # The gradient with respect to w and gate is very similar so we compute it together.
        if compute_w or compute_gate:

            # Compute the gradient for each block.
            block_pointer = 0
            for block_size in ctx.blocks:
                if compute_w:
                    w_block = w[:, block_pointer:block_pointer+block_size]
                    batch_eye = torch.diag_embed(torch.ones(batch_size, block_size, dtype=w.dtype))
                if compute_gate:
                    w_passed_block = w_passed[:, block_pointer:block_pointer+block_size]
                    w_passed_norm_squared_block = torch.sum(w_passed_block**2, dim=1, keepdim=True)

                x_normalized_plus_w_block = x_normalized_plus_w[:, block_pointer:block_pointer+block_size]
                x_norm_block = x_norm[:, block_pointer]
                w_norm_squared_block = w_norm_squared[:, block_pointer]
                xw_norm_squared_block = xw_norm_squared[:, block_pointer]

                # Compute common terms between the two.
                factor1 = 1 - w_norm_squared_block + xw_norm_squared_block
                factor2 = 2 * (1 - w_norm_squared_block) / xw_norm_squared_block
                factor3 = x_norm_block / xw_norm_squared_block

                # Compute the gradients.
                if compute_w:
                    grad_w_block = -2 * batchwise_outer(x_normalized_plus_w_block, w_block)
                    grad_w_block += factor1[:, None, None] * batch_eye
                    grad_w_block -= factor2[:, None, None] * batchwise_outer(x_normalized_plus_w_block, x_normalized_plus_w_block)
                    grad_w_block *= factor3[:, None, None]

                    grad_w[:, block_pointer:block_pointer+block_size, block_pointer:block_pointer+block_size] = grad_w_block

                if compute_gate:
                    grad_gate_block = -2 * gate * w_passed_norm_squared_block * x_normalized_plus_w_block
                    grad_gate_block += factor1[:, None] * w_passed_block
                    grad_gate_block -= factor2[:, None] * batchwise_dot(x_normalized_plus_w_block, w_passed_block) * x_normalized_plus_w_block
                    grad_gate_block *= factor3[:, None]

                    grad_gate[:, block_pointer:block_pointer+block_size] = grad_gate_block

                # Next block.
                block_pointer += block_size

        # Propagate to output gradient.
        if compute_w:
            if gate is not None:
                grad_w *= gate[:, :, None]

            # Batchwise matrix-vector product.
            grad_w = grad_y[:, None, :].matmul(grad_w)[:, 0, :]

        if compute_gate:
            # batchwise_dot returns shape (batch_size, 1)
            grad_gate = batchwise_dot(grad_y, grad_gate).squeeze(dim=1)

        return grad_x, grad_w, grad_blocks, grad_gate, grad_shorten_last_block

# Functional notation.
mobius_transformer = MobiusTransformer.apply


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _det(a):
    """
    Batch determinant.
    """
    if a.shape[1:] == (3, 3):
        return a[:, 0, 0] * _det(a[:, 1:, 1:]) - a[:, 0, 1] * _det(a[:, 1:, 0::2]) + a[:, 0, 2] * _det(a[:, 1:, :2])
    elif a.shape[1:] == (2, 2):
        return a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    elif a.shape[1:] == (1, 1):
        return a[:, 0, 0]
    return torch.det(a)

