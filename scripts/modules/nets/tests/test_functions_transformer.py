#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in functions.transformer.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import torch
import torch.autograd

from ..functions.transformer import (
    affine_transformer, affine_transformer_inv,
    sos_polynomial_transformer, neural_spline_transformer,
    mobius_transformer, unit_cube_to_inscribed_sphere
)
from ..utils import generate_block_sizes


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_random_input(batch_size, n_features, gate, n_parameters,
                        seed=0, x_func=torch.randn, par_func=torch.randn):
    """Create input, parameters and gates.

    Parameters
    ----------
    gate : bool
        If False, the returned gate will be None.

    """
    # Make sure the input arguments are deterministic.
    generator = torch.Generator()
    generator.manual_seed(seed)

    x = x_func(batch_size, n_features, generator=generator,
               dtype=torch.double, requires_grad=True)

    parameters = par_func(batch_size, n_parameters, n_features, generator=generator,
                          dtype=torch.double, requires_grad=True)

    if gate:
        gate = torch.rand(batch_size, generator=generator,
                          dtype=torch.double, requires_grad=True)

        # Set first and second batch to gate None (i.e., 1) and 0.0.
        gate.data[0] = 1.0
        gate.data[1] = 0.0
    else:
        gate = None

    return x, parameters, gate


# =============================================================================
# REFERENCE IMPLEMENTATIONS
# =============================================================================

def reference_log_det_J(x, y):
    """Compute the log(abs(det(J))) with autograd and numpy."""
    batch_size, n_features = x.shape

    # Compute the jacobian with autograd.
    jacobian = np.empty((batch_size, n_features, n_features))
    for i in range(n_features):
        loss = torch.sum(y[:, i])
        loss.backward(retain_graph=True)

        jacobian[:, i] = x.grad.detach().numpy()

        # Reset gradient for next calculation.
        x.grad.data.zero_()

    # Compute the log det J numerically.
    log_det_J = np.empty(batch_size)
    for batch_idx in range(batch_size):
        log_det_J[batch_idx] = np.log(np.abs(np.linalg.det(jacobian[batch_idx])))

    return log_det_J


def reference_sos_polynomial_transformer(x, coefficients, gate):
    """Reference implementation of SOSPolynomialTransformer for testing."""
    x = x.detach().numpy()
    coefficients = coefficients.detach().numpy()
    batch_size, n_coefficients, n_features = coefficients.shape
    n_polynomials = (n_coefficients - 1) // 2

    if gate is None:
        gate = np.ones(batch_size, dtype=x.dtype)
    else:
        gate = gate.detach().numpy()

    # This is the returned value.
    y = np.empty(shape=x.shape)
    det_J = np.ones(batch_size)

    for batch_idx in range(batch_size):
        for i in range(n_features):
            x_i = x[batch_idx, i]
            coefficients_i = coefficients[batch_idx, :, i]

            # Compute all squared polynomials.
            squared_polynomials = []
            for k in range(n_polynomials):
                a_k0 = coefficients_i[1 + k*2]
                a_k1 = coefficients_i[2 + k*2]
                poly = np.poly1d([a_k1, a_k0])
                squared_polynomials.append(np.polymul(poly, poly))

            # Sum the squared polynomials.
            sum_of_squares_poly = squared_polynomials[0]
            for poly in squared_polynomials[1:]:
                sum_of_squares_poly = np.polyadd(sum_of_squares_poly, poly)

            # The integrand is the derivative w.r.t. the input.
            det_J[batch_idx] *= gate[batch_idx]*np.polyval(sum_of_squares_poly, x_i) + (1 - gate[batch_idx])

            # Integrate and sum constant term.
            a_0 = coefficients_i[0]
            sum_of_squares_poly = np.polyint(sum_of_squares_poly, k=a_0)
            y[batch_idx, i] = gate[batch_idx]*np.polyval(sum_of_squares_poly, x_i) + (1 - gate[batch_idx])*x_i

    return y, np.log(np.abs(det_J))


def reference_neural_spline(x, x0, y0, widths, heights, slopes):
    """Reference implementation of neural_spline_transformer for testing."""
    x = x.detach().numpy()
    x0 = x0.detach().numpy()
    y0 = y0.detach().numpy()
    widths = widths.detach().numpy()
    heights = heights.detach().numpy()
    slopes = slopes.detach().numpy()

    batch_size, n_bins, n_features = widths.shape

    knots_x = np.empty((batch_size, n_bins+1, n_features), dtype=x.dtype)
    knots_x[:, 0] = x0
    knots_x[:, 1:] = x0 + np.cumsum(widths, axis=1)
    knots_y = np.empty((batch_size, n_bins+1, n_features), dtype=x.dtype)
    knots_y[:, 0] = y0
    knots_y[:, 1:] = y0 + np.cumsum(heights, axis=1)

    y = np.empty_like(x)
    log_det_J = np.zeros(batch_size, dtype=x.dtype)

    for batch_idx in range(batch_size):
        for feat_idx in range(n_features):
            bin_idx = np.digitize(x[batch_idx, feat_idx], knots_x[batch_idx, :, feat_idx], right=False) - 1

            xk = knots_x[batch_idx, bin_idx, feat_idx]
            xk1 = knots_x[batch_idx, bin_idx+1, feat_idx]
            yk = knots_y[batch_idx, bin_idx, feat_idx]
            yk1 = knots_y[batch_idx, bin_idx+1, feat_idx]
            if bin_idx == 0:
                deltak = 1
            else:
                deltak = slopes[batch_idx, bin_idx-1, feat_idx]
            if bin_idx == n_bins-1:
                deltak1 = 1
            else:
                deltak1 = slopes[batch_idx, bin_idx, feat_idx]

            sk = (yk1 - yk) / (xk1 - xk)
            epsilon = (x[batch_idx, feat_idx] - xk) / (xk1 - xk)

            numerator = (yk1 - yk) * (sk * epsilon**2 + deltak * epsilon * (1 - epsilon))
            denominator = sk + (deltak1 + deltak - 2*sk) * epsilon * (1 - epsilon)
            y[batch_idx, feat_idx] = yk + numerator / denominator

            numerator = sk**2 * (deltak1 * epsilon**2 + 2*sk*epsilon*(1 - epsilon) + deltak*(1 - epsilon)**2)
            denominator = (sk + (deltak1 + deltak + - 2*sk) * epsilon * (1 - epsilon))**2
            log_det_J[batch_idx] += np.log(numerator / denominator)

    return y, log_det_J


def reference_mobius_transformer(x, w, blocks, gate):
    """Reference implementation of MobiusTransformer for testing."""
    x = x.detach().numpy()
    w = w.detach().numpy()
    if gate is not None:
        gate = gate.detach().numpy()
    batch_size, n_features = x.shape

    # Blocks can be an int, in which case x is to be divided in blocks of equal size.
    if isinstance(blocks, int):
        assert n_features % blocks == 0
        blocks = [blocks] * int(n_features / blocks)

    # If there is no gate, set it to 1.
    if gate is None:
        gate = np.ones((batch_size, 1), dtype=x.dtype)
    else:
        # Add fake dimension for broadcasting.
        gate = gate[:, None]

    # Initialize the output array.
    y = np.empty_like(x)
    log_det_J = np.zeros(batch_size, dtype=x.dtype)

    # We return also the norm of the input and output.
    x_norm = np.empty(shape=(batch_size, len(blocks)))
    y_norm = np.empty(shape=(batch_size, len(blocks)))

    # The start of the next block.
    block_pointer = 0

    for block_idx, block_size in enumerate(blocks):
        # The input and parameters for the block.
        x_block = x[:, block_pointer:block_pointer+block_size]
        w_block = gate * w[:, block_pointer:block_pointer+block_size]

        # Move the x vector on the unit sphere. Keep the number
        # of dimensions so that broadcasting works.
        x_norm_block = np.linalg.norm(x_block, axis=1, keepdims=True)
        x_normalized_block = x_block / x_norm_block

        # We'll need these terms for the Jacobian as well.
        xw_block = x_normalized_block + w_block
        w_norm = np.linalg.norm(w_block, axis=1, keepdims=True)
        xw_norm = np.linalg.norm(xw_block, axis=1, keepdims=True)
        diff_w_norm = 1 - w_norm**2
        xw_norm_squared = xw_norm**2

        # Compute the output for the block.
        y_normalized_block = diff_w_norm / xw_norm_squared * xw_block + w_block
        y_block = x_norm_block * y_normalized_block

        y[:, block_pointer:block_pointer+block_size] = y_block
        x_norm[:, block_idx] = x_norm_block[:, 0]
        y_norm[:, block_idx] = np.linalg.norm(y_block, axis=1)

        # Compute dxnormalized_i/dx_j.
        dxnormalized_dx = np.empty((batch_size, block_size, block_size))
        for batch_idx in range(batch_size):
            for i in range(block_size):
                for j in range(block_size):
                    dxnormalized_dx[batch_idx, i, j] = - x_block[batch_idx, i] * x_block[batch_idx, j] / x_norm_block[batch_idx, 0]**3
                    if i == j:
                        dxnormalized_dx[batch_idx, i, j] += 1 / x_norm_block[batch_idx, 0]

        # Compute the block Jacobian dy_i/dx_j.
        jacobian = np.empty((batch_size, block_size, block_size), dtype=x.dtype)

        for batch_idx in range(batch_size):
            for i in range(block_size):
                for j in range(block_size):
                    # The first term is d||x||/dx_j * y_normalized, with (d||x||/dx_j)_i = x_j/||x||.
                    jacobian[batch_idx, i, j] = y_normalized_block[batch_idx, i] * x_normalized_block[batch_idx, j]

                    # This is the constant factor in front of the second term.
                    factor = x_norm_block[batch_idx, 0] * diff_w_norm[batch_idx, 0] / xw_norm_squared[batch_idx, 0]

                    # First and second additive terms in the numerator.
                    first_term = dxnormalized_dx[batch_idx, i, j]
                    second_term = 2 / xw_norm_squared[batch_idx, 0] * xw_block[batch_idx, i] * np.dot(xw_block[batch_idx], dxnormalized_dx[batch_idx, :, j])

                    jacobian[batch_idx, i, j] += factor * (first_term - second_term)

        # Compute the log determinant.
        for batch_idx in range(batch_size):
            log_det_J[batch_idx] += np.log(np.abs(np.linalg.det(jacobian[batch_idx])))

        # Point to next block.
        block_pointer += block_size

    return y, log_det_J, x_norm, y_norm


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features', [2, 5, 8])
@pytest.mark.parametrize('gate', [False, True])
def test_affine_transformer_round_trip(batch_size, n_features, gate):
    """Make sure the forward + inverse conposition of affine transformers is equal to the identity."""
    x, coefficients, gate = create_random_input(batch_size, n_features, gate,
                                                n_parameters=2, seed=0)
    shift, log_scale = coefficients[:, 0], coefficients[:, 1]

    # Check that a round trip gives the identity function.
    y, log_det_J_y = affine_transformer(x, shift, log_scale, gate=gate)
    x_inv, log_det_J_x_inv = affine_transformer_inv(y, shift, log_scale, gate=gate)

    assert torch.allclose(x, x_inv)
    assert torch.allclose(log_det_J_y + log_det_J_x_inv, torch.zeros_like(log_det_J_y))


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features', [2, 5, 8])
@pytest.mark.parametrize('gate', [False, True])
@pytest.mark.parametrize('func', [affine_transformer, affine_transformer_inv])
def test_affine_transformer_log_det_J(batch_size, n_features, gate, func):
    """Check that the log_det_J of the gated affine transformer is correct."""
    x, coefficients, gate = create_random_input(batch_size, n_features, gate,
                                                n_parameters=2, seed=0)
    shift, log_scale = coefficients[:, 0], coefficients[:, 1]

    # Check the log(abs(det(J))).
    y, log_det_J = func(x, shift, log_scale, gate)
    log_det_J_ref = reference_log_det_J(x, y)
    assert np.allclose(log_det_J.detach().numpy(), log_det_J_ref)


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features', [2, 5, 8])
@pytest.mark.parametrize('n_polynomials', [2, 3, 5])
@pytest.mark.parametrize('gate', [False, True])
def test_sos_polynomial_transformer_reference(batch_size, n_features, n_polynomials, gate):
    """Compare PyTorch and reference implementation of sum-of-squares transformer."""
    x, coefficients, gate = create_random_input(batch_size, n_features, gate=gate,
                                                n_parameters=1+2*n_polynomials, seed=0)

    ref_y, ref_log_det_J = reference_sos_polynomial_transformer(x, coefficients, gate)
    torch_y, torch_log_det_J = sos_polynomial_transformer(x, coefficients, gate)

    assert np.allclose(ref_y, torch_y.detach().numpy())
    assert np.allclose(ref_log_det_J, torch_log_det_J.detach().numpy())

    # Compute the reference log_det_J also with autograd and numpy.
    ref_log_det_J2 = reference_log_det_J(x, torch_y)
    assert np.allclose(ref_log_det_J2, torch_log_det_J.detach().numpy())


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features', [2, 5, 8])
@pytest.mark.parametrize('n_polynomials', [2, 3, 5])
@pytest.mark.parametrize('gate', [False, True])
def test_sos_polynomial_transformer_gradcheck(batch_size, n_features, n_polynomials, gate):
    """Run autograd.gradcheck on the SOS polynomial transformer."""
    x, coefficients, gate = create_random_input(batch_size, n_features, gate=gate,
                                                n_parameters=1+2*n_polynomials, seed=0)

    # With a None mask, the module should fall back to the native implementation.
    result = torch.autograd.gradcheck(
        func=sos_polynomial_transformer,
        inputs=[x, coefficients, gate]
    )
    assert result


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features', [2, 5, 8])
@pytest.mark.parametrize('x0', [-2, -1])
@pytest.mark.parametrize('y0', [1, 2])
@pytest.mark.parametrize('n_bins', [2, 3, 5])
def test_neural_spline_transformer_reference(batch_size, n_features, x0, y0, n_bins):
    """Compare PyTorch and reference implementation of neural spline transformer."""
    # Determine the first and final knots of the spline. We
    # arbitrarily set the domain of the first dimension to 0.0
    # to test different dimensions for different features.
    x0 = torch.full((n_features,), x0, dtype=torch.double, requires_grad=False)
    xf = -x0
    xf[0] = 0
    y0 = torch.full((n_features,), y0, dtype=torch.double, requires_grad=False)
    yf = y0 + xf - x0

    # Create widths, heights, and slopes of the bins.
    n_parameters = 3*n_bins - 1
    x, parameters, _ = create_random_input(batch_size, n_features, gate=False,
                                           n_parameters=n_parameters, seed=0,
                                           x_func=torch.rand)

    widths = torch.nn.functional.softmax(parameters[:, :n_bins], dim=1) * (xf - x0)
    heights = torch.nn.functional.softmax(parameters[:, n_bins:2*n_bins], dim=1) * (yf - y0)
    slopes = torch.nn.functional.softplus(parameters[:, 2*n_bins:])

    # x is now between 0 and 1 but it must be between x0 and xf. We detach
    # to make the new x a leaf variable and reset requires_grad.
    x = x.detach() * (xf - x0) + x0
    x.requires_grad = True

    ref_y, ref_log_det_J = reference_neural_spline(x, x0, y0, widths, heights, slopes)
    torch_y, torch_log_det_J = neural_spline_transformer(x, x0, y0, widths, heights, slopes)

    assert np.allclose(ref_y, torch_y.detach().numpy())
    assert np.allclose(ref_log_det_J, torch_log_det_J.detach().numpy())

    # Check y0, yf boundaries are satisfied
    assert torch.all(y0 < torch_y)
    assert torch.all(torch_y < yf)

    # Compute the reference log_det_J also with autograd and numpy.
    ref_log_det_J2 = reference_log_det_J(x, torch_y)
    assert np.allclose(ref_log_det_J2, torch_log_det_J.detach().numpy())


@pytest.mark.parametrize('n_features,blocks', [
    (3, 3),
    (6, 3),
    (6, 2),
    (5, [3, 2]),
    (7, [3, 2, 2]),
    (8, [3, 2, 2, 1])
])
def test_unit_cube_to_inscribed_sphere(n_features, blocks):
    """Test the mapping from unit cube to its inscribed sphere."""
    # Create a bunch of points within the hypercube with half-side = radius.
    radius = 1
    batch_size = 256
    generator = torch.Generator()
    generator.manual_seed(0)
    w = radius - 2 * radius * torch.rand(batch_size, n_features, generator=generator, dtype=torch.double)

    # In the last two batches we set two cube vertices.
    w[-1] = radius * torch.ones_like(w[-1])
    w[-2] = -radius * torch.ones_like(w[-2])

    # In the third to last batch we try to map the origin.
    w[-3] = torch.zeros_like(w[-3])

    # After the mapping, all points should be within the unit sphere.
    w_mapped = unit_cube_to_inscribed_sphere(w, blocks, shorten_last_block=True)

    blocks = generate_block_sizes(n_features, blocks, shorten_last_block=True)
    block_pointer = 0
    for block_size in blocks:
        norms = []
        for x in [w, w_mapped]:
            x_block = x[:, block_pointer:block_pointer+block_size]
            norms.append((x_block**2).sum(dim=1).sqrt())

        # The test is more meaningful if some of the initial vectors
        # started outside the hypersphere. Exclude the vertices since
        # those are always outside the sphere.
        if block_size > 1:
            assert (norms[0][:-2] > radius).any()
        assert (norms[1] <= radius).all()

        # The cube vertices should be mapped exactly on the sphere surface.
        assert torch.allclose(norms[1][-2:], radius * torch.ones_like(norms[1][-2:]))

        # And the zero should be mapped to zero.
        zero_block = w_mapped[-3, block_pointer:block_pointer+block_size]
        assert torch.all(zero_block == torch.zeros_like(zero_block))

        block_pointer += block_size


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features,blocks', [
    (3, 3),
    (6, 3),
    (6, 2),
    (5, [3, 2]),
    (7, [3, 2, 2]),
    (8, [3, 2, 2, 1])
])
@pytest.mark.parametrize('gate', [False, True])
def test_mobius_transformer_reference(batch_size, n_features, blocks, gate):
    """Compare PyTorch and reference implementation of sum-of-squares transformer."""
    x, w, gate = create_random_input(batch_size, n_features, gate=gate,
                                     n_parameters=1, seed=0, par_func=torch.rand)
    w = 1 - 2 * w[:, 0]

    # Compare PyTorch and reference.
    ref_y, ref_log_det_J, ref_x_norm, ref_y_norm = reference_mobius_transformer(x, w, blocks, gate)
    torch_y, torch_log_det_J = mobius_transformer(x, w, blocks, gate)

    assert np.allclose(ref_y, torch_y.detach().numpy())
    assert np.allclose(ref_log_det_J, torch_log_det_J.detach().numpy())

    # Make sure the transform doesn't alter the distance from the center of the sphere.
    assert np.allclose(ref_x_norm, ref_y_norm)

    # Compute the reference log_det_J also with autograd and numpy.
    ref_log_det_J2 = reference_log_det_J(x, torch_y)
    assert np.allclose(ref_log_det_J2, torch_log_det_J.detach().numpy())


@pytest.mark.parametrize('batch_size', [2, 5])
@pytest.mark.parametrize('n_features,blocks', [
    (3, 3),
    (6, 3),
    (6, 2),
    (5, [3, 2]),
    (7, [3, 2, 2]),
    (8, [3, 2, 2, 1])
])
@pytest.mark.parametrize('gate', [False, True])
def test_mobius_transformer_gradcheck(batch_size, n_features, blocks, gate):
    """Run autograd.gradcheck on the Mobius transformer."""
    x, w, gate = create_random_input(batch_size, n_features, gate=gate,
                                     n_parameters=1, seed=0, par_func=torch.rand)
    w = 1 - 2 * w[:, 0]

    # With a None mask, the module should fall back to the native implementation.
    result = torch.autograd.gradcheck(
        func=mobius_transformer,
        inputs=[x, w, blocks, gate]
    )
    assert result
