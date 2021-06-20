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

import numpy as np
import pytest
import torch

from ..modules.linear import MaskedLinear
from ..modules.flows import MAF, NormalizingFlow
from ..precond import _Preconditioner, _BatchGradPreconditioner, GradCovPreconditioner


# =============================================================================
# FIXTURES
# =============================================================================

def generate_test_nets(dtype=None):
    """Lazily generates the nets with the correct default types"""
    if dtype != None:
        old_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    nets = [
        torch.nn.Linear(in_features=4, out_features=5, bias=True),
        torch.nn.Linear(in_features=5, out_features=5, bias=False),
        MaskedLinear(in_features=6, out_features=5, bias=True,
                     mask=torch.tril(torch.ones(5, 6), diagonal=0)),
        MaskedLinear(in_features=5, out_features=5, bias=True,
                     mask=torch.tril(torch.ones(5, 5), diagonal=-1)),
        MaskedLinear(in_features=4, out_features=5, bias=True,
                     mask=torch.tril(torch.ones(5, 4), diagonal=0)[:, (1, 3, 0, 2)]),
        torch.nn.Sequential(
            torch.nn.Linear(in_features=6, out_features=5, bias=True),
            torch.nn.ELU(),
            MaskedLinear(in_features=5, out_features=6, bias=True,
                         mask=torch.tril(torch.ones(6, 5), diagonal=-1)),
            torch.nn.Tanh()
        ),
        MAF(dimension=5, dimension_conditioning=1, split_conditioner=True),
        NormalizingFlow(
            MAF(dimension=5, dimension_conditioning=1, split_conditioner=True),
            MAF(dimension=5, dimension_conditioning=1, split_conditioner=False),
            dimension_gate=1
        )
    ]

    if dtype != None:
        torch.set_default_dtype(old_default_dtype)

    return nets


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _get_net_in_features(net):
    linear_modules = [m for m in net.modules() if hasattr(m, 'in_features')]
    return linear_modules[0].in_features


def _forward_and_backward(net, x, loss_func):
    # Flows return a tuple with the output and the log_det_J terms.
    y = net(x)
    if isinstance(y, tuple):
        y = y[0]

    loss = loss_func(y)
    loss.backward()

    return y, loss


def _vectorize_gradient(param_groups):
    """Return the gradient of all parameters as a 1D numpy array."""
    # Collect all gradients.
    all_grads = []
    for group in param_groups:
        for p in group['params']:
            grad = p.grad.numpy()
            if len(grad.shape) > 1:
                grad = np.reshape(grad, grad.size)
            all_grads.append(grad)
    return np.concatenate(all_grads)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('net', generate_test_nets())
def test_precondition(net):
    """Check that, given the precision matrix, _Preconditioner conditions the gradient correctly."""
    batch_size = 2
    in_features = _get_net_in_features(net)
    n_parameters = sum([p.numel() for p in net.parameters()])

    # Create preconditioner.
    loss_func = torch.sum
    preconditioner = _Preconditioner(net.parameters())

    # Generate random input and precision matrix. Make sure the test is reproducible.
    generator = torch.Generator()
    generator.manual_seed(0)
    x = torch.randn(batch_size, in_features, generator=generator)
    precision_matrix = torch.randn(n_parameters, n_parameters, generator=generator)

    # Compute gradients.
    _forward_and_backward(net, x, loss_func)

    # Compute the conditioned gradients.
    grad_vec_ref = _vectorize_gradient(preconditioner.param_groups)
    conditioned_grad_vec_ref = precision_matrix.numpy().dot(grad_vec_ref)

    # Precondition the gradient.
    preconditioner.precondition_grad(precision_matrix)
    conditioned_grad_vec = _vectorize_gradient(preconditioner.param_groups)

    # Reset gradient for next computation.
    net.zero_grad()

    assert np.allclose(conditioned_grad_vec, conditioned_grad_vec_ref, atol=1e-5)


@pytest.mark.parametrize('damping', [0.0, 1.0])
def test_damp(damping):
    """Check that _Preconditioner.damp() works correctly."""
    # Generate random matrix and parameter. Make sure the test is reproducible.
    generator = torch.Generator()
    generator.manual_seed(0)
    m = torch.randn(3, 3, generator=generator)
    parameter = torch.nn.Parameter(torch.randn(4, generator=generator))

    # Initialize the preconditioner with a fake parameter list.
    preconditioner = _Preconditioner(params=[parameter], damping=damping)

    # Compute the expected damping matrix.
    m_ref = m + damping * torch.eye(3)

    # _Preconditioner.damp() does not instantiate a full diagonal matrix
    # to save time and memory. Check that the method computes it correctly.
    preconditioner.damp(m)

    assert torch.allclose(m_ref, m)


@pytest.mark.parametrize('net', generate_test_nets())
@pytest.mark.parametrize('loss_type', ['sum', 'mean'])
def test_batchwise_gradient(net, loss_type):
    """Test that the gradients w.r.t. to each batch samples are computed correctly."""
    batch_size = 2

    if loss_type == 'sum':
        loss_func = torch.sum
    else:
        loss_func = torch.mean

    # Collect all the modules in the network with training parameters.
    modules = []
    for mod in net.modules():
        if len(list(mod.parameters(recurse=False))) > 0:
            modules.append(mod)

    # Generate random input. Make sure the test is reproducible.
    generator = torch.Generator()
    generator.manual_seed(0)
    x = torch.randn(batch_size, modules[0].in_features, generator=generator)

    # Initialize reference calculation.
    batch_grad_w_ref = []
    batch_grad_b_ref = []
    for mod in modules:
        batch_grad_w_ref.append(torch.empty(batch_size, mod.out_features, mod.in_features))
        if mod.bias is None:
            batch_grad_b_ref.append(None)
        else:
            batch_grad_b_ref.append(torch.empty(batch_size, mod.out_features))

    # Compute the batchwise gradients for each module manually.
    for batch_idx in range(batch_size):
        y, loss = _forward_and_backward(net, x[batch_idx:batch_idx+1], loss_func)

        for mod_idx, mod in enumerate(modules):
            batch_grad_w_ref[mod_idx][batch_idx] = mod.weight.grad
            if mod.bias is not None:
                batch_grad_b_ref[mod_idx][batch_idx] = mod.bias.grad

        # Reset gradients for next batch.
        net.zero_grad()

    # Compute the batchwise gradients with the facilities.
    preconditioner = _BatchGradPreconditioner(net, loss_type=loss_type)
    y, loss = _forward_and_backward(net, x, loss_func)

    # Compare the gradients module by module.
    for mod_idx, mod in enumerate(modules):
        batch_grad_w, batch_grad_b = preconditioner.compute_module_batchwise_grad(mod)

        assert torch.allclose(batch_grad_w, batch_grad_w_ref[mod_idx])
        if mod.bias is not None:
            assert torch.allclose(batch_grad_b, batch_grad_b_ref[mod_idx])

    # Clear gradients for next test case.
    net.zero_grad()


@pytest.mark.parametrize('net', generate_test_nets(dtype=torch.double))
@pytest.mark.parametrize('damping', [0.0, 1.0])
@pytest.mark.parametrize('loss_type', ['mean', 'sum'])
def test_grad_cov_preconditioner(net, damping, loss_type):
    """Test GradCovPreconditioner against a reference implementation with numpy."""

    batch_size = 10
    in_features = _get_net_in_features(net)

    # The pseudoinverse is much more stable in double precision.
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)

    # Generate random input. Make sure the test is reproducible.
    generator = torch.Generator()
    generator.manual_seed(0)
    x = torch.randn(batch_size, in_features, generator=generator)

    # At this point the dtype should be consistent for the rest of the test.
    torch.set_default_dtype(old_default_dtype)

    # Create preconditioner and optimizer.
    loss_func = torch.sum if loss_type == 'sum' else torch.mean
    preconditioner = GradCovPreconditioner(net, loss_type=loss_type, damping=damping)

    # Compute gradients.
    y, loss = _forward_and_backward(net, x, loss_func)

    # Compute conditioned gradient with numpy.
    all_grads_vec = preconditioner.compute_all_batchwise_grads(clear=False).numpy()
    cov = np.cov(all_grads_vec.T, ddof=1)
    cov += damping * np.eye(cov.shape[0])
    precision = np.linalg.pinv(cov)
    grad_vec = _vectorize_gradient(preconditioner.param_groups)
    conditioned_grad_vec_ref = precision.dot(grad_vec)

    # Compute with the preconditioner.
    preconditioner.step()
    conditioned_grad_vec = _vectorize_gradient(preconditioner.param_groups)

    assert np.allclose(conditioned_grad_vec, conditioned_grad_vec_ref)

    # The gradient of the masked parameters should still be 0.0 after conditioning.
    for group in preconditioner.param_groups:
        for par in group['params']:
            if hasattr(group['mod'], 'mask') and len(par.shape) == 2:
                assert (par.grad * (1 - group['mod'].mask) == 0).byte().all()
