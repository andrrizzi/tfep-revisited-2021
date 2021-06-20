#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module loss.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pytest
import scipy.stats
import scipy.special
import torch

from ..loss import CVBoltzmannKLDivLoss, CVMetaDKLDivLoss, CVAlphaDiv2Loss


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def reference_cv_boltzmann_kl_div_loss(cv_hist_edges, n_nonempty_bins, potentials1, potentials2, cvs, metad_rbias):
    """Reference implementation of the CVBoltzmannKLDivLoss loss function.

    Assumes kT is 1. Return the loss and its gradient w.r.t. potentials2.

    """
    n_bins = len(cv_hist_edges) - 1
    metad_rbias = metad_rbias.detach().numpy()
    cv_indices = np.digitize(cvs, cv_hist_edges) - 1
    normalizing_factors = np.array([
        np.sum(np.exp(metad_rbias[np.where(cv_indices == bin_idx)]))
        for bin_idx in range(n_bins)
    ])
    reweighting_factors = torch.tensor(np.exp(metad_rbias) / normalizing_factors[cv_indices])

    loss = torch.sum(reweighting_factors * (potentials2 - potentials1)) / n_nonempty_bins
    loss_grad = reweighting_factors / n_nonempty_bins
    return loss, loss_grad


def reference_cv_metad_kl_div_loss(cv_hist_edges, n_nonempty_bins, potentials1, potentials2, cvs, metad_rbias):
    """Reference implementation of the CVMetaDKLDivLoss loss function.

    Assumes kT is 1. Return the loss and its gradient w.r.t. potentials2.
    """
    cv_indices = np.digitize(cvs, cv_hist_edges) - 1
    cv_hist, _ = np.histogram(cvs, cv_hist_edges)
    n_samples = torch.tensor(cv_hist[cv_indices], dtype=potentials2.dtype)
    loss = torch.sum((potentials2 - potentials1 - metad_rbias) / n_samples) / n_nonempty_bins
    loss_prime = 1 / n_samples / n_nonempty_bins
    return loss, loss_prime


def reference_cv_alpha_div_2_loss(cv_hist_edges, n_nonempty_bins, potentials1, potentials2, cvs, metad_rbias):
    """Reference implementation of the CVMetaDKLDivLoss loss function.

    Assumes kT is 1. Return the loss and its gradient w.r.t. potentials2.
    """
    metad_rbias = metad_rbias.detach().numpy()
    reduced_work = (potentials2 - potentials1).detach().numpy() + metad_rbias

    cv_logsumexp_w, _, _ = scipy.stats.binned_statistic(cvs, reduced_work, scipy.special.logsumexp, bins=cv_hist_edges)
    cv_normalization_factors, _, _ = scipy.stats.binned_statistic(cvs, metad_rbias, scipy.special.logsumexp, bins=cv_hist_edges)

    loss = np.nansum(cv_logsumexp_w - cv_normalization_factors) / n_nonempty_bins
    cv_indices = np.digitize(cvs, cv_hist_edges) - 1
    loss_prime = np.exp(reduced_work - cv_logsumexp_w[cv_indices]) / n_nonempty_bins

    return torch.tensor(loss, dtype=potentials2.dtype), torch.tensor(loss_prime, dtype=potentials2.dtype)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('loss_cls,reference_func',[
    (CVBoltzmannKLDivLoss, reference_cv_boltzmann_kl_div_loss),
    (CVMetaDKLDivLoss, reference_cv_metad_kl_div_loss),
    (CVAlphaDiv2Loss, reference_cv_alpha_div_2_loss),
])
def test_cv_kl_div_loss(loss_cls, reference_func):
    """Test the static constructor and loss function of CVBoltzmannKLDivLoss."""
    kT = 1
    cv_hist_edges = np.array([-2, -1., 0, 1, 2, 3, 4])
    cvs = np.array([0.1, 3.3, 1.1, 3.5, 0.7])
    n_nonempty_bins = 3

    metad_rbias = torch.tensor([1.0, 1.2, 1.5, 1.2, 1.0], dtype=torch.float64)
    potentials1 = torch.tensor(np.arange(len(cvs), dtype=np.float64))
    potentials2 = torch.tensor(np.arange(len(cvs), dtype=np.float64)[len(cvs)::-1] + 1, requires_grad=True)

    # Get the reference value for the loss.
    expected_loss, expected_loss_grad = reference_func(
        cv_hist_edges, n_nonempty_bins, potentials1, potentials2, cvs, metad_rbias)

    # Compute the grad w.r.t. potentials2.
    loss = loss_cls(kT, cv_hist_edges)
    loss_value = loss(potentials1, potentials2, cvs, metad_rbias=metad_rbias)
    loss_value.backward()

    assert torch.isclose(loss_value, expected_loss)
    assert torch.allclose(potentials2.grad, expected_loss_grad)
