#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Loss functions to train PyTorch normalizing flows for reweighting.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
from scipy.special import logsumexp
import torch


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class KLDivLoss(torch.nn.Module):
    """A loss function whose minimum corresponds to that of the KL divergence D[p_1||p_2].

    The loss function assumes the sampling is done in the p_1 distribution.
    If the samples are mapped through a bijection, the log absolute Jacobian
    determinant of the transformation must be passed as well.

    The loss function differs from the actual KL divergence by a constant
    additive term log(Z_2/Z_1).

    Parameters
    ----------
    kT : float
        The thermodynamic temperature of both ensembles in the units of
        energy that are passed to the loss function.

    """

    def __init__(self, kT):
        super().__init__()
        self._kT = kT

    def forward(self, potentials1, potentials2, log_det_J=None):
        """Compute the loss.

        Parameters
        ----------
        potentials1 : torch.Tensor
            The potential energy of the samples collected from distribution
            p_1 and evaluated using the potential of p_1. The shape is
            ``(batch_size)``.
        potentials2 : torch.Tensor
            The potential energy of the samples collected from distribution
            p_1 and evaluated using the potential of p_2. The shape is
            ``(batch_size)``.
        log_det_J : torch.Tensor, optional
            The logarithm of the absolute value of the determinant of the
            Jacobian of the bijection mapping the samples. The shape is
            ``(batch_size)``.

        """
        reduced_work = self._compute_reduced_work(potentials1, potentials2, log_det_J)
        return torch.mean(reduced_work)

    def _compute_reduced_work(self, potentials1, potentials2, log_det_J):
        reduced_work = (potentials2 - potentials1) / self._kT
        if log_det_J is not None:
            reduced_work -= log_det_J
        return reduced_work


class BoltzmannKLDivLoss(KLDivLoss):
    """A loss function whose minimum corresponds to that of the KL divergence D[p_1||p_2].

    The loss function assumes the sampling is done in the p_1 distribution.
    If the sampling is done with metadynamics, it is necessary to pass the
    normalized bias to the loss function so that the average is appropriately
    weighted. If the samples are mapped through a bijection, the log absolute
    Jacobian determinant of the transformation must be passed as well.

    The loss function differs from the actual KL divergence by a constant
    additive log(Z_2/Z_1).

    Parameters
    ----------
    kT : float
        The thermodynamic temperature of both ensembles in the units of
        energy that are passed to the loss function.

    """

    def forward(self, potentials1, potentials2, log_det_J=None, metad_rbias=None):
        """Compute the loss.

        Parameters
        ----------
        potentials1 : torch.Tensor
            The potential energy of the samples collected from distribution
            p_1 (or p_V if metadynamics is used) and evaluated using the
            potential of p_1. The shape is ``(batch_size)``.
        potentials2 : torch.Tensor
            The potential energy of the samples collected from distribution
            p_1 and evaluated using the potential of p_2. The shape is
            ``(batch_size)``.
        log_det_J : torch.Tensor, optional
            The logarithm of the absolute value of the determinant of the
            Jacobian of the bijection mapping the samples. The shape is
            ``(batch_size)``.
        metad_rbias : torch.Tensor, optional
            The metadynamics bias normalized by c(t) that is used to
            reweighting metadynamics samples to Boltzmann. The shape is
            ``(batch_size)``.

        """
        reduced_work = self._compute_reduced_work(potentials1, potentials2, log_det_J)
        if metad_rbias is not None:
            weights = torch.nn.functional.softmax(metad_rbias / self._kT)
            return torch.sum(weights * reduced_work)
        return torch.mean(reduced_work)


class MetaDKLDivLoss(KLDivLoss):
    """A loss function whose minimum corresponds to that of the KL divergence D[p_V||p_2].

    The loss function assumes the sampling is done in the metadynamics
    ensemble from a distribution p_1. If the samples are mapped through
    a bijection, the log absolute Jacobian determinant of the transformation
    must be passed as well.

    The loss function differs from the actual KL divergence by an additive
    log(Z_2/Z_1) factor.

    Parameters
    ----------
    kT : float
        The thermodynamic temperature of both ensembles in the units of
        energy that are passed to the loss function.

    """

    def forward(self, potentials1, potentials2, log_det_J=None, metad_rbias=None):
        """Compute the loss.

        Parameters
        ----------
        potentials1 : torch.Tensor
            The potential energy of the samples collected from biased
            distribution p_V and evaluated using the potential of p_1.
            The shape is ``(batch_size)``.
        potentials2 : torch.Tensor
            The potential energy of the samples collected from distribution
            p_V and evaluated using the potential of p_2. The shape is
            ``(batch_size)``.
        log_det_J : torch.Tensor, optional
            The logarithm of the absolute value of the determinant of the
            Jacobian of the bijection mapping the samples. The shape is
            ``(batch_size)``.
        metad_rbias : torch.Tensor, optional
            The metadynamics bias normalized by c(t) that is used to
            reweighting metadynamics samples to Boltzmann. The shape is
            ``(batch_size)``. This is not affected by the parameters
            and can be left out for optimization purposes.

        """
        if metad_rbias is not None:
            potentials1 = potentials1 + metad_rbias
        return super().forward(potentials1, potentials2, log_det_J)

