#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Function to compute restraint potentials with PyTorch.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch


# =============================================================================
# RESTRAINTS IMPLEMENTED IN PLUMED
# =============================================================================

def _walls_plumed(arg, at, kappa, offset=0.0, exp=2.0, eps=1.0, upper_wall=True):
    # We apply the bias only if arg > at - offset.
    dist = arg - at + offset
    if upper_wall:
        dist = torch.nn.functional.relu(dist)
    else: # Lower wall
        dist = torch.nn.functional.relu(-dist)
    return kappa * (dist / eps)**exp


def upper_walls_plumed(*args, **kwargs):
    """A restraint that is zero if the argument is above a certain threshold.

    The restraint potential energy is given by

        kappa * ((arg - at + offset) / eps)**exp

    if arg - at + offset is greater than 0, and 0 otherwise.

    Parameters
    ----------
    arg : torch.Tensor
        A 1D tensor of size N with the input variables for the restraint.
    at : float, torch.Tensor
        The threshold at which (without an offset) the restraint kicks in.
    offset : float, torch.Tensor
        An offset for the argument treshold.
    kappa : float, torch.Tensor
        The restraint force constant.
    exp : float, torch.Tensor
        The exponent of the displacement.

    Returns
    -------
    energy : torch.Tensor
        A 1D tensor of size N with the energy for each arg.

    """
    return _walls_plumed(*args, upper_wall=True, **kwargs)


def lower_walls_plumed(*args, **kwargs):
    """A restraint that is zero if the argument is below a certain threshold.

    The restraint potential energy is given by

        kappa * ((arg - at + offset) / eps)**exp

    if arg - at + offset is less than 0, and 0 otherwise.

    Parameters
    ----------
    arg : torch.Tensor
        A 1D tensor of size N with the input variables for the restraint.
    at : float, torch.Tensor
        The threshold at which (without an offset) the restraint kicks in.
    offset : float, torch.Tensor
        An offset for the argument treshold.
    kappa : float, torch.Tensor
        The restraint force constant.
    exp : float, torch.Tensor
        The exponent of the displacement.
    
    Returns
    -------
    energy : torch.Tensor
        A 1D tensor of size N with the energy for each arg.

    """
    return _walls_plumed(*args, upper_wall=False, **kwargs)
