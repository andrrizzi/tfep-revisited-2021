#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility functions to handle geometry with PyTorch.
"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def to_batch_atom_3_shape(positions):
    """Reshape a tensor or numpy array to have shape (batch_size, n_atoms, 3).

    This allows converting positions in flattened format, as yielded from
    the ``TrajectoryDataset`` object into the standard shape used by
    MDAnalysis trajectories.

    Parameters
    ----------
    positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        The input can have the following shapes: (batch_size, n_atoms, 3),
        (batch_size, n_atoms * 3), (n_atoms, 3).

    Returns
    -------
    reshaped_positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        A view of the original tensor or array with shape (batch_size, n_atoms, 3).

    """
    if (len(positions.shape) == 2 and positions.shape[-1] != 3 or
                len(positions.shape) == 3):
        # (batch_size, n_atoms * 3) or (batch_size, n_atoms, 3).
        batch_size = positions.shape[0]
    else:
        batch_size = 1

    if positions.shape[-1] != 3:
        n_atoms = positions.shape[-1] // 3
    else:
        n_atoms = positions.shape[-2]

    standard_shape = (batch_size, n_atoms, 3)
    if positions.shape != standard_shape:
        positions = positions.reshape(standard_shape)

    return positions


def to_atom_batch_3_shape(positions):
    """Reshape a tensor or numpy array to have shape (n_atoms, batch_size, 3).

    Parameters
    ----------
    positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        The input can have the following shapes: (batch_size, n_atoms, 3),
        (batch_size, n_atoms * 3), (n_atoms, 3).

    Returns
    -------
    reshaped_positions : torch.Tensor, numpy.ndarray, or pint.Quantity
        A view of the original tensor or array with shape (n_atoms, batch_size, 3).

    """
    positions = to_batch_atom_3_shape(positions)
    try:
        # PyTorch tensor.
        return positions.permute(1, 0, 2)
    except AttributeError:
        # Numpy array.
        return positions.swapaxes(0, 1)


def batchwise_atom_dist(batch_positions_atom1, batch_positions_atom2):
    """Compute the pairwise Euclidean distance between the two atom for all batch samples.

    Parameters
    ----------
    batch_positions_atom1 : torch.Tensor
        The positions of the first atom with shape (batch_size, 3).
    batch_positions_atom2 : torch.Tensor
        The positions of the second atom with shape (batch_size, 3).

    Returns
    -------
    distances : torch.Tensor
        ``distances[i]`` is the Euclidean distance between positions
        ``batch_positions_atom1[i]`` and ``batch_positions_atom2[i]``.

    """
    diff = (batch_positions_atom2 - batch_positions_atom1)**2
    try:
        # Tensor.
        return diff.sum(dim=1).sqrt()
    except TypeError:
        # Numpy array.
        return np.sqrt(diff.sum(axis=1))
