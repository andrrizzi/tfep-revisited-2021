#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module reweighting.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import tempfile

import numpy as np
from numpy.random import RandomState
import pint

from ..reweighting import DatasetReweighting


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Makes random test cases deterministic.
_random_state = RandomState(0)

_ureg = pint.UnitRegistry()


# =============================================================================
# TEST UTILITIES
# =============================================================================

class DummyStdReweighting(DatasetReweighting):
    """Dummy implementation of standard reweighting for testing."""

    U0 = 0.0

    def compute_potentials(self, batch_positions):
        kJ_mol = _ureg.kJ / _ureg.mol
        return (self.U0 + _random_state.rand(len(batch_positions))) * kJ_mol

    def get_traj_info(self):
        kJ_mol = _ureg.kJ / _ureg.mol
        cvs = np.array(range(len(self.dataset)))
        reference_potentials = _random_state.rand(len(cvs)) * kJ_mol
        metad_rbias = np.zeros(len(cvs)) * kJ_mol
        return cvs, reference_potentials, metad_rbias


# =============================================================================
# TESTS
# =============================================================================

def test_standard_reweighting_potentials_cache():
    """Test that DatasetReweighting caches and reuses the potentials correctly."""
    import MDAnalysis.coordinates
    from ..data import TrajectoryDataset, TrajectorySubset

    def _get_potentials(dataset, file_path, u0, indices, batch_size, write_interval):
        subset = TrajectorySubset(dataset, indices=indices)
        DummyStdReweighting.U0 = u0
        reweighting = DummyStdReweighting(
            subset, n_bins=len(subset), temperature=300*_ureg.kelvin,
            potentials_file_path=file_path)
        return reweighting.compute_dataset_potentials(
            batch_size=batch_size, write_interval=write_interval)

    # Load the test PDB file.
    pdb_file_path = os.path.join(os.path.dirname(__file__), 'data', 'chloro-fluoromethane.pdb')
    with MDAnalysis.coordinates.PDB.PDBReader(pdb_file_path) as trajectory:
        dataset = TrajectoryDataset(trajectory, return_batch_index=True)

        # Cache the potentials in a temporary file.
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, 'potentials.npz')

            # Cache a first value for the potentials of some of the frames.
            u1 = 10
            potentials1 = _get_potentials(dataset, file_path, u1, indices=[0, 2, 4],
                                          batch_size=1, write_interval=2)
            assert np.all((0 <= potentials1.magnitude - u1) &  (potentials1.magnitude - u1 < 1))

            # Check that what we have just computed does not get re-computed.
            u2 = 20
            potentials2 = _get_potentials(dataset, file_path, u2, indices=[1, 3, 4],
                                          batch_size=5, write_interval=2)
            assert potentials1[-1] == potentials2[-1]
            assert np.all((0 <= potentials2.magnitude[:-1] - u2) &  (potentials2.magnitude[:-1] - u2 < 1))

            # The cache should be up-to-date.
            times, potentials = DummyStdReweighting.load_cached_potentials_from_file(file_path)
            assert not np.isnan(potentials).any()
