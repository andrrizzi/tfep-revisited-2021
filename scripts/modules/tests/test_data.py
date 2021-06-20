#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module data.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os

import numpy as np
from numpy.random import RandomState
import pytest

from ..data import RemoveTransRotDOF, TrajectoryDataset, TrajectorySubset


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Makes random test cases deterministic.
_random_state = RandomState(0)


# =============================================================================
# TESTS
# =============================================================================

# Ten random tests on different sets of random positions.
@pytest.mark.parametrize('positions', [_random_state.randn(5, 3) * 10 for _ in range(10)])
def test_remove_translational_rotational_dof(positions):
    """Test the RemoveTransRotDOF transformation."""
    class DummyTimeStep:
        def __init__(self, positions):
            self.positions = positions
            self.n_atoms = len(positions)
    ts = DummyTimeStep(positions=positions)

    transform = RemoveTransRotDOF(
        center_atom_idx=0,
        axis_atom_idx=2,
        plane_atom_idx=3,
        axis='z',
        plane='xz'
    )
    ts = transform(ts)

    # The center atom is at the origin.
    assert np.allclose(ts.positions[transform._center_atom_idx], 0.0)

    # The axis atom is on the expected axis.
    expected_axis_atom_position = transform.axis * np.linalg.norm(
        positions[transform._axis_atom_idx] - positions[transform._center_atom_idx])
    assert np.allclose(ts.positions[transform._axis_atom_idx], expected_axis_atom_position)

    # The plane atom is orthogonal to the plane normal.
    assert np.allclose(np.dot(ts.positions[transform._plane_atom_idx], transform.plane), 0.0)

    # The constrained atom indices is correct.
    expected = np.array([0, 1, 2, 6, 7, 10])
    assert np.all(transform.constrained_dof_indices == expected)


def test_trajectory_subset_indices():
    """Check that the indices of nested subsets are handled correctly in TrajectorySubset."""
    import MDAnalysis.coordinates
    from torch.utils.data import DataLoader

    # Load the test PDB file.
    pdb_file_path = os.path.join(os.path.dirname(__file__), 'data', 'chloro-fluoromethane.pdb')
    with MDAnalysis.coordinates.PDB.PDBReader(pdb_file_path) as trajectory:
        # Create a nested subset of a TrajectoryDataset.
        dataset = TrajectoryDataset(trajectory, return_batch_index=True)
        nested_subset = TrajectorySubset(dataset, indices=[0, 2, 4])
        subset = TrajectorySubset(nested_subset, indices=[0, 2])

        assert subset.trajectory == dataset.trajectory
        assert len(subset) == 2

        # Check that the expected indices match.
        expected_traj_indices = [0, 4]
        for subset_idx, traj_idx in enumerate(expected_traj_indices):
            # We need to copy the positions since the Timestep.position
            # array gets overwritten when you switch to a different frame.
            subset_positions = subset.get_ts(subset_idx).positions.copy()
            traj_positions = dataset.get_ts(traj_idx).positions.copy()
            assert np.all(subset_positions == traj_positions)

        # Check that the returned batch index is correct.
        data_loader = DataLoader(subset, batch_size=1, shuffle=False)
        for batch_idx, batch in enumerate(data_loader):
            assert len(batch['index']) == 1
            assert batch['index'][0] == batch_idx


def _check_saved_trajectory(saved_pdb_file_path, reference_trajectory, indices=None):
    """Check that the frames of the saved trajectory have identical positions of reference_trajectory[indices]."""
    import MDAnalysis.coordinates

    if indices is None:
        indices = range(len(reference_trajectory))

    with MDAnalysis.coordinates.PDB.PDBReader(saved_pdb_file_path) as saved_trajectory:
        for saved_idx, reference_idx in enumerate(indices):
            saved_positions = saved_trajectory[saved_idx].positions

            # Check if the reference is a trajectory or custom positions.
            if isinstance(reference_trajectory, np.ndarray):
                reference_positions = reference_trajectory[reference_idx]
            else:
                reference_positions = reference_trajectory.get_ts(reference_idx).positions

            assert np.allclose(saved_positions, reference_positions, atol=1e-2, rtol=0.0)


def test_save_trajectory_dataset():
    """Test save() method in TrajectoryDataset and SubsetTrajectory."""
    import tempfile
    import MDAnalysis.coordinates

    pdb_file_path = os.path.join(os.path.dirname(__file__), 'data', 'chloro-fluoromethane.pdb')

    # Load the test PDB file.
    with MDAnalysis.coordinates.PDB.PDBReader(pdb_file_path) as trajectory:

        # Create a nested subset of a TrajectoryDataset.
        dataset = TrajectoryDataset(trajectory)
        nested_subset = TrajectorySubset(dataset, indices=[0, 2, 4])
        subset = TrajectorySubset(nested_subset, indices=[0, 2])

        for d, indices in [
            (dataset, range(len(trajectory))),
            (subset, [0, 4])
        ]:
            for custom_positions in [
                None,
                _random_state.randn(len(d), trajectory.n_atoms, 3) * 10
            ]:
                # We write the file in a temporary location.
                temp_file_path = None
                try:
                    f = tempfile.NamedTemporaryFile(delete=True, suffix='.pdb')
                    temp_file_path = f.name
                    f.close()

                    d.save(
                        pdb_file_path,
                        output_file_path=temp_file_path,
                        positions=custom_positions,
                        multiframe=True
                    )

                    # Check that the function has saved the correct positions/frames.
                    if custom_positions is None:
                        _check_saved_trajectory(temp_file_path, dataset, indices=indices)
                    else:
                        _check_saved_trajectory(temp_file_path, custom_positions)

                        # Check also that writing custom positions leave the original data intact.
                        for saved_idx, reference_idx in enumerate(indices):
                            original_positions = dataset.get_ts(reference_idx).positions
                            assert not np.allclose(original_positions, custom_positions[saved_idx],
                                                   atol=1e-2, rtol=0.0)
                finally:
                    # Make sure the temporary file is deleted.
                    if temp_file_path is not None:
                        os.unlink(temp_file_path)

