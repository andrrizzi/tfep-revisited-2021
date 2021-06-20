#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module functions.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.autograd

from ..functions.qm.psi4 import (notorch_potential_energy_psi4, notorch_force_psi4,
                                 potential_energy_psi4)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_water_molecule(batch_size=None):
    """Crete a water Molecule object with batch positions."""
    import pint
    from ..molecule import Molecule

    ureg = pint.UnitRegistry()

    # Water molecule with basic positions.
    molecule = Molecule(
        geometry=np.array([
            [-0.2950, -0.2180, 0.1540],
            [-0.0170, 0.6750, 0.4080],
            [0.3120, -0.4570, -0.5630],
        ], dtype=np.double) * ureg.angstrom,
        symbols=['O', 'H', 'H'],
    )

    if batch_size is not None:
        # Create small random perturbations around the initial geometry.
        # Using a RandomState with fixed seed makes it deterministic.
        random_state = np.random.RandomState(0)
        batch_positions = np.empty(shape=(batch_size, molecule.geometry.shape[0], 3), dtype=np.double)
        batch_positions *= ureg.angstrom
        for batch_idx in range(batch_size):
            perburbation = random_state.uniform(-0.3, 0.3, size=molecule.geometry.shape)
            batch_positions[batch_idx] = molecule.geometry + perburbation * ureg.angstrom
    else:
        batch_positions = None

    return molecule, batch_positions


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('batch_size', [None, 1, 2])
def test_notorch_potential_energy_psi4_batch_positions(batch_size):
    """Test that notorch_potential_energy_psi4 function handles batch_positions correctly."""
    molecule, batch_positions = create_water_molecule(batch_size=batch_size)

    # Test both with and without batches.
    ureg = molecule.geometry._REGISTRY
    if batch_size is not None:
        expected_potentials = np.array([-75.96441684781715, -76.03989190203629]) * ureg.hartree
        expected_potentials = expected_potentials[:batch_size]
    else:
        expected_potentials = -76.05605256451271 * ureg.hartree

    potentials = notorch_potential_energy_psi4(
        method='scf',
        molecule=molecule,
        batch_positions=batch_positions,
        psi4_global_options=dict(basis='cc-pvtz', reference='RHF'),
        psi4_output_file_path='quiet',
    )
    assert np.allclose(potentials.magnitude, expected_potentials.magnitude)


@pytest.mark.parametrize('batch_size,method', [
    (None, 'scf'),
    (1, 'scf'),
    (2, 'scf'),
    (None, 'mp2'),
])
def test_notorch_force_psi4_batch_positions(batch_size, method):
    """Test that notorch_force_psi4 function handles batch_positions correctly."""
    molecule, batch_positions = create_water_molecule(batch_size=batch_size)
    unit_registry = molecule.geometry._REGISTRY

    if batch_size is None:
        n_positions = 1
    else:
        n_positions = batch_size

    kwargs = dict(
        method=method,
        batch_positions=batch_positions,
        psi4_global_options=dict(basis='cc-pvtz', reference='RHF'),
        psi4_output_file_path='quiet'
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # The first call should create permanent restart files for the guess wavefunction.
        wavefunction_restart_file_paths = [os.path.join(temp_dir, '0_' + str(i) + '_wfn.npy')
                                           for i in range(n_positions)]

        forces, wavefunctions, energies = notorch_force_psi4(
            molecule=molecule,
            wavefunction_restart_file_paths=wavefunction_restart_file_paths,
            return_wavefunction=True,
            return_potential=True,
            **kwargs
        )

        # Check that forces computed from reference wavefunction are identical.
        forces2, energies2 = notorch_force_psi4(
            molecule=wavefunctions,
            unit_registry=unit_registry,
            return_potential=True,
            **kwargs
        )

        # Check that forces computed from the restart files are identical.
        forces3, energies3 = notorch_force_psi4(
            molecule=molecule,
            wavefunction_restart_file_paths=wavefunction_restart_file_paths,
            unit_registry=unit_registry,
            return_potential=True,
            **kwargs
        )

    # The shape should be identical to the positions.
    if batch_size is None:
        assert forces.shape == molecule.geometry.shape
    else:
        assert forces.shape == batch_positions.shape

    # Check equivalence of all calls above.
    for f, e in [(forces2, energies2), (forces3, energies3)]:
        assert np.allclose(forces.magnitude, f.magnitude)
        assert np.allclose(energies.magnitude, e.magnitude)

    # Check that the energies returned by the gradient
    # are identical to those returned by the potential.
    energies3 = notorch_potential_energy_psi4(molecule=molecule, **kwargs)
    assert np.allclose(energies.magnitude, energies3.magnitude)

    # By default, Psi4 orient the molecule so that the gradient along
    # the z-axis is 0 for 3 atoms so here we check that we remove this.
    assert np.all(forces.magnitude != 0)


@pytest.mark.parametrize('method', ['scf', 'mp2'])
def test_potential_energy_psi4_gradcheck(method):
    """Test that potential_energy_psi4 implements the correct gradient."""
    batch_size = 2
    molecule, batch_positions = create_water_molecule(batch_size)
    unit_registry = batch_positions._REGISTRY

    # Convert to tensor.
    batch_positions = np.reshape(batch_positions.to('angstrom').magnitude,
                                 (batch_size, molecule.n_atoms*3))
    batch_positions = torch.tensor(batch_positions, requires_grad=True, dtype=torch.double)

    # Use restart files to speedup gradcheck.
    with tempfile.TemporaryDirectory() as temp_dir:
        # The first call should create permanent restart files for the guess wavefunction.
        wavefunction_restart_file_paths = [os.path.join(temp_dir, '0_' + str(i) + '_wfn.npy')
                                           for i in range(batch_size)]

        torch.autograd.gradcheck(
            func=potential_energy_psi4,
            inputs=[
                batch_positions,
                molecule,
                method,
                unit_registry.angstrom,
                unit_registry.kJ / unit_registry.mol,
                wavefunction_restart_file_paths,
                dict(basis='cc-pvtz', reference='RHF'),
                None,
                None,
                1,
                'quiet',
                False
            ]
        )


@pytest.mark.parametrize('method', ['scf', 'mp2'])
def test_multiprocessing_torch_energy_gradient(method):
    """Test that the multiprocessing implementation is equivalent to the native intra-op implementation."""
    import psi4

    batch_size = 3
    molecule, batch_positions = create_water_molecule(batch_size)
    unit_registry = batch_positions._REGISTRY

    # Use this to restore the original number of Psi4 threads later.
    original_n_threads = psi4.core.get_num_threads()

    # Convert to tensor.
    batch_positions = np.reshape(batch_positions.to('angstrom').magnitude,
                                 (batch_size, molecule.n_atoms*3))
    batch_positions = torch.tensor(batch_positions, requires_grad=True, dtype=torch.double)

    kwargs = dict(
        batch_positions=batch_positions,
        molecule=molecule,
        method=method,
        positions_unit=unit_registry.angstrom,
        energy_unit=unit_registry.kJ / unit_registry.mol,
        psi4_global_options=dict(basis='cc-pvtz', reference='RHF'),
        psi4_output_file_path='quiet'
    )

    # Run with psi4 intra-op parallelization.
    energies_intra_op = potential_energy_psi4(
        n_threads=batch_size,
        processes=1,
        **kwargs
    )
    loss = torch.sum(energies_intra_op)
    loss.backward()
    forces_intra_op = batch_positions.grad

    # Reset gradients for next computation.
    batch_positions.grad.detach_()
    batch_positions.grad.data.zero_()

    # Next, use the multiprocessing implementation.
    energies_multiprocessing = potential_energy_psi4(
        n_threads=1,
        processes=batch_size,
        **kwargs
    )
    loss = torch.sum(energies_multiprocessing)
    loss.backward()
    forces_multiprocessing = batch_positions.grad

    assert torch.allclose(energies_intra_op, energies_multiprocessing)
    assert torch.allclose(forces_intra_op, forces_multiprocessing)

    # Restore the number of threads.
    psi4.core.set_num_threads(original_n_threads)
