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

import contextlib
import os
import shutil
import tempfile
import textwrap

import numpy as np
import pint
import pytest
import torch

from ..functions.qm.cpmd import (are_script_length_units_angstrom, set_script_coordinates,
                                 run_cpmd_script, srun_cpmd_script, potential_energy_cpmd)


# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

# Path where to store pseudopotentials for CPMD.
# TODO: ADD PSEUDOPOTENTIAL TO REPO FOR TESTING.
PP_DIR_PATH = '/p/project/cias-5/ippoliti/PROGRAMS/ARCHIVE/CPMD/PP'

# Unit registry for Pint units.
UNIT_REGISTRY = pint.UnitRegistry()
ANGSTROM = UNIT_REGISTRY.angstrom
NANOMETER = UNIT_REGISTRY.nanometer
BOHR = UNIT_REGISTRY.bohr
HARTREE = UNIT_REGISTRY.hartree

# Various CPMD scripts used for testing.
# SCRIPT_NO_ATOMS = """
# &CPMD
#  OPTIMIZE WAVEFUNCTION
#  CENTER MOLECULE OFF
#  PRINT FORCES ON
#  CONVERGENCE ORBITALS
#   1.0d-5
# &END
#
# &SYSTEM
#  SYMMETRY
#   1
#  ANGSTROM
#  CELL
#   8.00 1.0 1.0  0.0  0.0  0.0
#  CUTOFF
#   70.0
# &END
#
# &DFT
#  FUNCTIONAL LDA
# &END
#
# {atoms}
# """

SCRIPT_NO_ATOMS = """
&CPMD
 MOLECULAR DYNAMICS BO
 CENTER MOLECULE OFF
 TRAJECTORY SAMPLE FORCES
  1
 PRINT FORCES ON
 CONVERGENCE ORBITALS
  1.0d-5
 MAXSTEP

&END

&SYSTEM
 SYMMETRY
  1
 ANGSTROM
 CELL
  8.00 1.0 1.0  0.0  0.0  0.0
 CUTOFF
  70.0
&END

&DFT
 FUNCTIONAL LDA
&END

{atoms}
"""

SCRIPT_H2_ATOMS = """
&ATOMS
*H_MT_LDA.psp
 LMAX=S
  2
 4.371   4.000   4.000
 3.629   4.000   4.000
&END
"""

SCRIPT_H2_ATOMS_2 = """
&ATOMS
*H_MT_LDA.psp
 LMAX=S
  2
 0.2   0.3   0.4
 1.2   1.3   1.4
&END
"""

SCRIPT_CH3Cl_ATOMS = """
&ATOMS

*C_MT_BLYP.psp KLEINMAN-BYLANDER
 LMAX=D
  1
 23.149143360105   21.580670789584   22.733403642617

*Cl_MT_BLYP.psp KLEINMAN-BYLANDER
 LMAX=D
  1
 25.511300845830   19.483074942260   24.018417314852

*H_MT_BLYP.psp KLEINMAN-BYLANDER
 LMAX=P
  3
 21.297211891297   20.654705055179   22.752300902503
 23.905033755537   21.958615987300   20.824780394152
 23.073554320562   23.262526919420   23.923931015423

&END
"""

SCRIPT_H2 = SCRIPT_NO_ATOMS.format(atoms=SCRIPT_H2_ATOMS)

SCRIPT_CH3Cl = SCRIPT_NO_ATOMS.format(atoms=SCRIPT_CH3Cl_ATOMS)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# TODO: USE ACTUAL TEMPORARY DIRECTORY.
def get_tmp_dir(dir_path='tmp'):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    return dir_path


def determine_cpmd_output_paths(n_folders=1):
    """Create working directory and output file paths for CPMD."""
    if n_folders == 1:
        working_dir_path = get_tmp_dir()
        stdout_file_path = os.path.join(working_dir_path, 'cpmd.out')
    else:
        working_dir_path = [get_tmp_dir('tmp' + str(i)) for i in range(n_folders)]
        stdout_file_path = [os.path.join(d, 'cpmd.out') for d in working_dir_path]

    return working_dir_path, stdout_file_path


@contextlib.contextmanager
def created_script_file(script_str):
    """Create a temporary script file from its string representation."""
    script_file = tempfile.NamedTemporaryFile('w', delete=False)
    try:
        script_file.write(script_str)
        script_file.close()
        yield script_file.name
    finally:
        os.remove(script_file.name)


# =============================================================================
# TEST FOR THE HELPER FUNCTIONS
# =============================================================================

@pytest.mark.parametrize('script_str,are_units_angstrom', [
    (SCRIPT_H2, True),
    (SCRIPT_H2.replace('ANGSTROM', ''), False)
])
def test_are_script_length_units_angstrom(script_str, are_units_angstrom):
    """Test the method are_script_length_units_angstrom."""
    f = tempfile.NamedTemporaryFile('w', delete=False)
    try:
        f.write(script_str)
        f.close()

        assert are_script_length_units_angstrom(f.name) is are_units_angstrom
    finally:
        os.remove(f.name)


@pytest.mark.parametrize('expected_output,input_atoms_section,positions', [
    (SCRIPT_H2_ATOMS_2, SCRIPT_H2_ATOMS, [[0.2, 0.3, 0.4], [1.2, 1.3, 1.4]] * ANGSTROM),
    (SCRIPT_H2_ATOMS_2, SCRIPT_H2_ATOMS, [[0.02, 0.03, 0.04], [0.12, 0.13, 0.14]] * NANOMETER),
    (SCRIPT_H2_ATOMS_2, SCRIPT_H2_ATOMS,
     [[0.3779452249244559, 0.5669178373866838, 0.7558904498489118], [2.267671349546735, 2.4566439620089633, 2.645616574471191]] * BOHR),
    (SCRIPT_H2_ATOMS_2, SCRIPT_H2_ATOMS, [[0.2, 0.3, 0.4], [1.2, 1.3, 1.4]]),
    (
        textwrap.dedent("""
        &ATOMS

        *C_MT_BLYP.psp KLEINMAN-BYLANDER
         LMAX=D
          1
         0.2   0.3   0.4

        *Cl_MT_BLYP.psp KLEINMAN-BYLANDER
         LMAX=D
          1
         1.2   1.3   1.4

        *H_MT_BLYP.psp KLEINMAN-BYLANDER
         LMAX=P
          3
         2.2   2.3   2.4
         3.2   3.3   3.4
         4.2   4.3   4.4

        &END
        """),
        SCRIPT_CH3Cl_ATOMS,
        [[0.2, 0.3, 0.4], [1.2, 1.3, 1.4], [2.2, 2.3, 2.4], [3.2, 3.3, 3.4], [4.2, 4.3, 4.4]] * ANGSTROM
    ),
    (ValueError, SCRIPT_H2_ATOMS, [[0.2, 0.3, 0.4], [1.2, 1.3, 1.4], [2.1, 2.2, 2.3]] * ANGSTROM),
    (ValueError, SCRIPT_H2_ATOMS, [[0.2, 0.3, 0.4]] * ANGSTROM),
    (ValueError, SCRIPT_CH3Cl_ATOMS,
     [[0.2, 0.3, 0.4], [1.2, 1.3, 1.4], [2.2, 2.3, 2.4], [3.2, 3.3, 3.4], [4.2, 4.3, 4.4], [5.2, 5.3, 5.4]] * ANGSTROM),
    (ValueError, SCRIPT_CH3Cl_ATOMS,
     [[0.2, 0.3, 0.4], [1.2, 1.3, 1.4], [2.2, 2.3, 2.4], [3.2, 3.3, 3.4]] * ANGSTROM)
])
def test_set_script_coordinates(expected_output, input_atoms_section, positions):
    """Test that the coordinates are set correctly in the CPMD script."""
    # Create temporary directory.
    # TODO: USE ACTUAL TEMPORARY DIRECTORY.
    tmp_dir_path = get_tmp_dir()

    # Create input script.
    script_file_path = os.path.join(tmp_dir_path, 'script.in')
    with open(script_file_path, 'w') as f:
        f.write(SCRIPT_NO_ATOMS.format(atoms=input_atoms_section))

    if not isinstance(expected_output, str):
        with pytest.raises(expected_output):
            set_script_coordinates(script_file_path, positions)
    else:
        set_script_coordinates(script_file_path, positions, precision=10)

        # Read the atoms section.
        with open(script_file_path, 'r') as f:
            script = f.read()

        assert script == SCRIPT_NO_ATOMS.format(atoms=expected_output)


# =============================================================================
# TEST FOR THE CPMD WRAPPER
# =============================================================================

# Each test case is a tuple (batch_positions, expected_potentials, expected_gradients).
_TEST_CASES = [
    (
        None,
        -1.13245953 * HARTREE,
        np.array([
            [ 1.780e-02, -1.782e-16, -1.194e-16],
            [-1.780e-02, -2.118e-16, -1.830e-16]
        ]) * HARTREE/BOHR
    ),
    (
        [
            [
                [8.26190877673025, 7.55890449913159, 7.55890449913159],
                [6.85590022153292, 7.55890449913159, 7.55890449913159]
            ], [
                [8.26734304296695, 7.55890449913159, 7.55890449913159],
                [6.85046595529623, 7.55890449913159, 7.55890449913159]
            ]
        ] * BOHR,
        np.array([-1.1325249211, -1.1326803816]) * HARTREE,
        np.array([
            [
                [ 0.01632574430106, 0.0, 0.0],
                [-0.01632574430106, 0.0, 0.0],
            ], [
                [ 0.01229412232850, 0.0, 0.0],
                [-0.01229412232850, 0.0, 0.0]
            ]
        ]) * HARTREE/BOHR
    ),
]

def check_cpmd_run(run_func, batch_positions, expected_potentials, expected_gradients, n_processes):
    """Helper function to run and check potentials and gradients computed with CPMD."""
    # Create the paths to the working dirs and output.
    if batch_positions is None:
        working_dir_path, stdout_file_path = determine_cpmd_output_paths(n_folders=1)
    else:
        working_dir_path, stdout_file_path = determine_cpmd_output_paths(n_folders=len(batch_positions))

    # Create input script.
    with created_script_file(SCRIPT_H2) as script_file_path:
        potentials, gradients = run_func(
            input_script_file_path=script_file_path,
            pp_dir_path=PP_DIR_PATH,
            return_final_gradient=True,
            batch_positions=batch_positions,
            working_dir_path=working_dir_path,
            stdout_file_path=stdout_file_path,
            unit_registry=UNIT_REGISTRY,
            n_processes=n_processes
        )

    assert np.allclose(potentials.magnitude, expected_potentials.magnitude)
    # TODO: FIX TOLERANCE ONCE YOU FIND A WAY TO READ FINAL GRADIENT WITH LARGER PRECISION.
    assert np.allclose(gradients.magnitude, expected_gradients.magnitude, rtol=1.e-3)


@pytest.mark.skipif(shutil.which('cpmd.x') is None, reason='Requires cpmd.x in PATH')
@pytest.mark.parametrize('batch_positions,expected_potentials,expected_gradients', _TEST_CASES)
@pytest.mark.parametrize('parallel', [False, True])
def test_run_cpmd_script_batch_positions(batch_positions, expected_potentials, expected_gradients, parallel):
    """Test that potentials and gradients computed with CPMD are correct.

    The expected values have been verified through an independent run of CPMD.

    """
    n_processes = 1
    if parallel and batch_positions is not None:
        n_processes = len(batch_positions)
    check_cpmd_run(run_cpmd_script, batch_positions, expected_potentials, expected_gradients, n_processes)


@pytest.mark.skipif('SLURM_JOB_NUM_NODES' not in os.environ, reason='Requires running as a SLURM job')
@pytest.mark.skipif(shutil.which('cpmd.x') is None, reason='Requires cpmd.x in PATH')
@pytest.mark.parametrize('n_processes', [1, 2])
def test_srun_cpmd_script_batch_positions(n_processes):
    """Test running CPMD with srun command in SLURM settings."""
    #Test case with batch_size=2 with H2 molecule.
    test_case_idx = 1

    check_cpmd_run(
        srun_cpmd_script,
        batch_positions=_TEST_CASES[test_case_idx][0],
        expected_potentials=_TEST_CASES[test_case_idx][1],
        expected_gradients=_TEST_CASES[test_case_idx][2],
        n_processes=n_processes
    )


@pytest.mark.skipif(shutil.which('cpmd.x') is None, reason='Requires cpmd.x in PATH')
def test_potential_energy_cpmd_gradcheck():
    """Test that potential_energy_psi4 implements the correct gradient."""
    #Test case with batch_size=2 with H2 molecule.
    test_case_idx = 1

    # Create batch positions input tensor in the appropriate shape.
    batch_positions = _TEST_CASES[test_case_idx][0].magnitude
    batch_size, n_atoms = batch_positions.shape[:2]
    batch_positions = batch_positions.reshape((batch_size, n_atoms*3))
    batch_positions = torch.tensor(batch_positions, requires_grad=True)

    position_unit = _TEST_CASES[test_case_idx][0].units
    # working_dir_path, stdout_file_path = determine_cpmd_output_paths(n_folders=batch_size)
    working_dir_path, stdout_file_path = None, None

    with created_script_file(SCRIPT_H2) as script_file_path:
        torch.autograd.gradcheck(
            func=potential_energy_cpmd,
            inputs=[
                batch_positions,
                script_file_path,
                PP_DIR_PATH,
                position_unit,
                HARTREE,
                'cpmd.x',
                working_dir_path,
                stdout_file_path,
                batch_size,  # n_processes
                'auto'  # srun
            ]
        )
