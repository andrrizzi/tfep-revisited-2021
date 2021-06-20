#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Function to compute QM energies and gradients with Psi4.

Functions with a "notorch_" prefix are not compatible with PyTorch backpropagation.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import copy
import multiprocessing
import multiprocessing.pool
import os
import shutil
import tempfile

import numpy as np
import torch
import torch.autograd


# =============================================================================
# PSI4 WRAPPERS (NOT PYTORCH-COMPATIBLE)
# =============================================================================

def configure_psi4(
        memory=None,
        n_threads=None,
        psi4_output_file_path=None,
        psi4_scratch_dir_path=None
):
    """Configure some general parameters of Psi4.

    Parameters
    ----------
    memory : str, optional
        The memory available to psi4.
    n_threads : int, optional
        Number of MP threads used by psi4.
    psi4_output_file_path : str, optional
        Path to the output log file. If not given, the information is
        directed towards stdout. If the string "quiet", output is suppressed.
    psi4_scratch_dir_path : str, optional
        Path to the scratch directory.

    """
    import psi4

    if memory is not None:
        psi4.set_memory(memory)
    if n_threads is not None:
        psi4.core.set_num_threads(n_threads)

    # Set output file.
    if psi4_output_file_path == 'quiet':
        psi4.core.be_quiet()
    elif psi4_output_file_path is not None:
        psi4.core.set_output_file(psi4_output_file_path)

    # Scratch dir.
    if psi4_scratch_dir_path is not None:
        psi4_io = psi4.core.IOManager.shared_object()
        psi4_io.set_default_path(psi4_scratch_dir_path)


def notorch_potential_energy_psi4(*args, **kwargs):
    """Compute the potential energy of the molecule with Psi4.

    Parameters
    ----------
    method : str
        The level of theory to pass to ``psi4.energy()``.
    molecule : molecule.Molecule or psi4.core.Molecule
        The molecule object with the topology and geometry information.
    batch_positions : pint.Quantity, optional
        An array of positions. This can have the following shapes:
        (batch, n_atoms, 3), (batch, 3*n_atoms), (n_atoms, 3). If this
        is given, then the geometry in ``molecule`` can be ``None``.
    wavefunction_restart_file_paths : str or List[str], optional
        A list of file paths where to cache the converged SCF wavefunction.
        If a file is present at this path, it will be used to restart the
        calculation.
    return_wavefunction : bool, optional
        If ``True``, the wavefunction(s) is(are) also returned. Default
        is ``False``.
    unit_registry : pint.UnitRegistry, optional
        The unit registry to use for the force units. The function will
        attempt to automatically detect the unit registry to use for
        units from ``molecule`` and ``batch_positions``, but if this is
        not possible and this is not given, the forces will be returned
        as unit-less arrays.
    psi4_global_options : dict, optional
        Options to pass to ``psi4.set_options()``.
    memory : str, optional
        The memory available to psi4. By default the global Psi4 option
        is used.
    n_threads : int, optional
        Number of MP threads used by psi4. This is not compatible with
        batch parallelization (i.e., ``processes > 1``). By default
        the global Psi4 option is used.
    processes : int or multiprocessing.Pool, optional
        Number of processes to spawn or a pool of processes to be used
        to parallelize the calculation across batch samples. In the latter
        case, the default number of threads, memory, and scratch directory
        must be properly configured (although setting ``n_threads`` and
        ``memory`` in this function will work as expected). Parallelization
        over processes is not compatible with parallelization with Psi4
        threads (i.e. ``n_threads > 1``) or with ``molecule`` being different
        than a ``molecule.Molecule`` object. Default is 1.
    psi4_output_file_path : str, optional
        Path to the output log file. If not given, the information is
        directed towards stdout. If the string "quiet", output is suppressed.

    Returns
    -------
    potentials : pint.Quantity
        If ``batch_positions`` was not given, this is a single float with
        the energy in Hartrees of the molecule's geometry. Otherwise,
        ``potentials[i]`` is the potential of configuration ``batch_positions[i]``.
        This has units only if a suitable unit registry is found. Otherwise,
        it is a a unit-less array.
    wavefunctions : psi4.core.Wavefunction or List[psi4.core.Wavefunction], optional
        The wavefunction or, if ``batch_positions`` is given, a list of
        wavefunctions for each batch sample. This is returned only if
        ``return_wavefunction`` is ``True``.

    """
    import psi4
    return _run_func_psi4(psi4.energy, 'hartree', *args, **kwargs)


def notorch_force_psi4(*args, **kwargs):
    """Compute the potential energy of the molecule with Psi4.

    Parameters
    ----------
    method : str
        The level of theory to pass to ``psi4.gradient()``.
    molecule : molecule.Molecule, psi4.core.Molecule, psi4.core.Wavefunction, or List[psi4.core.Wavefunction]
        The molecule object with the topology and geometry information.
        If a ``Wavefunction`` object, this will be used as a reference
        wavefunction for the gradient calculation so that the SCF is
        skipped. The wavefunction passed to ``psi.gradient`` is that
        obtained from ``Wavefunction.reference_wavefunction()`` (if it
        is not ``None``). If a list of ``Wavefunction``s, this must have
        equal length to ``batch_positions``.
    batch_positions : pint.Quantity, optional
        An array of positions. This can have the following shapes:
        (batch, n_atoms, 3), (batch, 3*n_atoms), (n_atoms, 3). If this
        is given, then the geometry in ``molecule`` is ignored and can
        be ``None``.
    wavefunction_restart_file_paths : str or List[str], optional
        A list of file paths where to cache the converged SCF wavefunction.
        If a file is present at this path, it will be used to restart the
        calculation.
    return_wavefunction : bool, optional
        If ``True``, the wavefunction(s) is(are) also returned. Default
        is ``False``.
    return_potential : bool, optional
        If ``True``, the potential energy(ies) is(are) also returned.
        Default is ``False``.
    unit_registry : pint.UnitRegistry, optional
        The unit registry to use for the force units. The function will
        attempt to automatically detect the unit registry to use for
        units from ``molecule`` and ``batch_positions``, but if this is
        not possible and this is not given, the forces will be returned
        as unit-less arrays.
    psi4_global_options : dict, optional
        Options to pass to ``psi4.set_options()``.
    memory : str, optional
        The memory available to psi4. By default the global Psi4 option
        is used.
    n_threads : int, optional
        Number of MP threads used by psi4. This is not compatible with
        batch parallelization (i.e., ``processes > 1``). By default
        the global Psi4 option is used.
    processes : int or multiprocessing.Pool, optional
        Number of processes to spawn or a pool of processes to be used
        to parallelize the calculation across batch samples. In the latter
        case, the default number of threads, memory, and scratch directory
        must be properly configured (although setting ``n_threads`` and
        ``memory`` in this function will work as expected). Parallelization
        over processes is not compatible with parallelization with Psi4
        threads (i.e. ``n_threads > 1``) or with ``molecule`` being different
        than a ``molecule.Molecule`` object. Default is 1.
    psi4_output_file_path : str, optional
        Path to the output log file. If not given, the information is
        directed towards stdout. If the string "quiet", output is suppressed.

    Returns
    -------
    forces : pint.Quantity or numpy.ndarray
        If ``batch_positions`` was not given, this is a ``(n_atoms, 3)``
        array of forces in Hartrees/bohr for the molecule's geometry.
        Otherwise, ``forces[i]`` are the forces for the configuration
        ``batch_positions[i]``. This has units only if a suitable
        unit registry is found. Otherwise, it is a a unit-less array.
    wavefunctions : psi4.core.Wavefunction or List[psi4.core.Wavefunction], optional
        The wavefunction or, if ``batch_positions`` is given, a list of
        wavefunctions for each batch sample. This is returned only if
        ``return_wavefunction`` is ``True``.
    potentials : pint.Quantity or numpy.ndarray, optional
        If ``batch_positions`` was not given, this is a single float with
        the energy in Hartrees of the molecule's geometry. Otherwise,
        ``potentials[i]`` is the potential of configuration ``batch_positions[i]``.
        This has units only if a suitable unit registry is found. Otherwise,
        it is a a unit-less array. This is returned only if ``return_potential``
        is ``True``.

    """
    import psi4
    return _run_func_psi4(psi4.gradient, 'hartree/bohr', *args, **kwargs)


def _run_func_psi4(
    func_psi4,
    unit,
    method,
    molecule,
    batch_positions=None,
    wavefunction_restart_file_paths=None,
    return_wavefunction=False,
    return_potential=False,
    unit_registry=None,
    psi4_global_options=None,
    memory=None,
    n_threads=None,
    processes=1,
    psi4_output_file_path=None
):
    """Shared function to compute energies and forces with Psi4.

    This has the same exact arguments of ``notorch_force_psi4`` with the
    addition of ``unit``, which is used to add appropriate units to the
    main return value (force or energy).

    Returns the same values of ``notorch_force_psi4``.

    """
    import psi4
    from ..geometry import to_batch_atom_3_shape

    # Do we parallelize across batch samples with multiprocessing?
    run_with_multiprocessing = (
        (isinstance(processes, multiprocessing.pool.Pool) or processes > 1) and
        (batch_positions is not None) and
        (len(batch_positions) > 1)
    )
    spawn_process_pool = run_with_multiprocessing and not isinstance(processes, multiprocessing.pool.Pool)

    # Handle mutable defaults.
    if psi4_global_options is None:
        psi4_global_options = {}

    # Check if we need to compute the potential for multiple positions.
    if batch_positions is None:
        batch_size = 1
    else:
        # We have access to a unit registry in this case.
        if unit_registry is None:
            unit_registry = batch_positions._REGISTRY

        # Convert to standard positions shape.
        batch_positions = to_batch_atom_3_shape(batch_positions)
        batch_size = batch_positions.shape[0]

        # The psi4.core.Molecule.set_geometry() function wants the
        # coordinates in units of Bohr. We need this only in serial execution.
        if not run_with_multiprocessing:
            batch_positions_unitless = batch_positions.to('bohr').magnitude

    # If a single wavefunction is passed, encapsulate it in a list
    # so that the next block of code takes care of both cases.
    if isinstance(molecule, psi4.core.Wavefunction):
        molecule = [molecule]

    # Create the Psi4 molecule.
    if isinstance(molecule, list):
        # This is a list of Wavefunction objects. One for each batch sample.
        if ((batch_positions is not None) and
                (len(batch_positions) != len(molecule))):
            raise ValueError('The number of positions must be equal to '
                             'the number of reference wavefunctions.')
        if run_with_multiprocessing:
            raise ValueError('Cannot run with multiprocessing starting '
                             'from a reference wavefunction.')

        # The ref_wfn argument must be the SCF wavefunction
        reference_wavefunctions = []
        for wfn in molecule:
            ref_wfn = wfn.reference_wavefunction()
            if ref_wfn is None:
                reference_wavefunctions.append(wfn)
            else:
                reference_wavefunctions.append(ref_wfn)

        molecule = reference_wavefunctions[0].molecule()
    else:
        reference_wavefunctions = None

    # Check that a correct number of wavefunction restart files was passed.
    if wavefunction_restart_file_paths is not None:
        if reference_wavefunctions is not None:
            raise ValueError('Cannot pass both reference wavefunctions and restart file paths.')

        if isinstance(wavefunction_restart_file_paths, str):
            wavefunction_restart_file_paths = [wavefunction_restart_file_paths]

        if len(wavefunction_restart_file_paths) != batch_size:
            raise ValueError('Incorrect number of wavefunction restart file path passed.')

    # Make sure the molecule is activated and that its geometry
    # correspond to the coordinate of the first batch sample.
    if isinstance(molecule, psi4.core.Molecule):
        if run_with_multiprocessing:
            raise ValueError('Multiprocessing is not supported with psi4.core.Molecule.')

        psi4.core.set_active_molecule(molecule)
        if batch_positions is not None:
            molecule.set_geometry(psi4.core.Matrix.from_array(batch_positions_unitless[0]))
            molecule.update_geometry()
    else:
        # This is our Molecule object. We have access to a unit registry here.
        if unit_registry is None:
            unit_registry = molecule.geometry._REGISTRY

        if batch_positions is not None:
            # Create a shallow copy of the molecule to avoid modifying the input.
            molecule = copy.copy(molecule)
            molecule.geometry = batch_positions[0]

        # We create the Psi4 molecule only once if we don't need
        # to pickle it and send it to the parallel processes.
        if not run_with_multiprocessing:
            # This is our Molecule class. We never reorient/translate so
            # that the forces are not in a different reference frame.
            molecule = molecule.to_psi4(reorient=False, translate=False)
            psi4.core.set_active_molecule(molecule)

    # Check options, number of threads/memory to pass to _func_psi4_wrapper.
    if spawn_process_pool:
        # Copy the number of threads, memory, and scratch dir of the main
        # process so that we can send them to the parallel processes.
        if memory is None:
            memory = psi4.get_memory()
        if n_threads is None:
            n_threads = psi4.core.get_num_threads()

        psi4_io = psi4.core.IOManager.shared_object()
        scratch_dir_path = psi4_io.get_default_path()
    elif not run_with_multiprocessing:
        # The function won't use parallel processes so we can avoid
        # setting this multiple times in _func_psi4_wrapper.
        configure_psi4(memory, n_threads, psi4_output_file_path)
        memory, n_threads, scratch_dir_path, psi4_output_file_path = None, None, None, None
        psi4.set_options(psi4_global_options)
        psi4_global_options = None
    else:
        # Use the pool of processes passed as arguments. We assume here
        # that the default memory, n_threads, and scratch are configured.
        scratch_dir_path = None

    # Compile a list of kwargs to pass to _func_psi4_wrapper for each batch sample.
    # We work in a separate temporary scratch if we run with multiple processes.
    create_tmp_scratch = run_with_multiprocessing
    batch_func_psi4_args = [
        [
            func_psi4, method, molecule, return_wavefunction, return_potential,
            create_tmp_scratch, psi4_global_options, memory, n_threads, scratch_dir_path, psi4_output_file_path,
            None, None, None  # Position, reference wfn and restart files are filled below.
         ]
        for _ in range(batch_size)
    ]

    # Add batch positions to args. The first execution uses the positions
    # of molecule, which correspond to batch_positions[0] at this point.
    if batch_positions is not None:
        assigned_positions = batch_positions if run_with_multiprocessing else batch_positions_unitless
        for batch_idx in range(1, batch_size):
            batch_func_psi4_args[batch_idx][-3] = assigned_positions[batch_idx]

    # Add reference wavefunctions to args.
    if reference_wavefunctions is not None:
        for batch_idx in range(1, batch_size):
            batch_func_psi4_args[batch_idx][-2] = reference_wavefunctions[batch_idx]

    # Add cached wavefunction paths to restart files.
    if wavefunction_restart_file_paths is not None:
        for batch_idx in range(batch_size):
            batch_func_psi4_args[batch_idx][-1] = wavefunction_restart_file_paths[batch_idx]

    # Run the psi4 function parallelizing across the batch samples with processes.
    if spawn_process_pool:
        # The default "fork" method doesn't play well with Psi4 threads.
        try:
            mp_context = multiprocessing.get_context('forkserver')
        except ValueError:
            mp_context = multiprocessing.get_context('spawn')

        # Parallelize batch over a pool of processes.
        with mp_context.Pool(processes) as process_pool:
            all_results = process_pool.starmap(_func_psi4_wrapper, batch_func_psi4_args)
    elif run_with_multiprocessing:
        # processes is a multiprocessing.Pool instance.
        all_results = processes.starmap(_func_psi4_wrapper, batch_func_psi4_args)
    else:
        # Run serially using only Psi4's intra-op parallelization.
        all_results = [None for _ in range(batch_size)]
        for batch_idx, batch_args in enumerate(batch_func_psi4_args):
            all_results[batch_idx] = _func_psi4_wrapper(*batch_args)

    # Separate the wavefunctions from energy/gradients and convert to array.
    return_optional_result = return_wavefunction or return_potential
    if return_optional_result:
        # zip() returns tuples rather than lists.
        all_results = [list(x) for x in zip(*all_results)]
        if return_wavefunction:
            all_wavefunctions = all_results[1]
        if return_potential:
            all_potentials = np.array(all_results[-1])
        all_results = all_results[0]
    all_results = np.array(all_results)

    # Add units using the same registry of the positions.
    if unit_registry is not None:
        all_results *= unit_registry.parse_expression(unit)
        if return_potential:
            all_potentials *= unit_registry.hartree

    # If there is no batch, we simply return the potential.
    if batch_positions is None:
        all_results = all_results[0]
        if return_wavefunction:
            all_wavefunctions = all_wavefunctions[0]
        if return_potential:
            all_potentials = all_potentials[0]

    # Return requested values.
    if not return_optional_result:
        return all_results

    returned_values = [all_results]
    if return_wavefunction:
        returned_values.append(all_wavefunctions)
    if return_potential:
        returned_values.append(all_potentials)
    return returned_values


def _redirect_psi4_file_180(file_path):
    """Wrap Wavefunction.get_scratch_filename to redirect file 180.

    For some reason, IOManager.set_specific_path(180) does not work.
    Instead, Psi4 uses a custom path (see also psi4#918). The function
    is dynamically assigned to core.Wavefunction in psi4.driver.p4util.python_helpers.
    """
    def _wrapper_func(self, filenumber):
        if filenumber == 180:
            return os.path.splitext(file_path)[0]
        return self.get_scratch_filename(filenumber)
    return _wrapper_func


def _func_psi4_wrapper(
        func_psi4,
        method,
        molecule,
        return_wavefunction,
        return_potential,
        create_tmp_scratch,
        psi4_global_options,
        memory,
        n_threads,
        scratch_dir_path,
        psi4_output_file_path,
        positions,
        reference_wavefunction,
        wavefunction_restart_file_path
):
    """Helper function to handle parallelization in _run_func_psi4.

    This can be passed to a process for multiprocessing parallelization.
    Arguments and return values are largely those of _run_func_psi4 with
    these exceptions.

    Parameters
    ----------
    create_tmp_scratch : bool
        Creates a temporary scratch space for Psi4 for the duration of
        this task. With multiprocessing, it's best to set this to ``True``
        as there seem to be some conflict with the scratch space with
        multiple concurrent executions of Psi4, and a single scratch
        space can run out of space very quickly.
    scratch_dir_path : str, optional
        The main scratch directory. This will be used as prefix for tmp_scratch
        if it is True. It might be useful when starting psi4 in a subprocess,
        otherwise the scratch_dir_path is read automatically from the API.
    positions : None or pint.Quantity or numpy.ndarray
        The position of a single configuration (i.e. no batch). If ``None``,
        the coordinates in ``molecule`` are used. If a unitless array,
        these must be in units of Bohr radius.
    reference_wavefunction : None or psi4.core.Wavefunction
        The reference wavefunction of a single configuration (i.e. no batch)
        if available.

    """
    import psi4
    from psi4.driver.p4util import OptionsState
    from psi4.driver.p4util.python_helpers import _core_wavefunction_get_scratch_filename

    return_optional_result = return_wavefunction or return_potential

    # Save the options that we'll restore at the end of the function.
    if psi4_global_options is not None:
        option_stash = OptionsState(*[[x] for x in psi4_global_options.keys()])
    else:
        option_stash = OptionsState()

    # If requested, we execute this try block in a temporary scratch.
    psi4_io = psi4.core.IOManager.shared_object()
    tmp_dir = None
    original_scratch_dir_path = None
    try:
        # Store the original scratch dir that we'll have to restore.
        if create_tmp_scratch or (scratch_dir_path is not None):
            original_scratch_dir_path = psi4_io.get_default_path()

        # Set new scratch dir.
        if create_tmp_scratch:
            if scratch_dir_path is None:
                tmp_dir_prefix = os.path.join(original_scratch_dir_path, 'tmp')
            else:
                tmp_dir_prefix = os.path.join(scratch_dir_path, 'tmp')
            tmp_dir = tempfile.mkdtemp(prefix=tmp_dir_prefix)

            psi4_io.set_default_path(tmp_dir)
        elif scratch_dir_path is not None:
            psi4_io.set_default_path(scratch_dir_path)

        # Configure Psi4.
        configure_psi4(memory, n_threads, psi4_output_file_path)
        if psi4_global_options is not None:
            psi4.set_options(psi4_global_options)

        # Create psi4 molecule making sure the positions are correct.
        if not isinstance(molecule, psi4.core.Molecule):
            # This is our Molecule class.
            if positions is not None:
                molecule.geometry = positions

            # We never reorient/translate so that the forces are not in
            # a different reference frame.
            molecule = molecule.to_psi4(reorient=False, translate=False)
        elif positions is not None:
            molecule.set_geometry(psi4.core.Matrix.from_array(positions))
            molecule.update_geometry()

        psi4.core.set_active_molecule(molecule)

        # Configure cached/reference wavefunction options.
        kwargs = {}
        if reference_wavefunction is not None:
            kwargs['ref_wfn'] =reference_wavefunction
        if wavefunction_restart_file_path is not None:
            # Make sure psi4 will start from the file (if there is one)
            # and updates it at the end of the calculation.
            kwargs['write_orbitals'] = True
            option_stash.add_option(['SCF', 'GUESS'])
            psi4.core.set_local_option('SCF', 'GUESS', 'READ')

            # At the moment we can't control the specific path of file 180,
            # so we dynamically wrap the function used to retrive the path.
            # We reassign the old function before returning.
            assert psi4.core.Wavefunction.get_scratch_filename is _core_wavefunction_get_scratch_filename
            psi4.core.Wavefunction.get_scratch_filename = _redirect_psi4_file_180(wavefunction_restart_file_path)

        # Run function.
        result = func_psi4(
            method,
            molecule=molecule,
            return_wfn=return_optional_result,
            **kwargs
        )

        if return_optional_result:
            wavefunction = result[1]
            result = result[0]

        # If func_psi4 is gradient/energy, the result is a matrix/float.
        try:
            result = result.to_array()
        except AttributeError:
            pass # Result is a float.
    except Exception as e:
        # Psi4 raises exceptions containing objects that can be pickled and
        # the program crashes when this is called with multiprocessing.
        raise RuntimeError(str(e))
    finally:
        if wavefunction_restart_file_path is not None:
            # Restore the original file path for file 180.
            psi4.core.Wavefunction.get_scratch_filename = _core_wavefunction_get_scratch_filename

            # Make sure the wavefunction file is not removed from disk. We
            # expect the file towards the end as it is appended on registration.
            # Convert to real path to make sure the comparison is done correctly.
            wavefunction_restart_file_path = os.path.realpath(wavefunction_restart_file_path)
            for npy_file_idx in range(len(psi4.extras.numpy_files)-1, -1, -1):
                if wavefunction_restart_file_path == os.path.realpath(psi4.extras.numpy_files[npy_file_idx]):
                    del psi4.extras.numpy_files[npy_file_idx]
                    break

        # Restore the original scratch space and options.
        if tmp_dir is not None:
            psi4.core.clean()
            shutil.rmtree(tmp_dir)
        if original_scratch_dir_path is not None:
            psi4_io.set_default_path(original_scratch_dir_path)
        option_stash.restore()

    # Check what we have to return.
    if not return_optional_result:
        return result

    returned_values = [result]
    if return_wavefunction:
        returned_values.append(wavefunction)
    if return_potential:
        returned_values.append(wavefunction.energy())
    return returned_values


# =============================================================================
# PyTorch COMPATIBLE FUNCTIONS
# =============================================================================

class PotentialEnergyPsi4(torch.autograd.Function):
    """PyTorch-compatible calculation of the potential and gradient of a molecule with Psi4."""

    @staticmethod
    def forward(
            ctx,
            batch_positions,
            molecule,
            method,
            positions_unit,
            energy_unit,
            wavefunction_restart_file_paths=None,
            psi4_global_options=None,
            memory=None,
            n_threads=None,
            processes=1,
            psi4_output_file_path=None,
            precompute_gradient=True
    ):
        """Compute the potential energy of the molecule with Psi4.

        For efficiency reasons, the function computes and cache the
        gradient (i.e., the forces) during the forward pass so that
        it can be used during backpropagation. If backpropagation is
        not necessary, set ``precompute_gradient`` to ``False``.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            A context to save information for the gradient.
        batch_positions : torch.Tensor
            A tensor of positions. This can have the following shapes:
            ``(batch_size, n_atoms, 3)``, ``(batch_size, 3*n_atoms)``.
        molecule : molecule.Molecule or psi4.core.Molecule
            The molecule object with the topology information. The geometry
            information can be ``None`` as the positions will be passed by
            the forward function.
        method : str
            The level of theory to pass to ``psi4.energy()``.
        positions_unit : pint.Unit
            The units used for the positions.
        energy_unit : pint.Unit
            The units used to represent the energies.
        wavefunction_restart_file_paths : str or List[str], optional
            A list of file paths where to cache the converged SCF wavefunction.
            If a file is present at this path, it will be used to restart the
            calculation.
        global_options_psi4 : dict, optional
            Options to pass to ``psi4.set_options()``. Default is ``None``.
        memory : str, optional
            The memory available to psi4. By default the global Psi4 option
            is used.
        n_threads : int, optional
            Number of MP threads used by psi4. This is not compatible with
            batch parallelization (i.e., ``processes > 1``). By default
            the global Psi4 option is used.
        processes : int or multiprocessing.Pool, optional
            Number of processes to spawn or a pool of processes to be used
            to parallelize the calculation across batch samples. In the latter
            case, the default number of threads, memory, and scratch directory
            must be properly configured (although setting ``n_threads`` and
            ``memory`` in this function will work as expected). Parallelization
            over processes is not compatible with parallelization with Psi4
            threads (i.e. ``n_threads > 1``) or with ``molecule`` being different
            than a ``molecule.Molecule`` object. Default is 1.
        output_file_path_psi4 : str, optional
            Path to the output log file. If not given, the information is
            directed towards stdout. If the string "quiet", output is suppressed.
            Default is ``None``.
        precompute_gradient : bool, optional
            If ``True``, the gradient is computed in the forward pass and
            saved to be consumed during backward. This gives better
            performance overall when backpropagation is necessary as the
            wavefunction is converged just once.

        Returns
        -------
        potentials : torch.Tensor
            ``potentials[i]`` is the potential energy of configuration
            ``batch_positions[i]``.

        """
        # Convert tensor to numpy array with units.
        batch_positions_arr = batch_positions.detach().numpy() * positions_unit

        if precompute_gradient:
            # The gradient computation already computes the potentials so
            # we do everything in a single pass and avoid re-doing the
            # MP2 wavefunction convergence twice.
            forces, potentials = notorch_force_psi4(
                method,
                molecule,
                batch_positions_arr,
                return_wavefunction=False,
                return_potential=True,
                psi4_global_options=psi4_global_options,
                memory=memory,
                n_threads=n_threads,
                processes=processes,
                psi4_output_file_path=psi4_output_file_path
            )

            # Save the variables used to compute the gradient in backpropagation.
            forces = PotentialEnergyPsi4._convert_forces_to_tensor(
                forces, energy_unit, positions_unit, dtype=batch_positions.dtype)
            ctx.save_for_backward(forces)
        else:
            # Compute the potential energies. Save the wavefunction so that
            # the gradient computation will avoid re-doing the SCF later.
            potentials, wavefunctions = notorch_potential_energy_psi4(
                method,
                molecule,
                batch_positions_arr,
                return_wavefunction=True,
                psi4_global_options=psi4_global_options,
                n_threads=n_threads,
                memory=memory,
                psi4_output_file_path=psi4_output_file_path
            )

            # Save the variables used to compute the gradient in backpropagation.
            ctx.wavefunctions = wavefunctions
            ctx.batch_positions_arr = batch_positions_arr
            ctx.method = method
            ctx.energy_unit = energy_unit
            ctx.positions_unit = positions_unit
            ctx.psi4_global_options = psi4_global_options
            ctx.n_threads = n_threads
            ctx.memory = memory
            ctx.psi4_output_file_path = psi4_output_file_path

        # Convert units. Psi4 returns the energy in Hartrees.
        try:
            # Convert to Hartree/mol.
            potentials = (potentials * energy_unit._REGISTRY.avogadro_constant).to(energy_unit)
        except TypeError:
            potentials = potentials.to(energy_unit)

        # Reconvert Pint array to tensor.
        return torch.tensor(potentials.magnitude, dtype=batch_positions.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # We still need to return a None gradient for each
        # input of forward() beside batch_positions.
        n_input_args = 12
        grad_input = [None for _ in range(n_input_args)]

        # Compute gradient w.r.t. batch_positions.
        if ctx.needs_input_grad[0]:
            # Check if we have already computed the forces.
            if len(ctx.saved_tensors) == 1:
                # Retrieve pre-computed forces.
                forces, = ctx.saved_tensors
            else:
                forces = notorch_force_psi4(
                    ctx.method,
                    ctx.wavefunctions,
                    ctx.batch_positions_arr,
                    psi4_global_options=ctx.psi4_global_options,
                    n_threads=ctx.n_threads,
                    memory=ctx.memory,
                    psi4_output_file_path=ctx.psi4_output_file_path
                )
                forces = PotentialEnergyPsi4._convert_forces_to_tensor(
                    forces, ctx.energy_unit, ctx.positions_unit, dtype=grad_output.dtype)

            # Accumulate gradient.
            grad_input[0] = forces * grad_output[:, None]

        return tuple(grad_input)

    @staticmethod
    def _convert_forces_to_tensor(forces, energy_unit, positions_unit, dtype):
        """Helper function to convert the Psi4 forces to a PyTorch tensor."""
        force_unit = energy_unit / positions_unit
        try:
            # Convert to Hartree/(Bohr mol).
            forces = (forces * force_unit._REGISTRY.avogadro_constant).to(force_unit)
        except TypeError:
            forces = forces.to(force_unit)

        # The tensor must be unitless and with shape (batch_size, n_atoms*3).
        forces = np.reshape(forces.magnitude, (forces.shape[0], forces.shape[1]*forces.shape[2]))
        return torch.tensor(forces, dtype=dtype)


def potential_energy_psi4(
        batch_positions,
        molecule,
        method,
        positions_unit,
        energy_unit,
        wavefunction_restart_file_paths=None,
        psi4_global_options=None,
        memory=None,
        n_threads=None,
        processes=1,
        psi4_output_file_path=None,
        precompute_gradient=True
):
    """Functional notation and keyword arguments for ``PotentialEnergyPsi4.apply``.

    See Also
    --------
    PotentialEnergyPsi4

    """
    # apply() does not accept keyword arguments.
    return PotentialEnergyPsi4.apply(
        batch_positions,
        molecule,
        method,
        positions_unit,
        energy_unit,
        wavefunction_restart_file_paths,
        psi4_global_options,
        memory,
        n_threads,
        processes,
        psi4_output_file_path,
        precompute_gradient
    )
