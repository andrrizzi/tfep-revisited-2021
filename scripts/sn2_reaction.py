#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
This script perform standard and targeted FEP analysis of the SN2 reaction.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import contextlib
import copy
import glob
import os

import numpy as np
import torch

from modules.nets.modules.flows import (AffineTransformer, SOSPolynomialTransformer,
                                        NeuralSplineTransformer, MobiusTransformer)
from modules import global_unit_registry
from modules.molecule import Molecule
from modules.reweighting import DatasetReweighting


# =============================================================================
# GLOBAL CONFIGURATIONS.
# =============================================================================

# Path to directories and files.
SCRIPT_DIR_PATH = os.path.realpath(os.path.dirname(__file__))

# Number of processes to use for QM.
N_PROCESSES = 1

# ----------------------------------#
# SN2 reaction simulation constants #
# ----------------------------------#

# The temperature at which the simulation is run.
TEMPERATURE = 300 * global_unit_registry.kelvin
KT = (TEMPERATURE * global_unit_registry.molar_gas_constant).to('kJ/mol')

# The path to the main output directory for the SN2 reaction experiment.
SN2_VACUUM_DIR_PATH = os.path.join(SCRIPT_DIR_PATH, '..', 'sn2_vacuum')
# The path to the topology file. This is used to save the mapped coordinates in PDB format.
SN2_VACUUM_PRMTOP_FILE_PATH =  os.path.join('..', 'system_files', 'prmtop_inpcrd_files', 'sn2_vacuum.prmtop')

# Dataset directory structure.
DATASET_INFO_SUBDIR = 'datasets_info'
STRIDE_PREFIX = 'stride_'
TRAIN_SUFFIX = '_train'
TEST_SUFFIX = '_test_'
INPUT_POTENTIALS_FILE_NAME = 'input_potentials.dat'
INPUT_COORDINATES_FILE_NAME = 'input_coordinates.pdb'

TRAINING_RESULTS_SUBDIR = 'training_results'
LOSS_FILE_NAME = 'loss.npy'
NET_FINAL_FILE_NAME = 'net_final.pth'
OPTIMIZER_CHECKPOINT_FILE_NAME = 'optimizer_checkpoint.pth'
NETS_SUBDIR = 'nets'
BATCH_DATA_SUBDIR = 'batch_data'
WAVEFUNCTIONS_SUBDIR = 'wavefunctions'

REWEIGHTING_RESULTS_SUBDIR = 'reweighting_results'
MP2_POTENTIALS_CACHE_FILE_NAME = 'mp2_potentials.npz'
DET_DCV_CACHE_FILE_NAME = 'det_dcv.npz'
DELTA_FES_FILE_NAME = 'delta_fes.dat'
BOOT_DELTA_FES_FILE_NAME = 'bootstrap_delta_fes.npz'
DELTA_F_FILE_NAME = 'delta_f.json'

# Number of checkpoints saved during training per epoch.
TRAINING_N_CHECKPOINTS_PER_EPOCH = 2
# Whether to shuffle the training dataset to train the neural network.
SHUFFLE = True
# Whether to drop the last batch if its shorter than TRAINING_BATCH_SIZE when training the neural network.
DROP_LAST = True
# The size of the batch size used for training.
TRAINING_BATCH_SIZE = 256
# Number of training epochs.
TRAINING_N_EPOCHS = 20
# The name of the optimizer used to train the normalizing flow.
OPTIMIZER_NAME = 'adam'
# The learning rate for the optimizer.
LEARNING_RATE = 1e-3
# If not None, cyclical learning rate is used and the learning rate goes
# from a minimum of CYCLICAL_LEARNING_RATE_MIN to a maximum of LEARNING_RATE
# in CYCLICAL_LEARNING_EPOCHS_PER_STEP epochs, and then it goes back to
# CYCLICAL_LEARNING_RATE_MIN in the same number of epochs.
CYCLICAL_LEARNING_RATE_MIN = None
CYCLICAL_LEARNING_EPOCHS_PER_STEP = 5
# If not None, the ReduceLROnPlateau PyTorch scheduler is used with this factor.
REDUCE_LR_ON_PLATEAU_FACTOR = None
# If not None, the MultiplicativeLR is implemented with factor MULTIPLICATIVE_LR_GAMMA at each epoch.
MULTIPLICATIVE_LR_GAMMA = None
# The weight decay constant.
WEIGHT_DECAY = 0.01

# The FES is computed only between these two CV bounds.
FES_CV_BOUNDS = (-2.0, 2.6)
# The number of bins used for the FES.
FES_GRID_N_BINS = 400
# The CV bounds used to define the CH3F metastable state.
BASIN1_CV_BOUNDS = (-1.8, 0.1)  # Used for TRP
# BASIN1_CV_BOUNDS = (-1.45, -0.35)  # Used for TFEP
# The CV bounds used to define the CH3Cl metastable state.
BASIN2_CV_BOUNDS = (0.6, 2.4)

# Whether to compute the Delta FES.
COMPUTE_DFES = True
# Whether to compute from the FES the difference in free energy
# and barriers between the metastable states.
COMPUTE_DF_BASINS_FROM_FES = True
# Compute the total free energy difference from the low to high
# level potential for each of these basins.
COMPUTE_DF_BASINS_FROM_H = {
    'basin1': BASIN1_CV_BOUNDS,
    'basin2': BASIN2_CV_BOUNDS,
}
# Number of cycles to compute bootstrap statistics.
N_BOOTSTRAP_CYCLES = 10000
# The size of the moving average window used to smooth out the FES.
SMOOTHING_WINDOW_SIZE = 11

# Used to analyze the reweighting for increasing sample size.
SUPPORTED_STRIDE_TIMES = [200000, 100000, 70000, 40000, 20000, 10000, 5000, 2500, 1000, 500] * global_unit_registry.fs
# SUPPORTED_STRIDE_TIMES = [200000, 100000, 70000, 40000, 20000, 10000, 5000, 2500, 1000, 500, 250, 100] * global_unit_registry.fs
# SUPPORTED_STRIDE_TIMES = [200000, 100000, 70000, 40000, 20000, 10000, 5000, 2500, 1000, 500, 250, 100, 50] * global_unit_registry.fs
# SUPPORTED_STRIDE_TIMES = [400000, 280000, 160000, 80000, 40000, 20000, 10000, 4000, 2000, 1000, 400, 250, 100, 50] * global_unit_registry.fs

# The dataset that is resampled in the bootstrap analysis subsamples the trajectory using this stride.
BOOTSTRAP_STRIDE_TIME = 500 * global_unit_registry.femtoseconds
SUBSAMPLE_SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]#, 100000, 200000]

# This molecule object encapsulate the topology of the simulated molecules.
# The geometry is taken from the input inpcrd.
SN2_VACUUM_MOLECULE = Molecule(
    geometry=np.array([
        [4.716, 4.973, 5.222],
        [5.811, 3.608, 5.530],
        [4.956, 5.231, 4.209],
        [4.987, 5.842, 5.745],
        [3.674, 4.692, 5.395],
        [3.741, 7.573, 4.154]]
    ) * global_unit_registry.angstrom,
    symbols=['C', 'Cl', 'H', 'H', 'H', 'F'],
    molecular_charge=-1,
    molecular_multiplicity=1,
)

# The keyword arguments to forward to potential_energy_psi4 for the reweighting.
SN2_VACUUM_PSI4_KWARGS = dict(
    method='mp2',
    molecule=SN2_VACUUM_MOLECULE,
    psi4_global_options=dict(
        basis='aug-cc-pvtz',               # Primary basis set.
        df_basis_scf='aug-cc-pvtz-jkfit',  # Auxiliary basis for density-fitting of the SCF calculation.
        df_basis_mp2='aug-cc-pvtz-ri',     # Auxiliary basis for density-fitting of the MP2 calculations.
        e_convergence=1.e-8,               # Energy convergence criterion.
        freeze_core='true',                # Freeze core electron for correlated computations (default is false).
        reference='RHF',                   # Restricted HF.
        scf_type='DF',                     # Use density-fitting (or RI) for the JK integrals in SCF and the MP2 integrals.
    )
)


# =============================================================================
# UTILITY FUNCTIONS TO HANDLE MULTIPROCESSING PARALLELIZATION
# =============================================================================

@contextlib.contextmanager
def create_global_process_pool(n_processes):
    """Context manager to create and terminate a pool of processes (if n_processes > 1)."""
    if n_processes > 1:
        from multiprocessing import get_context

        # The default "fork" method doesn't play well with Psi4 threads.
        try:
            mp_context = get_context('forkserver')
        except ValueError:
            mp_context = get_context('spawn')

        with mp_context.Pool(n_processes) as process_pool:
            yield process_pool
    else:
        yield None


# =============================================================================
# UTILITY FUNCTIONS TO HANDLE FILES
# =============================================================================

def load_metad_fes(fes_dir_path):
    """Load all the PLUMED free energy profiles saved by sumhills."""
    from modules.plumedwrapper import io as plumedio

    # Read the FES in time.
    fes_file_prefix_path = os.path.join(fes_dir_path, 'fes_')
    fes_time = plumedio.read_table(fes_file_prefix_path + 'time.dat')
    all_metad_fes = []
    for i in range(len(fes_time['time'])):
        all_metad_fes.append(plumedio.read_table(fes_file_prefix_path + str(i) + '.dat'))

    return all_metad_fes, fes_time


def epoch_batch_sort_key(file_path):
    """Pass to list.sort(key=) to order '0_batch_2_*' strings by epoch and batch."""
    file_name = os.path.basename(file_path)
    file_name = file_name.split('_')
    epoch_idx = int(file_name[0])
    try:
        batch_idx = int(file_name[2])
    except ValueError:
        batch_idx = int(os.path.splitext(file_name[2])[0])
    return (epoch_idx, batch_idx)


def read_all_plumed_rows(dataset, col_indices):
    """Read all the values of the plumed info in the dataset for the given column indices."""
    if isinstance(col_indices, int):
        col_indices = [col_indices]

    data = [np.empty(len(dataset)) for _ in col_indices]
    for sample_idx in range(len(dataset)):
        ts = dataset.get_ts(sample_idx)
        for data_idx, col_idx in enumerate(col_indices):
            data[data_idx][sample_idx] = ts.aux['plumed'][col_idx]

    if len(data) == 1:
        return data[0]
    return data


def create_datasets(
        trajectory_dataset,
        equilibration_time,
        train_stride_time,
        randomize_training_dataset=False,
        cv_range=None,
        return_eval_datasets=True,
        return_bootstrap_datasets=False
):
    """Split dataset in a train and increasingly larger test datasets.

    Multiple test datasets of increasing size are returned to evaluate
    the asymptotic behavior of the estimator. The train do not overlap
    with the test datasets, but the test datasets overlap with each
    other.

    Parameters
    ----------
    trajectory_dataset : TrajectoryDataset
        The trajectory dataset.
    equilibration_time : pint.Quantity
        The initial time of the simulation to discard.
    train_stride_time : pint.Quantity
        The stride time to use to subsample the trajectory
    randomize_training_dataset : bool, optional
        If True, a training dataset of random samples will be generated.
        The total number of samples would be the same as that generated
        by subsampling the trajectory with stride time ``train_stride_time``.
    cv_range : Tuple[float]
        The CV bounds for the dataset
    return_eval_datasets : bool
        If True, then ``test_dataset`` will include also the
        datasets with the samples used for training removed.
    return_bootstrap_datasets : bool
        If True, then ``bootstrap_datasets`` is returned as well
        (see below).

    Returns
    -------
    train_dataset : Dict[str, modules.data.TrajectorySubset]
        A dictionary associating the dataset used to train the network to its
        name.
    test_stride_datasets : Dict[str, modules.data.TrajectorySubset]
        A dictionary associating the test datasets obtained by subsampling
        the trajectory with increasing stride times to their names. If the
        name ends in "_train" the dataset may include samples in the training
        set.
    bootstrap_datasets : Dict[str, modules.data.TrajectorySubset]
        The datasets that can be used for bootstrapping. The one named
        "bootstrap_test" includes the training samples, while "bootstrap_eval"
        does not.

    """
    from modules.data import TrajectorySubset, TrajectorySubsampler

    # For training, we support only one less so that there is
    # always at least one test set with different samples.
    supported_stride_times_fs = SUPPORTED_STRIDE_TIMES.to('fs').magnitude
    training_supported_stride_times_fs = supported_stride_times_fs[:-1]

    # Check if the stride time is supported.
    train_stride_time_fs = train_stride_time.to('fs').magnitude
    try:
        train_stride_time_idx = np.where(training_supported_stride_times_fs == train_stride_time_fs)[0][0]
    except IndexError:
        raise ValueError('Only the following stride times are '
                         'supported (in fs) {}'.format(training_supported_stride_times_fs))

    test_stride_times = SUPPORTED_STRIDE_TIMES[train_stride_time_idx:]

    if trajectory_dataset.trajectory_indices is not None:
        raise ValueError('This function requires a trajectory dataset '
                         'that has not been already subsampled.')

    # These are the trajectory indices that we will use for
    # the bootstrap analysis and to randomize the training dataset.
    # It is important to make sure the cv_range for the bootstrap
    # datasets will be the same as that passed to ReweightingSn2Vacuum
    # or the fixed_bootstrap_indices features won't work.
    bootstrap_subsampler = TrajectorySubsampler(
        equilibration_time=equilibration_time,
        stride_time=BOOTSTRAP_STRIDE_TIME,
        cv_range=cv_range if cv_range is not None else FES_CV_BOUNDS
    )
    bootstrap_indices = bootstrap_subsampler(trajectory_dataset.trajectory)
    dataset = TrajectorySubset(dataset=trajectory_dataset, indices=bootstrap_indices)
    dataset = remove_plumed_unmatched_frames(dataset)
    bootstrap_indices = dataset.trajectory_indices
    bootstrap_datasets = {'bootstrap_test': dataset}

    # Now determine the trajectory indices forming the training dataset.
    # Whether we generate it by subsampling the trajectory using a constant
    # stride time or we randomly pick samples from bootstrap_indices, we
    # need to maintain the total number of samples in the dataset constant.
    subsampler = TrajectorySubsampler(
        equilibration_time=equilibration_time,
        stride_time=train_stride_time,
        cv_range=cv_range
    )
    indices_in_train_dataset = subsampler(trajectory_dataset.trajectory)
    dataset = TrajectorySubset(dataset=trajectory_dataset, indices=indices_in_train_dataset)
    dataset = remove_plumed_unmatched_frames(dataset)
    indices_in_train_dataset = dataset.trajectory_indices

    n_train_samples = len(indices_in_train_dataset)
    if randomize_training_dataset:
        suffix = TRAIN_SUFFIX + '_randomized'
        indices_in_train_dataset = np.random.choice(bootstrap_indices, n_train_samples, replace=False)
        indices_in_train_dataset.sort()

        # We don't need to remove the PLUMED-unmatched frames since we
        # already did it when we created the bootstrap_test dataset.
        dataset = TrajectorySubset(dataset=trajectory_dataset, indices=indices_in_train_dataset)
    else:
        suffix = TRAIN_SUFFIX
        # We don't need to generate an extra stride_X_train dataset to include in
        # the test_datasets dictionary as it would be identical to the train set.
        test_stride_times = test_stride_times[1:]

    # Store the training datasets in the returned values.
    train_dataset = {STRIDE_PREFIX + str(int(train_stride_time_fs)) + suffix: dataset}

    # Create the datasets to test by subsampling the trajectory with increasing stride time.
    test_stride_dataset = {}
    indices_in_train_dataset_set = set(indices_in_train_dataset)
    for stride_idx, stride_time in enumerate(sorted(test_stride_times, reverse=True)):
        # Determine the trajectory frame indices to include.
        subsampler = TrajectorySubsampler(
            equilibration_time=equilibration_time,
            stride_time=stride_time,
            cv_range=cv_range
        )
        indices = subsampler(trajectory_dataset.trajectory)

        # Dataset including the training trajectory frames.
        dataset = TrajectorySubset(dataset=trajectory_dataset, indices=indices)

        # Remove the last snapshot if needed.
        dataset = remove_plumed_unmatched_frames(dataset)
        indices = dataset.trajectory_indices

        # The name prefix of this dataset.
        stride_time_fs = stride_time.to('fs').magnitude
        stride_name_prefix = STRIDE_PREFIX + str(int(stride_time_fs))

        # Store the test dataset that include the training samples.
        test_stride_dataset[stride_name_prefix + TRAIN_SUFFIX] = dataset

        # Store the test dataset excluding the training samples.
        if return_eval_datasets:
            indices = np.array([x for x in indices if x not in indices_in_train_dataset_set])
            dataset = TrajectorySubset(dataset=trajectory_dataset, indices=indices)
            stride_name_suffix = TEST_SUFFIX + str(int(train_stride_time_fs))
            test_stride_dataset[stride_name_prefix + stride_name_suffix] = dataset

    # Finally create the evaluation dataset used for bootstrap analysis.
    if return_bootstrap_datasets:
        indices = np.array([x for x in bootstrap_indices if x not in indices_in_train_dataset_set])
        dataset = TrajectorySubset(dataset=trajectory_dataset, indices=indices)
        bootstrap_datasets['bootstrap_eval'] = dataset

        return train_dataset, test_stride_dataset, bootstrap_datasets

    return train_dataset, test_stride_dataset


def create_bootstrap_test_dataset(
        trajectory_dataset,
        equilibration_time,
        train_stride_time,
        cv_range=None
):
    """Return a test dataset for the bootstrap-based analysis without training samples.

    The dataset include all the samples, and removes the training samples
    (when ``train_stride_time`` is not None) and the samples outside the
    cv_range.

    Parameters
    ----------
    trajectory_dataset : TrajectoryDataset
        The trajectory dataset.
    equilibration_time : pint.Quantity
        The initial time of the simulation to discard.
    train_stride_time : pint.Quantity
        The stride time to use to subsample the trajectory for training.
        This is used to discard the training samples. If None, the training
        samples are not discarded.
    cv_range : Tuple[float]
        The CV bounds for the dataset.

    Returns
    -------
    test_dataset : modules.data.TrajectorySubset
        The test dataset.

    """
    from modules.data import TrajectorySubset, TrajectorySubsampler

    # Open the dataset.
    subsampler = TrajectorySubsampler(
        equilibration_time=equilibration_time,
        stride_time=50 * global_unit_registry.femtoseconds,
        cv_range=cv_range
    )
    bootstrap_indices = subsampler(trajectory_dataset.trajectory)

    # Remove the training samples.
    if train_stride_time is not None:
        subsampler = TrajectorySubsampler(
            equilibration_time=equilibration_time,
            stride_time=train_stride_time,
            cv_range=cv_range
        )
        train_indices = set(subsampler(trajectory_dataset.trajectory))
        bootstrap_indices = np.array([x for x in bootstrap_indices if x not in train_indices])

    # Create a test dataset to bootstrap on without the samples used for training.
    return TrajectorySubset(dataset=trajectory_dataset, indices=bootstrap_indices)


def remove_plumed_unmatched_frames(trajectory_dataset):
    """Remove from the dataset the frames that do not have a match in the PLUMED log.

    Because PLUMED is called before the integration step, the logs it
    saves miss the very last step. This function checks if the last
    sample in the trajectory dataset lacks PLUMED info and deletes it
    from the dataset to avoid errors.

    """
    from modules.data import TrajectorySubset

    # PLUMED does not record the very last frame saved in the amber
    # trajectory so we check if we need to discard the last sample.
    if not torch.isnan(trajectory_dataset[-1]['plumed'][0]):
        return trajectory_dataset

    print('Removing last frame of trajectory since '
          'PLUMED does not have a record for it', flush=True)

    indices = np.arange(len(trajectory_dataset)-1)
    trajectory_dataset = TrajectorySubset(trajectory_dataset, indices)
    return trajectory_dataset


@contextlib.contextmanager
def open_trajectory_sn2_vacuum(
        subsampler=None,
        with_plumed=True,
        remove_rot_trans_dof=True,
        return_dataset=True,
        remove_unmatched_frames=False,
        crd_file_path=None,
        colvar_file_path=None
):
    """Context manager to open the trajectory of the SN2 simulation in vacuum.

    Parameters
    ----------
    subsampler : Callable, optional
        The subsampler to pass to TrajectoryDataset's constuctor. This
        is ignored if ``return_dataset`` is ``False.``.
    remove_rot_trans_dof : bool, optional
        If True, add to the MDAnalysis trajectory the RemoveTransRotDOF
        transformation.
    return_dataset : bool, optional
        If False, the MDAnalysis trajectory is returned, otherwise a
        ``TrajectoryDataset`` object.
    remove_unmatched_frames : bool, optional
        If True and both ``with_plumed`` and ``return_dataset`` are
        returned, the last sample of the dataset is discarded if there
        is no PLUMED entry in the colvar file for the frame. Default
        is False.
    crd_file_path : str, optional
        The path to the trajectory coordinate file to read.
    colvar_file_path : str, optional
        The path to the PLUMED log.

    """
    from MDAnalysis.coordinates.TRJ import NCDFReader

    from modules.plumedwrapper.mdanalysis import PLUMEDReader
    from modules.data import TrajectoryDataset, RemoveTransRotDOF

    if crd_file_path is None:
        crd_file_path = SN2_VACUUM_CRD_FILE_PATH
    if colvar_file_path is None:
        colvar_file_path = SN2_VACUUM_COLVAR_FILE_PATH

    # Open the dataset.
    with NCDFReader(filename=crd_file_path) as trajectory:
        # Add PLUMED output as auxiliary information.
        if with_plumed:
            trajectory.add_auxiliary(
                auxname='plumed',
                auxdata=PLUMEDReader(
                    file_path=colvar_file_path,
                    col_names=['time', 'cv', 'metad.rbias', 'ene'],
                    units={'time': 'fs'}
                )
            )

        # In vacuum, we can remove translational/rotational degrees of
        # freedom to remove the problem of symmetric configurations. We
        # center the coordinate system on the carbon (atom 0), with the
        # chlorine (atom 1) on the z axis, and an hydrogen (atom 2) on
        # the zx plane.
        if remove_rot_trans_dof:
            trans_rot_dof_transform = RemoveTransRotDOF(
                center_atom_idx=0,
                axis_atom_idx=1,
                plane_atom_idx=2,
                axis='z',
                plane='xz',
                round_off_imprecisions=True
            )
            # Note that even if we later split the dataset, the transformation
            # is carried in the trajectory object so both training and
            # test sets will have it.
            trajectory.add_transformations(trans_rot_dof_transform)

        # Initialize the data set, subsampling the trajectory.
        if return_dataset:
            trajectory = TrajectoryDataset(
                trajectory=trajectory,
                subsampler=subsampler,
                return_batch_index=True
            )

            if with_plumed and remove_unmatched_frames:
                trajectory = remove_plumed_unmatched_frames(trajectory)

        yield trajectory


def save_dataset_input_information_sn2_vacuum(dataset, dir_path, save_coordinates=True):
    """Save the input information of the dataset in the given directory.

    Two files are created in the directory. A 'input_coordinates.pdb' file with
    the (unmapped) coordinates, and a 'input_potentials.dat' XVG file including
    time, metadynamics normalized bias, reference potential, and CV.

    If the files already exist, they are not overwritten. Saving the coordinates
    can be avoided by setting ``save_coordinates=False``.

    """
    from modules.plumedwrapper import io as plumedio

    # Create directory.
    os.makedirs(dir_path, exist_ok=True)

    # Save coordinates.
    if save_coordinates:
        pdb_file_path = os.path.join(dir_path, INPUT_COORDINATES_FILE_NAME)
        if not os.path.isfile(pdb_file_path):
            dataset.save(
                topology_file_path=SN2_VACUUM_PRMTOP_FILE_PATH,
                output_file_path=pdb_file_path,
                multiframe=True
            )

    # Save potential information.
    potentials_file_path = os.path.join(dir_path, INPUT_POTENTIALS_FILE_NAME)
    if not os.path.isfile(potentials_file_path):
        # Get the indices in the PLUMED aux info.
        get_column_idx_func = dataset.trajectory.get_aux_attribute('plumed', 'get_column_idx')
        col_indices = {
            'time': 0,
            'rbias': get_column_idx_func('metad.rbias'),
            'reference_potential': get_column_idx_func('ene'),
            'cv': get_column_idx_func('cv')
        }

        # Initialize arrays to save.
        temp_table = {}
        for col_name in col_indices.keys():
            temp_table[col_name] = np.empty(len(dataset))

        for i, ts in enumerate(dataset.iter_as_ts()):
            for col_name, col_idx in col_indices.items():
                temp_table[col_name][i] = ts.aux['plumed'][col_idx]

        plumedio.write_table(temp_table, file_path=potentials_file_path)


def compute_DF(reweighting, output_file_path, fes_dir_path=None):
    """Compute the difference of free energy between metastable states and save the result on disk.

    The output file is a dictionary saved in JSON format that (depending
    on the global configuration of the script), include a subset of the following keys

    If COMPUTE_DF_BASINS_FROM_FES is True, the dictionary contains

        'delta_f' :
            The Delta f between the two metastable states computed by
            integrating the FES in kJ/mol.
        'f_basin1_from_fes' :
            The free energy of the first metastable state computed by
            summing the the Delta f of the metastable state between
            low and high-level potential to the free energy obtained
            by integrating the reference FES in kJ/mol.
        'f_basin2_from_fes' :
            The free energy of the second metastable state computed by
            summing the the Delta f of the metastable state between
            low and high-level potential to the free energy obtained
            by integrating the reference FES.
        'f_barrier' :
            The free energy barrier f_TS - f_1, where f_TS is the free
            energy of the transition state and f_1 is the free energy
            of the first metastable state.
        'gf_barrier' :
            The free energy barrier gf_TS - f_1, where gf_TS is the GEOMETRIC
            free energy of the transition state and f_1 is the free energy
            of the first metastable state.

    For each of these values, the dictionary might include bootstrap statistics.
    For example, for 'delta_f' there are

        'bootstrap_delta_f_ci_lb':
            The lower bound of the 95% CI for the Delta f in kJ/mol.
        'bootstrap_delta_f_ci_hb':
            The upper bound of the 95% CI for the Delta f in kJ/mol.
        'bootstrap_delta_f_mean':
            The mean bootstrap statistic for the Delta f in kJ/mol.
        'bootstrap_delta_f_median':
            The median bootstrap statistic for the Delta f in kJ/mol.

    Moreover, a version of those statistic computed from a "smoothed"
    FES with a moving average filter is also given with the keyword
    convention 'smooth_STATISTIC' (e.g., 'smooth_delta_f').

    Finally, for each basin with name BASINNAME defined in the
    COMPUTE_DF_BASINS_FROM_H global variable, the dictionary includes
    a 'delta_f_BASINNAME': The Delta f between the basin at the low and
    high-level potential in kJ/mol.

    Parameters
    ----------
    reweighting : DatasetReweighting
        The reweighting facility class.
    output_file_path : str
        The path where the output data is stored in JSON format.
    fes_dir_path : str or List[str]
        The path to the "fes" folder created by PLUMED with sumhills.
        This is read to obtain the reference FES, which is necessary
        to compute the final Delta f between metastable states and the
        Delta FES. If a list of files, all of them are read, and the
        reference FES is computed as the average FES.

    """
    import json
    energy_unit = reweighting.unit_registry.kJ / reweighting.unit_registry.mol

    if fes_dir_path is None:
        # fes_dir_path = [os.path.join(SN2_VACUUM_AMBER_DIR_PATH, 'fes')]
        # Use the average of the 4 repeats for the reference metadynamics FES.
        main_amber_dir_path = os.path.dirname(SN2_VACUUM_AMBER_DIR_PATH)
        fes_dir_path = [os.path.join(main_amber_dir_path, 'repeat-'+str(i), 'fes') for i in range(4)]
    elif isinstance(fes_dir_path, str):
        # Make sure fes_dir_path is a list of paths pointing to metad fes
        # that must be averaged out. This is useful for merging datasets.
        fes_dir_path = [fes_dir_path]

    # Reweight the free energy surface.
    if COMPUTE_DFES:
        # The FES is stored in the separate file reweighting.fes_file_path.
        reweighting.reweight_fes()

    # This data will be saved in JSON format at the end.
    data = {}

    # Compute F_basin2 - F_basin1 from the integration of the final FES.
    if COMPUTE_DF_BASINS_FROM_FES:
        # Build a reference FES as the average of the repeated MetaD calcultions.
        all_metad_fes = [load_metad_fes(path)[0][-1]['file.free'] for path in fes_dir_path]
        if len(all_metad_fes) == 1:
            reference_f_s = all_metad_fes[0]
        else:
            reference_f_s = np.mean(all_metad_fes, axis=0)

        # Add units.
        reference_f_s *= energy_unit

        # Compute both the raw and smoothed (with an average filter) free energy.
        for window_size in [0, SMOOTHING_WINDOW_SIZE]:
            # Compute the free energy difference.
            delta_f_from_fes_data = reweighting.reweight_DF_metastable_states(
                reference_f_s=reference_f_s,
                basin1_cv_bounds=BASIN1_CV_BOUNDS,
                basin2_cv_bounds=BASIN2_CV_BOUNDS,
                smooth_window_size=window_size
            )

            # Convert everything to kJ/mol before storing.
            for k, v in delta_f_from_fes_data.items():
                delta_f_from_fes_data[k] = v.to('kJ/mol').magnitude

            keys_to_delete = []
            update_dict = {}
            for k, v in delta_f_from_fes_data.items():
                new_k = k

                # Add the "smooth" prefix when the filter is used.
                if window_size > 0:
                    if 'delta_f' in k:
                        new_k = new_k.replace('delta_f', 'smooth_delta_f')
                    elif 'gf_' in k:  # geometric  barrier
                        new_k = new_k.replace('gf_', 'smooth_gf_')
                    else:
                        new_k = new_k.replace('f_', 'smooth_f_')

                # Add the "from_fes" suffix to avoid overriding quantities calculated with FEP.
                if 'f_basin1' in k:
                    new_k = new_k.replace('f_basin1', 'f_basin1_from_fes')
                elif 'f_basin2' in k:
                    new_k = new_k.replace('f_basin2', 'f_basin2_from_fes')

                # Unpack the bootstrap CIs into lower and upper bounds.
                if ('bootstrap' in k) and ('_ci' in k):
                    update_dict[new_k + '_lb'] = v[0]
                    update_dict[new_k + '_hb'] = v[1]
                else:
                    update_dict[new_k] = v
                keys_to_delete.append(k)

            for k in keys_to_delete:
                del delta_f_from_fes_data[k]
            delta_f_from_fes_data.update(update_dict)
            data.update(delta_f_from_fes_data)

    # Compute the total free energy difference of the different
    # basins from the low-level to the high-level Hamiltonian.
    for basin_name, basin_cv_range in COMPUTE_DF_BASINS_FROM_H.items():
        # Compute the total difference in free energy between the two Hamiltonians.
        total_delta_f, total_bootstrap_ci, total_bootstrap_mean, total_bootstrap_median = reweighting.fep(cv_range=basin_cv_range)

        # Convert everything to kJ/mol before storing.
        total_delta_f = total_delta_f.to('kJ/mol').magnitude
        total_bootstrap_ci = total_bootstrap_ci.to('kJ/mol').magnitude
        total_bootstrap_mean = total_bootstrap_mean.to('kJ/mol').magnitude
        total_bootstrap_median = total_bootstrap_median.to('kJ/mol').magnitude

        data.update({
            'delta_f_'+basin_name: total_delta_f,
            'bootstrap_delta_f_'+basin_name+'_ci_lb': total_bootstrap_ci[0],
            'bootstrap_delta_f_'+basin_name+'_ci_hb': total_bootstrap_ci[1],
            'bootstrap_delta_f_'+basin_name+'_mean': total_bootstrap_mean,
            'bootstrap_delta_f_'+basin_name+'_median': total_bootstrap_median,
        })

    # Store everything.
    with open(output_file_path, 'w') as f:
        json.dump(data, f)


# =============================================================================
# STANDARD REWEIGHTING OF THE SN2 VACUUM EXPERIMENT
# =============================================================================

class ReweightingSn2Vacuum(DatasetReweighting):
    """Implementation of dataset reweighting for the SN2 reaction in vacuum.

    See documentation of modules.reweighting.DatasetReweighting for more info.

    """

    @staticmethod
    def get_traj_info(dataset):
        """Implementation of DatasetReweighting.get_traj_info()."""
        get_column_idx_func = dataset.trajectory.get_aux_attribute('plumed', 'get_column_idx')
        col_indices = [get_column_idx_func(name) for name in ['cv', 'metad.rbias', 'ene']]
        cvs, metad_rbias, reference_potentials = read_all_plumed_rows(dataset, col_indices)

        # Add units before returning.
        reference_potentials *= global_unit_registry.kJ / global_unit_registry.mol
        metad_rbias *= global_unit_registry.kJ / global_unit_registry.mol
        return cvs, reference_potentials, metad_rbias

    def compute_potentials(self, batch_positions):
        """Implementation of DatasetReweighting.compute_potentials()."""
        from modules.functions.qm.psi4 import notorch_potential_energy_psi4

        # Compute potentials.
        mp2_potentials = notorch_potential_energy_psi4(
            batch_positions=batch_positions,
            processes=N_PROCESSES if self.process_pool is None else self.process_pool,
            **SN2_VACUUM_PSI4_KWARGS
        )

        # DatasetReweighting requires the energy per mole.
        mp2_potentials *= batch_positions._REGISTRY.avogadro_constant
        return mp2_potentials

    def compute_det_dcv(self, batch_positions):
        """Implementation of DatasetReweighting.compute_det_dcv()."""
        from modules.functions.geometry import to_batch_atom_3_shape
        from modules.nets.functions.math import batchwise_dot

        # Convert ndarray to Tensor to use accelerated version.
        if not isinstance(batch_positions, torch.Tensor):
            batch_positions = torch.tensor(batch_positions)

        # Make sure the positions have shape (batch, n_atoms, 3).
        batch_positions = to_batch_atom_3_shape(batch_positions)

        # Compute the difference vectors.
        diff_F_C = batch_positions[:, 5] - batch_positions[:, 0]
        diff_Cl_C = batch_positions[:, 1] - batch_positions[:, 0]

        # Compute the distances w.r.t. carbon.
        dist_F_C = (diff_F_C**2).sum(dim=1, keepdim=True).sqrt()
        dist_Cl_C = (diff_Cl_C**2).sum(dim=1, keepdim=True).sqrt()

        # Compute unit vectors connecting carbon to F and Cl.
        unit_F_C = diff_F_C / dist_F_C
        unit_Cl_C = diff_Cl_C / dist_Cl_C

        # Compute the norm of the CV gradient.
        norm_dcv = torch.sqrt(2 * (0.81**2 + 0.59**2 - 0.81*0.59 * batchwise_dot(unit_F_C, unit_Cl_C)))

        # Convert to numpy before returning.
        return norm_dcv.squeeze(-1).detach().numpy()


def run_standard_reweighting_sn2_vacuum(
        equilibration_time,
        output_dir_path,
        train_stride_time=None,
        process_pool=None
):
    """
    Run the standard reweighting of the SN2 reaction in vacuum.

    The function performs the standard FEP analysis for all the
    datasets created by the function ``create_datasets()``, each
    subsampled with a different stride time. This enables observing
    the performance of reweighting as the number of samples in
    the dataset increases.

    Parameters
    ----------
    equilibration_time : pint.Quantity
        The initial time of the simulation to discard.
    output_dir_path : str
        The path of the directory where to save the result files. The
        directory will contain a file used to cache the MP2 potential
        energies and several subdirectories, each holding the reweighted
        FES for a different value of ``stride_times``.
    train_stride_time : pint.Quantity, optional
        The stride time used by the training dataset. If not given, only
        the training datasets are analyzed, starting with the largest
        supported stride time. Otherwise, only the test datasets are.
    process_pool : multiprocessing.Pool, optional
        A pool of processes to use to parallelize the MP2 potential
        calculations.

    """
    # Determine datasets to analyze.
    if train_stride_time is None:
        train_stride_time = SUPPORTED_STRIDE_TIMES[0]
        analyze_eval_datasets = False
    else:
        analyze_eval_datasets = True

    # Open the dataset.
    with open_trajectory_sn2_vacuum(remove_rot_trans_dof=False) as trajectory_dataset:
        # Determine all the stride times used for the train/test datasets to evaluate.
        train_dataset, test_datasets = create_datasets(
            trajectory_dataset, equilibration_time,
            train_stride_time=train_stride_time,
            return_eval_datasets=analyze_eval_datasets
        )
        all_datasets = {**train_dataset, **test_datasets}

        for dataset_name, dataset in all_datasets.items():
            # Check if we need to analyze this dataset.
            if analyze_eval_datasets and '_train' in dataset_name:
                continue

            # Determine file paths.
            dataset_info_dir_path = os.path.join(output_dir_path, DATASET_INFO_SUBDIR, dataset_name)
            reweighting_results_dir_path = os.path.join(output_dir_path, REWEIGHTING_RESULTS_SUBDIR)
            dataset_output_dir_path = os.path.join(reweighting_results_dir_path, dataset_name)

            # We save only the potential information (i.e., no input coordinates)
            # since the trajectory can become very large for small stride times.
            # We use the potentials in the analysis notebook.
            save_dataset_input_information_sn2_vacuum(
                dataset, dataset_info_dir_path, save_coordinates=False)

            # Standard reweighting. We keep a single cache file for potentials
            # and save one FES file for each value of stride time.
            reweighting = ReweightingSn2Vacuum(
                datasets=dataset,
                n_bins=FES_GRID_N_BINS,
                temperature=TEMPERATURE,
                cv_bounds=FES_CV_BOUNDS,
                compute_geometric_fes=True,
                n_bootstrap_cycles=N_BOOTSTRAP_CYCLES,
                fes_file_path=os.path.join(dataset_output_dir_path, DELTA_FES_FILE_NAME),
                bootstrap_fes_file_path=os.path.join(dataset_output_dir_path, BOOT_DELTA_FES_FILE_NAME),
                potentials_file_paths=os.path.join(reweighting_results_dir_path, MP2_POTENTIALS_CACHE_FILE_NAME),
                det_dcv_file_paths=os.path.join(reweighting_results_dir_path, DET_DCV_CACHE_FILE_NAME),
                potentials_batch_size=N_PROCESSES,
                potentials_write_interval=4,
                process_pool=process_pool
            )

            # Compute the difference in free energy.
            compute_DF(reweighting, os.path.join(dataset_output_dir_path, DELTA_F_FILE_NAME))


def analyze_bootstrap_standard_reweighting_sn2_vacuum(
        equilibration_time,
        output_dir_path,
        train_stride_time=None,
        process_pool=None
):
    """
    Run the standard reweighting of the SN2 reaction in vacuum.

    The function performs the standard FEP for several random subsets
    of the full trajectory for an increasing size of the subset. This
    to implement bootstrap analysis and observe the convergence of
    standard FEP with an increasing number of samples.

    Parameters
    ----------
    equilibration_time : pint.Quantity
        The initial time of the simulation to discard.
    output_dir_path : str
        The path of the directory where to save the result files. The
        directory will contain a file used to cache the MP2 potential
        energies and several subdirectories, each holding the reweighted
        FES for a different value of ``stride_times``.
    train_stride_time : pint.Quantity, optional
        The stride time used by the training dataset. If not given, only
        the training datasets are analyzed, starting with the largest
        supported stride time. Otherwise, only the test datasets are.
    process_pool : multiprocessing.Pool, optional
        A pool of processes to use to parallelize the MP2 potential
        calculations.

    """
    reweighting_results_dir_path = os.path.join(output_dir_path, REWEIGHTING_RESULTS_SUBDIR)

    # Open the dataset.
    with open_trajectory_sn2_vacuum(remove_rot_trans_dof=False) as trajectory_dataset:
        # Create a test dataset to bootstrap on without the samples used for training.
        test_dataset = create_bootstrap_test_dataset(
            trajectory_dataset, equilibration_time, train_stride_time
        )

        # This determines the name of the dataset.
        if train_stride_time is None:
            dataset_name_suffix = '_train'
        else:
            train_stride_time_str = str(int(train_stride_time.to('fs').magnitude))
            dataset_name_suffix = '_test_' + train_stride_time_str

        # Compute free energies.
        for subsample_size in SUBSAMPLE_SIZES:
            dataset_name = 'bootstrap_' + str(subsample_size) + dataset_name_suffix

            # Determine file paths.
            dataset_output_dir_path = os.path.join(reweighting_results_dir_path, dataset_name)

            # Standard reweighting. We keep a single cache file for potentials
            # and save one FES file for each value of stride time.
            reweighting = ReweightingSn2Vacuum(
                datasets=test_dataset,
                n_bins=FES_GRID_N_BINS,
                temperature=TEMPERATURE,
                cv_bounds=FES_CV_BOUNDS,
                compute_geometric_fes=True,
                n_bootstrap_cycles=N_BOOTSTRAP_CYCLES,
                subsample_size=subsample_size,
                fes_file_path=os.path.join(dataset_output_dir_path, DELTA_FES_FILE_NAME),
                bootstrap_fes_file_path=os.path.join(dataset_output_dir_path, BOOT_DELTA_FES_FILE_NAME),
                potentials_file_paths=os.path.join(reweighting_results_dir_path, MP2_POTENTIALS_CACHE_FILE_NAME),
                det_dcv_file_paths=os.path.join(reweighting_results_dir_path, DET_DCV_CACHE_FILE_NAME),
                potentials_batch_size=N_PROCESSES,
                potentials_write_interval=4,
                process_pool=process_pool
            )

            # Compute the difference in free energy.
            compute_DF(reweighting, os.path.join(dataset_output_dir_path, DELTA_F_FILE_NAME))


def run_merged_standard_reweighting_sn2_vacuum(
        equilibration_time,
        output_dir_path,
        train_stride_time=None,
        process_pool=None
):
    """
    Same as run_standard_reweighting_sn2_vacuum but with the repeated trajectories concatenated.

    This is used to compute the reference Delta f from the independent
    metadynamics simulations.

    """
    # Determine datasets to analyze.
    if train_stride_time is None:
        train_stride_time = SUPPORTED_STRIDE_TIMES[0]
        analyze_eval_datasets = False
    else:
        analyze_eval_datasets = True

    # Find the paths to the repeated trajectories and mp2 potentials file paths.
    main_dir_path = os.path.dirname(output_dir_path)
    crd_file_paths = [None for _ in range(4)]
    colvar_file_paths = [None for _ in range(4)]
    reference_fes_dir_paths = [None for _ in range(4)]
    potentials_file_paths = [None for _ in range(4)]
    det_dcv_file_paths = [None for _ in range(4)]
    for i in range(4):
        repeat_name = 'repeat-' + str(i)
        amber_dir_path = os.path.join(SN2_VACUUM_AMBER_DIR_PATH, repeat_name)
        crd_file_paths[i] = os.path.join(amber_dir_path, 'sn2_vacuum.crd')
        colvar_file_paths[i] = os.path.join(amber_dir_path, 'colvar')
        reference_fes_dir_paths[i] = os.path.join(amber_dir_path, 'fes')
        potentials_file_paths[i] = os.path.join(main_dir_path, repeat_name, REWEIGHTING_RESULTS_SUBDIR, MP2_POTENTIALS_CACHE_FILE_NAME)
        det_dcv_file_paths[i] = os.path.join(main_dir_path, repeat_name, REWEIGHTING_RESULTS_SUBDIR, DET_DCV_CACHE_FILE_NAME)

    # Open the dataset.
    with open_trajectory_sn2_vacuum(remove_rot_trans_dof=False, crd_file_path=crd_file_paths[0], colvar_file_path=colvar_file_paths[0]) as dataset0:
        with open_trajectory_sn2_vacuum(remove_rot_trans_dof=False, crd_file_path=crd_file_paths[1], colvar_file_path=colvar_file_paths[1]) as dataset1:
            with open_trajectory_sn2_vacuum(remove_rot_trans_dof=False, crd_file_path=crd_file_paths[2], colvar_file_path=colvar_file_paths[2]) as dataset2:
                with open_trajectory_sn2_vacuum(remove_rot_trans_dof=False, crd_file_path=crd_file_paths[3], colvar_file_path=colvar_file_paths[3]) as dataset3:
                    traj_datasets = [dataset0, dataset1, dataset2, dataset3]

                    # Create all the datasets used for training.
                    all_datasets = {}
                    for traj_dataset in traj_datasets:
                        # Determine all the stride times used for the train/test datasets to evaluate.
                        train_dataset, test_datasets = create_datasets(
                            traj_dataset, equilibration_time,
                            train_stride_time=train_stride_time,
                            return_eval_datasets=analyze_eval_datasets
                        )

                        # Update the list of datasets.
                        for dataset_name, dataset in {**train_dataset, **test_datasets}.items():
                            try:
                                all_datasets[dataset_name].append(dataset)
                            except KeyError:
                                all_datasets[dataset_name] = [dataset]

                    for dataset_name, datasets in all_datasets.items():
                        # Check if we need to analyze this dataset.
                        if analyze_eval_datasets and '_train' in dataset_name:
                            continue

                        # Determine file paths.
                        dataset_info_dir_path = os.path.join(output_dir_path, DATASET_INFO_SUBDIR, dataset_name)
                        reweighting_results_dir_path = os.path.join(output_dir_path, REWEIGHTING_RESULTS_SUBDIR)
                        dataset_output_dir_path = os.path.join(reweighting_results_dir_path, dataset_name)

                        # We save only the potential information (i.e., no input coordinates)
                        # since the trajectory can become very large for small stride times.
                        # We use the potentials in the analysis notebook.
                        # save_dataset_input_information_sn2_vacuum(
                        #     datasets, dataset_info_dir_path, save_coordinates=False)

                        # Standard reweighting. We keep a single cache file for potentials
                        # and save one FES file for each value of stride time.
                        reweighting = ReweightingSn2Vacuum(
                            datasets=datasets,
                            n_bins=FES_GRID_N_BINS,
                            temperature=TEMPERATURE,
                            cv_bounds=FES_CV_BOUNDS,
                            compute_geometric_fes=True,
                            n_bootstrap_cycles=N_BOOTSTRAP_CYCLES,
                            fes_file_path=os.path.join(dataset_output_dir_path, DELTA_FES_FILE_NAME),
                            bootstrap_fes_file_path=os.path.join(dataset_output_dir_path, BOOT_DELTA_FES_FILE_NAME),
                            potentials_file_paths=potentials_file_paths,
                            det_dcv_file_paths=det_dcv_file_paths,
                            potentials_batch_size=N_PROCESSES,
                            potentials_write_interval=4,
                            process_pool=process_pool
                        )

                        # Compute the difference in free energy.
                        compute_DF(reweighting, os.path.join(dataset_output_dir_path, DELTA_F_FILE_NAME),
                                   fes_dir_path=reference_fes_dir_paths)


# =============================================================================
# TARGETED REWEIGHTING OF THE SN2 VACUUM EXPERIMENT
# =============================================================================

class SN2VacuumFlow(torch.nn.Module):
    """The normalizing flow used for the targeted reweighting of the SN2 reaction in vacuum.

    Parameters
    ----------
    trajectory : mdanalysis.Trajectory
        The trajectory with information on the atoms.
    n_maf_layers : int, optional
        Number of MAF layers in the network.
    batch_norm : bool, optional
        If True, use batch normalization between layers.
    weight_norm : bool, optional
        If True, use weight normalization.
    degrees_hidden_motif : numpy.ndarray[int], optional
        This can be used to control the degrees assigned to the hidden
        layers within the MAF. See the MAF class documentation for the
        the syntax.
    transformer : torch.Module or List[torch.Module]
        A transformer or a list of transformers (one for each MAF layer)
        to pass to the MAF object constructor. If not given, the affine
        transformer is used.
    cv_range : Tuple[float]
        This is used in combination with the NeuralSplineTransformer, and
        it defines the limits of the CV to be mapped.
    **maf_kwargs
        More keyword arguments to pass to the MAF object constructor.

    See Also
    --------
    from modules.nets.modules.flows import MAF

    """
    # TODO: Implement flatten + translation/rotation as pytorch Transformation rather than MDAnalysis'?

    # Blocks for the atoms. The first atom (carbon) is constant. The second
    # (chlorine) has only the z coordinate free. The third (hydrogen) has
    # both x and z coordinate free.
    BLOCKS = [1, 2, 3, 3, 3]

    # For TFEP, we don't condition based on the CV.
    DIMENSION_CONDITIONING = 0

    # Lower and upper bounds for the 18 degrees of freedom and the CV extracted
    # from the trajectory amber_output-100ns/repeat-0 with stride 500fs.
    DOF_LB = np.array([ 0.0000,  0.0000,  0.0000,
                        0.0000,  0.0000,  1.5751,
                       -1.4768,  0.0000, -1.1415,
                       -1.1823, -1.4190, -1.0909,
                       -1.2138, -1.3375, -1.1029,
                       -5.5739, -5.7555, -6.1482])
    DOF_UB = np.array([0.0000, 0.0000, 0.0000,
                       0.0000, 0.0000, 6.4258,
                       1.4764, 0.0000, 1.2673,
                       1.1928, 1.4154, 1.2100,
                       1.1777, 1.4731, 1.2285,
                       5.7642, 6.1905, 1.6971])

    # Standard architectures.
    ARCHITECTURES = {
        'split': dict(dimensions_hidden=1, split_conditioner=True, transformer=None),
        'cheap': dict(dimensions_hidden=[11], split_conditioner=False, transformer=None),
        'expensive': dict(dimensions_hidden=[22], split_conditioner=False, transformer=None),
        'expensive2': dict(dimensions_hidden=[22, 22], split_conditioner=False, transformer=None),

        'sossplit': dict(dimensions_hidden=1, split_conditioner=True, transformer=SOSPolynomialTransformer(2)),
        'soscheap': dict(dimensions_hidden=[11], split_conditioner=False, transformer=SOSPolynomialTransformer(2)),
        'sosexpensive': dict(dimensions_hidden=[22], split_conditioner=False, transformer=SOSPolynomialTransformer(2)),

        # The x0 and xf parameters of NeuralSplineTransformer are initialized in __init__.
        'splineexpensive2': dict(dimensions_hidden=[22, 22], split_conditioner=False,
                                 transformer=NeuralSplineTransformer(torch.tensor([]), torch.tensor([]), n_bins=5)),
    }

    def __init__(
            self,
            trajectory,
            n_maf_layers=4,
            batch_norm=False,
            weight_norm=False,
            degrees_hidden_motif=None,
            transformer=None,
            cv_range=None,
            **maf_kwargs
    ):
        from modules.nets.modules.flows import InvertibleBatchNorm1d, MAF, NormalizingFlow

        super().__init__()

        dimension = 3*trajectory.n_atoms - 6

        # Make sure there is one transformer and degree motif for each maf layer.
        if not isinstance(transformer, list):
            transformer = [transformer] * n_maf_layers
        else:
            assert len(transformer) == n_maf_layers

        if degrees_hidden_motif is None or np.issubdtype(type(degrees_hidden_motif[0]), np.integer):
            degrees_hidden_motif = [degrees_hidden_motif] * n_maf_layers
        else:
            assert len(degrees_hidden_motif) == n_maf_layers

        # When the NeuralSplineTransformer is used with a cv_range, this
        # flag is True and the first DOF is set as the CV by forward().
        self._propagate_cv = False

        flows = []
        for maf_layer_idx in range(n_maf_layers):
            degrees_in = 'input' if (maf_layer_idx%2 == 0) else 'reversed'

            if isinstance(transformer[maf_layer_idx], MobiusTransformer):
                # Initialize the blocks parameters.
                transformer[maf_layer_idx].blocks = self.BLOCKS
                blocks = self.BLOCKS
            else:
                blocks = 1

            if isinstance(transformer[maf_layer_idx], NeuralSplineTransformer):
                # Initialize the x0 and xf parameters (the first and final knots).
                # Ignore the constrained indices.
                constrained_dof_indices_set = set(trajectory.transformations[0].constrained_dof_indices.tolist())
                unconstrained_atom_indices = [i for i in range(len(self.DOF_LB)) if i not in constrained_dof_indices_set]
                x0 = self.DOF_LB[unconstrained_atom_indices] - 1
                xf = self.DOF_UB[unconstrained_atom_indices] + 1

                # If the CV is limited, set the bounds accordingly in the first dimension.
                if cv_range is not None:
                    self._propagate_cv = True
                    x0[0] = cv_range[0]
                    xf[0] = cv_range[1]

                # Remove the conditioning DOFs before setting the transformer parameters.
                x0 = x0[self.DIMENSION_CONDITIONING:]
                xf = xf[self.DIMENSION_CONDITIONING:]

                transformer[maf_layer_idx].x0 = torch.tensor(x0)
                transformer[maf_layer_idx].xf = torch.tensor(xf)
                transformer[maf_layer_idx].y0 = torch.tensor(x0)
                transformer[maf_layer_idx].yf = torch.tensor(xf)

            flows.append(MAF(
                dimension=dimension,
                dimension_conditioning=self.DIMENSION_CONDITIONING,
                degrees_in=degrees_in,
                degrees_hidden_motif=degrees_hidden_motif[maf_layer_idx],
                weight_norm=weight_norm,
                blocks=blocks,
                transformer=transformer[maf_layer_idx],
                initialize_identity=True,
                **maf_kwargs
            ))

            if batch_norm and maf_layer_idx != n_maf_layers-1:
                flows.append(InvertibleBatchNorm1d(
                    n_features=dimension,
                    n_constant_features=self.DIMENSION_CONDITIONING,
                    affine=True,
                    track_running_stats=True,
                    initialize_identity=True
                ))

        # The flow should map only the dimensions that are not kept fixed
        # by the RemoveTransRotDOF transformation to incorporate symmetry.
        constrained_indices = trajectory.transformations[0].constrained_dof_indices
        self.flow = NormalizingFlow(
            *flows,
            constant_indices=constrained_indices,
            weight_norm=weight_norm
        )

        # For forward to work correctly, the chlorine atom should be constrained
        # on the z-axis so that a minimal transformation is required to
        # implement a mapping that is only conditional (i.e. doesn't alter) the CV.
        constrained_indices = set(trajectory.transformations[0].constrained_dof_indices)
        assert {0, 1, 2, 3, 4}.issubset(constrained_indices)
        assert 5 not in constrained_indices

        # Also for the self.BLOCKS constant to be valid, only another coordinate
        # of the third atom should be constant, and no others.
        assert len(constrained_indices) == 6
        assert len({6, 7, 8}.intersection(constrained_indices)) == 1

    @classmethod
    def from_architecture(cls, trajectory, architecture_name, n_maf_layers=4,
                          batch_norm=False, weight_norm=False, cv_range=None):
        """Static constuctor used to start the neural network from pre-defined architectures.

        The parameters are all directly passed to the main constructor, except
        for ``architecture_name``, which must match one of the keyword in the
        cls.ARCHITECTURES dictionary.

        """
        return cls(
            trajectory=trajectory,
            dimensions_hidden=cls.ARCHITECTURES[architecture_name]['dimensions_hidden'],
            degrees_hidden_motif=cls.ARCHITECTURES[architecture_name].get('degrees_hidden_motif', None),
            split_conditioner=cls.ARCHITECTURES[architecture_name]['split_conditioner'],
            transformer=cls.ARCHITECTURES[architecture_name]['transformer'],
            n_maf_layers=n_maf_layers,
            batch_norm=batch_norm,
            weight_norm=weight_norm,
            cv_range=cv_range
        )

    def n_parameters(self):
        """The total number of trainable parameters"""
        return self.flow.n_parameters()

    def forward(self, x):
        if self._propagate_cv:
            return self._forward_with_cv(x)
        return self.flow(x)

    def _forward_with_cv(self, x):
        """Transform the first degree of freedom in the network so that it is the
        CV rather than the C-Cl distance. Then, once the flow has propagated it (or
        preserved if self.DIMENSION_CONDITIONING > 0) reconvert the first DOF so
        that it is the C-Cl distance.

        """
        from modules.functions.geometry import to_atom_batch_3_shape, batchwise_atom_dist

        # Convert the representation using s rather than dist_c_cl.
        # The xyz of C and x-y of Cl coordinates are constrained to be 0.0.
        c_pos = torch.zeros((x.shape[0], 3), dtype=x.dtype)
        atom_positions = to_atom_batch_3_shape(x)
        dist_c_f = batchwise_atom_dist(c_pos, atom_positions[5])
        dist_c_cl = x[:, 5]
        cv = 0.81*dist_c_f - 0.59*dist_c_cl

        # Do not modify the input in place.
        y = torch.empty_like(x)
        y[:, :5] = x[:, :5]
        y[:, 5] = cv
        y[:, 6:] = x[:, 6:]

        # Map the coordinates.
        x, log_det_J = self.flow(y)

        # Convert back to using dist_c_cl instead of s.
        atom_positions = to_atom_batch_3_shape(x)
        dist_c_f = batchwise_atom_dist(c_pos, atom_positions[5])

        # The CV is the first degree of freedom fed to the neural network
        # because indices 0-4 are constrained to 0.0. If self.DIMENSION_CONDITIONING
        # is >= 1, then the CV is preserved and we don't need to update it.
        if self.DIMENSION_CONDITIONING == 0:
            cv = x[:, 5]
        # Convert the CV back to the chlorine position.
        x[:, 5] = (0.81*dist_c_f - cv) / 0.59

        return x, log_det_J


class CVPreservingSN2VacuumFlow(SN2VacuumFlow):
    """A normalizing flow that preserves the CV for the SN2 reaction system in vacuum."""

    # Blocks for the atoms. The first atom (carbon) is constant. The second
    # (chlorine) has only the z coordinate free, but it's a conditioning
    # feature. The third (hydrogen) has both x and z coordinate free.
    BLOCKS = [2, 3, 3, 3]

    # The flow is independent but conditioned on the CV value.
    DIMENSION_CONDITIONING = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We always propagate the CV here.
        self._propagate_cv = True


def compute_restraints_potential_energy_sn2_vacuum(
        batch_positions,
        s_restraint_bounds=None,
        cl_h_restraint=False,
        f_h_restraint=False,
):
    """Compute the potential energy of the restraints used in the AMBER simulations.

    Parameters
    ----------
    batch_positions
        The positions in Angstrom.

    Returns
    -------
    restraints_potential_energy
        The total potential energy of the restraints in kJ/mol.

    """
    from modules.functions.geometry import to_atom_batch_3_shape, batchwise_atom_dist
    from modules.functions.restraints import upper_walls_plumed, lower_walls_plumed

    batch_size = batch_positions.shape[0]

    # Convert to (n_atom, batch_size, 3) shape to take distances between singular atoms.
    atom_positions = to_atom_batch_3_shape(batch_positions)

    # Compute all the restrained distances.
    dist_c_cl = batchwise_atom_dist(atom_positions[0], atom_positions[1])
    dist_c_f = batchwise_atom_dist(atom_positions[0], atom_positions[5])

    restraints_potential_energy = torch.zeros(batch_size, dtype=batch_positions.dtype)
    for x in [dist_c_cl, dist_c_f]:
        restraints_potential_energy += upper_walls_plumed(x, at=4.0, kappa=150.0)

    if s_restraint_bounds is not None:
        cvs = 0.81 * dist_c_f - 0.59 * dist_c_cl
        restraints_potential_energy += lower_walls_plumed(cvs, at=s_restraint_bounds[0], kappa=1000.0)
        restraints_potential_energy += upper_walls_plumed(cvs, at=s_restraint_bounds[1], kappa=1000.0)

    if cl_h_restraint:
        dist_cl_h1 = batchwise_atom_dist(atom_positions[1], atom_positions[2])
        dist_cl_h2 = batchwise_atom_dist(atom_positions[1], atom_positions[3])
        dist_cl_h3 = batchwise_atom_dist(atom_positions[1], atom_positions[4])
        for x in [dist_cl_h1, dist_cl_h2, dist_cl_h3]:
            restraints_potential_energy += lower_walls_plumed(x, at=1.7, kappa=150.0)

    # Check if there are F-H restraints.
    if f_h_restraint:
        dist_f_h1 = batchwise_atom_dist(atom_positions[5], atom_positions[2])
        dist_f_h2 = batchwise_atom_dist(atom_positions[5], atom_positions[3])
        dist_f_h3 = batchwise_atom_dist(atom_positions[5], atom_positions[4])
        for x in [dist_f_h1, dist_f_h2, dist_f_h3]:
            restraints_potential_energy += lower_walls_plumed(x, at=1.7, kappa=150.0)

    return restraints_potential_energy


def train_normalizing_flow_sn2_vacuum(
        dataset,
        flow,
        loss_func,
        s_restraint_bounds=None,
        cl_h_restraint=False,
        f_h_restraint=False,
        grad_cov_precond=False,
        damping=None,
        net_model_file_path=None,
        cache_wavefunctions=False,
        process_pool=None
):
    """Train the normalizing flow.

    The total number of epochs is defined globally in the TRAINING_N_EPOCHS
    variable.

    Parameters
    ----------
    dataset : data.TrajectoryDataset
        The dataset used for training.
    flow : torch.nn.Module
        The neural network to train.
    loss_func : type
        The class of the loss function to use. It will be initialized
        in this function.
    s_restraint_bounds : Tuple[float], optional
        A pair of lower and upper bounds for the CV. If given, a quadratic
        penalty term is added to the loss function to maintain the CV
        within these limits.
    cl_h_restraint : bool, optional
        If True, a restraint energy is added to penalize short distances
        between the Cl ion and H.
    f_h_restraint : bool, optional
        If True, a restraint energy is added to penalize short distances
        between the F ion and H.
    grad_cov_precond : bool, optional
        If True, use the gradient covariance preconditioner (the class
        is GradCovPrecond).
    damping : bool, optional
        If grad_cov_precond, this is the damping factor used for this
        second order optimization method.
    net_model_file_path : str, optional
        Where to save the final model.
    cache_wavefunctions : bool, optional
        Whether to cache the latest wavefunctions during training and
        use them to restart the energy/force calculation in the next
        epoch.
    process_pool : multiprocessing.Pool, optional
        A pool of processes to use to parallelize the MP2 potential
        calculations.

    Returns
    -------
    flow : torch.nn.Module
        The final model.

    """
    from torch.utils.data import DataLoader
    from modules.plumedwrapper import io as plumedio
    from modules.nets.precond import GradCovPreconditioner
    from modules.loss import CVLoss, VariationalLoss, BoltzmannKLDivLoss, MetaDKLDivLoss
    from modules.functions.qm.psi4 import potential_energy_psi4

    # Shortcut for units.
    positions_unit = global_unit_registry.angstrom
    energy_unit = global_unit_registry.kJ / global_unit_registry.mol

    # Simulation constants shortcut.
    kT = KT.to(energy_unit).magnitude

    # We may need these indices to compute the reweighting factor.
    get_columns_idx_func = dataset.trajectory.get_aux_attribute('plumed', 'get_column_idx')
    potential_aux_idx = dataset.trajectory.get_aux_attribute('plumed', 'get_column_idx')('ene')
    rbias_col_idx = get_columns_idx_func('metad.rbias')
    cv_col_idx = get_columns_idx_func('cv')

    # Initialize loss_func module.
    if issubclass(loss_func, CVLoss):
        # Make the histogram range large enough for the min and max value belong to a bin.
        cvs = read_all_plumed_rows(dataset, [cv_col_idx])
        cv_hist_edges = np.linspace(-0.001 + np.min(cvs), np.max(cvs) + 0.001, num=101)
        loss_func = loss_func(kT, cv_hist_edges)
    elif issubclass(loss_func, BoltzmannKLDivLoss):
        rbias = read_all_plumed_rows(dataset, [rbias_col_idx])
        loss_func = loss_func(kT, shift_metad_rbias=-float(np.mean(rbias)))
    else:
        loss_func = loss_func(kT)

    # Search for a previous net to restart. We save the intermediate
    # models and data in the same directory as the final net model.
    output_dir_path = os.path.dirname(net_model_file_path)
    restart_net_file_pattern = os.path.join(output_dir_path, NETS_SUBDIR, '*_net.pth')
    restart_net_file_paths = list(glob.glob(restart_net_file_pattern))
    if len(restart_net_file_paths) > 0:
        restart_net_file_paths.sort(key=epoch_batch_sort_key)
        last_net_file_path = restart_net_file_paths[-1]
        last_epoch_idx, last_batch_idx = epoch_batch_sort_key(last_net_file_path)
        flow.load_state_dict(torch.load(last_net_file_path))
    else:
        last_epoch_idx = 0
        last_batch_idx = -1

    # Check if we have already completed training.
    if (last_epoch_idx+1 >= TRAINING_N_EPOCHS) and (net_model_file_path is not None) and (os.path.exists(net_model_file_path)):
        flow.load_state_dict(torch.load(net_model_file_path))
        return flow

    # This is used all the losses in time.
    all_loss_file_path = os.path.join(output_dir_path, LOSS_FILE_NAME)
    n_batches_per_epoch = len(dataset) / TRAINING_BATCH_SIZE
    if not DROP_LAST:
        n_batches_per_epoch = np.ceil(n_batches_per_epoch)
    n_batches_per_epoch = int(n_batches_per_epoch)

    all_loss = np.zeros(shape=(TRAINING_N_EPOCHS, n_batches_per_epoch))
    if os.path.isfile(all_loss_file_path):
        tmp = np.load(all_loss_file_path)
        if tmp.shape == all_loss.shape:
            # Resuming.
            all_loss = tmp
        else:
            # Extending
            all_loss[:tmp.shape[0], :tmp.shape[1]] = tmp
        del tmp

    # Determine the batch idx to save to create a restart.
    batch_indices_to_save = frozenset(range(
        n_batches_per_epoch-1, -1, -n_batches_per_epoch//TRAINING_N_CHECKPOINTS_PER_EPOCH))

    # Allocate the dictionary used to save the training data.
    # rows: batch_index mp2_potential log_det_J reweighting_factors restraint_potentials
    saved_col_names = ['time', 'index', 'reference_potential', 'mp2_potential', 'log_det_J', 'restraint_potential']
    training_data = {col_name: None for col_name in saved_col_names}

    # Turn on the training flag for layers like batch norm/dropout.
    flow.train()

    # Otherwise train the model with a small amount of regularization.
    if OPTIMIZER_NAME == 'adam':
        optimizer = torch.optim.Adam(flow.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # optimizer = torch.optim.Adam(flow.parameters(), weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER_NAME == 'sgd':
        optimizer = torch.optim.SGD(flow.parameters(),lr=LEARNING_RATE,  weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)
    else:
        raise ValueError('Unrecognized optimizer')

    # Create scheduler.
    if (REDUCE_LR_ON_PLATEAU_FACTOR is not None and
               CYCLICAL_LEARNING_RATE_MIN is not None):
        raise ValueError('Only one between cyclical and reduce on plateau schedulers can be used.')

    if REDUCE_LR_ON_PLATEAU_FACTOR is not None:
        # Reduce LR if in the 3rd epoch the loss has not dropped.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2,
            factor=REDUCE_LR_ON_PLATEAU_FACTOR
        )
    elif CYCLICAL_LEARNING_RATE_MIN is not None:
        n_batches_per_step = n_batches_per_epoch * CYCLICAL_LEARNING_EPOCHS_PER_STEP
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=CYCLICAL_LEARNING_RATE_MIN,
            max_lr=LEARNING_RATE,
            step_size_up=n_batches_per_step, step_size_down=n_batches_per_step,
            # Adam doesn't have momentum parameter, SGD does.
            cycle_momentum='momentum' in optimizer.defaults
        )
    elif MULTIPLICATIVE_LR_GAMMA is not None:
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambda epoch: MULTIPLICATIVE_LR_GAMMA)
    else:
        scheduler = None

    # Load state of optimzer and scheduler if we are resuming.
    output_dir_path = os.path.dirname(net_model_file_path)
    optimizer_checkpoint_path = os.path.join(output_dir_path, OPTIMIZER_CHECKPOINT_FILE_NAME)
    if os.path.exists(optimizer_checkpoint_path):
        optimizer_checkpoint = torch.load(optimizer_checkpoint_path)
        optimizer.load_state_dict(optimizer_checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            try:
                scheduler.load_state_dict(optimizer_checkpoint['scheduler_state_dict'])
            except KeyError:
                # The scheduler has changed. This is likely the final annealing.
                pass

    # For the preconditioning.
    if grad_cov_precond:
        if damping is None:
            damping = WEIGHT_DECAY
        preconditioner = GradCovPreconditioner(flow, loss_type='mean', damping=damping)

    # Create main output directory and subdirectories for log.
    subdir_to_create = ['', NETS_SUBDIR]
    if cache_wavefunctions:
        subdir_to_create.append(WAVEFUNCTIONS_SUBDIR)
    for subdir in subdir_to_create:
        os.makedirs(os.path.join(output_dir_path, subdir), exist_ok=True)

    # Train the network.
    data_loader = DataLoader(dataset, batch_size=TRAINING_BATCH_SIZE,
                             shuffle=SHUFFLE, drop_last=DROP_LAST)
    for epoch_idx in range(last_epoch_idx, TRAINING_N_EPOCHS):
        for batch_idx, batch in enumerate(data_loader):

            # Discard the batches that we have already processed.
            if (epoch_idx == last_epoch_idx) and batch_idx <= last_batch_idx:
                continue

            # Determine the paths where to cache the wavefunctions.
            if not cache_wavefunctions:
                wavefunction_restart_file_paths = None
            else:
                wavefunction_restart_file_paths = [
                    f'wfn_{sample_idx}.npy' for sample_idx in batch['index']]

            # File paths for logging.
            if batch_idx in batch_indices_to_save:
                file_prefix = os.path.join(output_dir_path, '{subdir}', str(epoch_idx) + '_batch_' + str(batch_idx) + '_')
                batch_net_model_file_path = file_prefix.format(subdir=NETS_SUBDIR) + 'net.pth'
                batch_potentials_file_path = file_prefix.format(subdir=BATCH_DATA_SUBDIR) + 'batch_potentials.dat'
                batch_mapped_coordinates_file_path = file_prefix.format(subdir=BATCH_DATA_SUBDIR) + 'mapped_coordinates.pdb'

            # Map all configurations through the flow.
            batch_positions, log_det_J = flow(batch['positions'])

            # Save the mapped configurations in case the QM calculation explodes.
            # TODO: Avoid saving this for now since it can be very disk consuming.
            # if batch_idx in batch_indices_to_save:
            #     dataset.save(
            #         topology_file_path=SN2_VACUUM_PRMTOP_FILE_PATH,
            #         output_file_path=batch_mapped_coordinates_file_path,
            #         positions=batch_positions.detach().numpy(),
            #         indices=batch['index'],
            #         multiframe=True
            #     )

            # Compute the MP2 potential of the mapped configurations.
            mp2_potentials = potential_energy_psi4(
                batch_positions=batch_positions,
                positions_unit=positions_unit,
                energy_unit=energy_unit,
                wavefunction_restart_file_paths=wavefunction_restart_file_paths,
                processes=N_PROCESSES if process_pool is None else process_pool,
                **SN2_VACUUM_PSI4_KWARGS
            )

            # Add the PLUMED restraint potential energy to the MP2 potential.
            restraint_potentials = compute_restraints_potential_energy_sn2_vacuum(
                batch_positions, s_restraint_bounds=s_restraint_bounds,
                cl_h_restraint=cl_h_restraint, f_h_restraint=f_h_restraint)

            # Get the metad bias and cvs that may be used to compute the loss.
            reference_potentials = batch['plumed'][:, potential_aux_idx]
            rbias = batch['plumed'][:, rbias_col_idx]
            cvs = batch['plumed'][:, cv_col_idx]

            # Compute the loss.
            loss_kwargs = {
                'potentials1': reference_potentials,
                'potentials2': mp2_potentials + restraint_potentials,
                'log_det_J': log_det_J
            }
            if isinstance(loss_func, CVLoss):
                loss_kwargs['cvs'] = cvs
            if (isinstance(loss_func, BoltzmannKLDivLoss) or
                    isinstance(loss_func, MetaDKLDivLoss)):
                loss_kwargs['metad_rbias'] = rbias

            loss = loss_func(**loss_kwargs)

            # Backpropagation.
            optimizer.zero_grad()
            loss.backward()
            if grad_cov_precond:
                preconditioner.step()
            optimizer.step()

            # Update learning rate.
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss)
                else:
                    scheduler.step()

            # Save all the rest of the logged information.
            # Loss function.
            all_loss[epoch_idx][batch_idx] = loss

            if batch_idx in batch_indices_to_save:
                np.save(all_loss_file_path, all_loss)

                # Intermediate net.
                torch.save(flow.state_dict(), batch_net_model_file_path)

                # Optimizer and scheduler.
                optimizer_info = {'optimizer_state_dict': optimizer.state_dict()}
                if scheduler is not None:
                    optimizer_info['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(optimizer_info, optimizer_checkpoint_path)

                # Potential energy information.
                times = batch['plumed'][:, 0]
                saved_info = [times, batch['index'], reference_potentials, mp2_potentials, log_det_J, restraint_potentials]
                for data_idx, (col_name, values) in enumerate(zip(saved_col_names, saved_info)):
                    training_data[col_name] = values.detach().numpy()
                plumedio.write_table(training_data, file_path=batch_potentials_file_path)

    # Save the final version of the net separately.
    torch.save(flow.state_dict(), net_model_file_path)

    return flow


def test_normalizing_flow_sn2_vacuum(
        flow,
        test_dataset,
        dataset_name,
        output_dir_path,
        analyzed_epochs=None,
        job_id=None,
        process_pool=None,
        subsample_size=0,
        fixed_bootstrap_indices=None
):
    """
    Run the targeted FEP of the SN2 reaction in vacuum.

    The function runs the TFEP analysis on all the neural networks trained
    for the specified number of epochs. The analysis is run by calling
    compute_DF(). See the docs for that function for more info.

    Parameters
    ----------
    flow : torch.nn.Module
        The neural network.
    test_dataset : modules.data.TrajectoryDataset
        The dataset to analyze.
    dataset_name : str
        The name of this dataset. This is used to determine the name
        of the output subdirectory.
    output_dir_path : str
        The path where to save the normalizing flow models, and the
        reweighting data.
    analyzed_epochs : List[int], optional
        If given, only these NN trained after these epochs are used
        for the analysis. The epochs are 0-based so 0 correspond to
        the NN after 1 epoch of training. If not give, the NN at all
        epochs are tested.
    job_id : int, optional
        If passed, on the ``job_id``-th epoch in ``analyzed_epochs``
        will be analyzed by this job.
    process_pool : multiprocessing.Pool, optional
        A pool of processes to use to parallelize the MP2 potential
        calculations.
    subsample_size : int, optional
        The size of each bootstrap sample.
    fixed_bootstrap_indices
        The indices in the dataset to always include when bootstrapping.

    """
    # Determine directories where to save the reweighting information.
    reweighting_results_dir_path = os.path.join(
        output_dir_path, REWEIGHTING_RESULTS_SUBDIR)

    # Select which intermediate nets generated by the training we should tests for reweighting.
    nets_dir_path = os.path.join(output_dir_path, TRAINING_RESULTS_SUBDIR, NETS_SUBDIR)
    net_file_paths = list(glob.glob(os.path.join(nets_dir_path, '*_net.pth')))
    net_file_paths.sort(key=epoch_batch_sort_key)

    # Find the final net for each epoch.
    n_nets_per_epochs = len([x for x in net_file_paths if os.path.basename(x)[:2] == '0_'])
    net_file_paths = net_file_paths[n_nets_per_epochs-1::n_nets_per_epochs]

    # Create a subset of nets used for analysis.
    if analyzed_epochs is not None:
        analyzed_epochs = set(analyzed_epochs)
        net_file_paths = [n for i, n in enumerate(net_file_paths) if i in analyzed_epochs]

    # Check if this is an array job.
    if job_id is not None:
        net_file_paths = [net_file_paths[job_id]]
        print('Reweighting only with', net_file_paths[0], flush=True)

    # Turn off the training flag for layers like batch norm/dropout.
    flow.eval()

    for net_file_path in net_file_paths:
        # Determine various file names.
        prefix = os.path.basename(net_file_path).split('net')[0]  # E.g. "0_batch_12_"
        net_results_dir_path = os.path.join(reweighting_results_dir_path, prefix[:-1])
        mp2_potentials_file_path = os.path.join(net_results_dir_path, MP2_POTENTIALS_CACHE_FILE_NAME)
        det_dcv_file_path = os.path.join(net_results_dir_path, DET_DCV_CACHE_FILE_NAME)
        fes_dir_path = os.path.join(net_results_dir_path, dataset_name)
        fes_file_path = os.path.join(fes_dir_path, DELTA_FES_FILE_NAME)
        bootstrap_fes_file_path = os.path.join(fes_dir_path, BOOT_DELTA_FES_FILE_NAME)
        delta_f_file_path = os.path.join(fes_dir_path, DELTA_F_FILE_NAME)

        # Check if we need to recompute this fes.
        if (os.path.isfile(fes_file_path) and
                os.path.isfile(bootstrap_fes_file_path) and os.path.isfile(delta_f_file_path)):
            print("Skipping", net_file_path, flush=True)
            continue

        os.makedirs(fes_dir_path, exist_ok=True)

        # Load the model.
        flow.load_state_dict(torch.load(net_file_path))

        # Standard reweighting. We keep a single cache file for potentials
        # and save one FES file for each value of stride time.
        reweighting = ReweightingSn2Vacuum(
            datasets=test_dataset,
            n_bins=FES_GRID_N_BINS,
            temperature=TEMPERATURE,
            cv_bounds=FES_CV_BOUNDS,
            map=flow,
            compute_geometric_fes=COMPUTE_DF_BASINS_FROM_FES,
            n_bootstrap_cycles=N_BOOTSTRAP_CYCLES,
            subsample_size=subsample_size,
            fixed_bootstrap_indices=fixed_bootstrap_indices,
            fes_file_path=fes_file_path,
            bootstrap_fes_file_path=bootstrap_fes_file_path,
            potentials_file_paths=mp2_potentials_file_path,
            det_dcv_file_paths=det_dcv_file_path,
            potentials_batch_size=N_PROCESSES,
            potentials_write_interval=4,
            process_pool=process_pool
        )

        # Compute the difference in free energy.
        compute_DF(reweighting, delta_f_file_path)


def run_targeted_reweighting_sn2_vacuum(
        equilibration_time,
        train_stride_time,
        output_dir_path,
        flow_cls,
        loss_func,
        randomize_training_dataset=False,
        cv_range=None,
        cl_h_restraint=False,
        f_h_restraint=False,
        grad_cov_precond=False,
        damping=None,
        cache_wavefunctions=False,
        analyzed_epochs=None,
        job_id=None,
        process_pool=None,
        **kwargs
):
    """
    Run the targeted reweighting of the SN2 reaction in vacuum.

    The function performs the bootstrap analysis for all the datasets
    created by create_datasets() function.

    Parameters
    ----------
    equilibration_time : pint.Quantity
        The initial time of the simulation to discard.
    train_stride_time : pint.Quantity
        The stride time to use to subsample the trajectory for the
        training set. Increasingly larger test sets will be built
        with stride times smaller than this.
    output_dir_path : str
        The path of the directory where to save the result files.
    randomize_training_dataset : bool, optional
        If True, a training dataset of random samples will be generated.
        The total number of samples would be the same as that generated
        by subsampling the trajectory with stride time ``train_stride_time``.
    process_pool : multiprocessing.Pool, optional
        A pool of processes to use to parallelize the MP2 potential
        calculations.

    """
    training_results_dir_path = os.path.join(output_dir_path, TRAINING_RESULTS_SUBDIR)
    latest_net_model_file_path = os.path.join(training_results_dir_path, 'net_final.pth')

    # Open the dataset.
    with open_trajectory_sn2_vacuum() as trajectory_dataset:
        # Determine all the stride times used for the train/test datasets to evaluate.
        train_dataset, test_datasets, bootstrap_datasets = create_datasets(
            trajectory_dataset, equilibration_time, train_stride_time,
            randomize_training_dataset, cv_range, return_bootstrap_datasets=True
        )
        all_test_datasets = {**train_dataset, **test_datasets}

        # Load the network model or train it. If we pass a job_id, we skip
        # the call to the training and jump straight to run targeted reweighting
        # so that we can check the results using one of the intermediate nets
        # while training in parallel. The function test_normalizing_flow()
        # already loads the intermediate models so we don't need to load the
        # NN state dict here.
        train_dataset_name = list(train_dataset.keys())[0]
        train_dataset = train_dataset[train_dataset_name]
        flow = flow_cls.from_architecture(train_dataset.trajectory, cv_range=cv_range, **kwargs)
        if job_id is None or job_id == 'training':
            print('Running training routine', flush=True)
            flow = train_normalizing_flow_sn2_vacuum(
                dataset=train_dataset,
                flow=flow,
                loss_func=loss_func,
                s_restraint_bounds=cv_range,
                cl_h_restraint=cl_h_restraint,
                f_h_restraint=f_h_restraint,
                grad_cov_precond=grad_cov_precond,
                damping=damping,
                net_model_file_path=latest_net_model_file_path,
                cache_wavefunctions=cache_wavefunctions,
                process_pool=process_pool
            )

            if job_id == 'training':
                # Return and leave training for parallel array jobs.
                return
        else:
            print('Skipping training routine', flush=True)

        # Evaluate the model on all test datasets obtained with increasing striding time.
        for dataset_name, dataset in all_test_datasets.items():
            # We save the input information of both training and reweighting:
            # semi-empirical potential, and metadynamics rbias, CV, and
            # (optionally) coordinates to facilitate the analysis.
            dataset_info_dir_path = os.path.join(output_dir_path, DATASET_INFO_SUBDIR, dataset_name)
            save_dataset_input_information_sn2_vacuum(dataset, dataset_info_dir_path, save_coordinates=False)

            # Compute free energies.
            test_normalizing_flow_sn2_vacuum(
                flow,
                dataset,
                dataset_name=dataset_name,
                output_dir_path=output_dir_path,
                analyzed_epochs=analyzed_epochs,
                job_id=job_id,
                process_pool=process_pool
            )

        # Determine the indices of the training set that must be hold
        # fixed in the bootstrap_test dataset while bootstrapping.
        train_traj_indices_set = set(train_dataset.trajectory_indices)
        fixed_bootstrap_indices = [i for i, idx in enumerate(bootstrap_datasets['bootstrap_test'].trajectory_indices)
                                   if idx in train_traj_indices_set]

        # Now evaluate the model with a bootstrap analysis.
        train_stride_time_str = str(int(train_stride_time.to('fs').magnitude))
        for subsample_size in SUBSAMPLE_SIZES:

            # First run the bootstrap analysis without training samples.
            test_normalizing_flow_sn2_vacuum(
                flow,
                bootstrap_datasets['bootstrap_eval'],
                dataset_name='bootstrap_' + str(subsample_size) + '_test_' + train_stride_time_str,
                output_dir_path=output_dir_path,
                analyzed_epochs=analyzed_epochs,
                job_id=job_id,
                process_pool=process_pool,
                subsample_size=subsample_size
            )

            # Then analyze including the training set.
            if subsample_size > len(train_dataset):
                dataset = bootstrap_datasets['bootstrap_test']
                fixed_indices = fixed_bootstrap_indices
            else:
                dataset = train_dataset
                fixed_indices = None

            # First run the bootstrap analysis without training samples.
            test_normalizing_flow_sn2_vacuum(
                flow,
                dataset,
                dataset_name='bootstrap_' + str(subsample_size) + '_train',
                output_dir_path=output_dir_path,
                analyzed_epochs=analyzed_epochs,
                job_id=job_id,
                process_pool=process_pool,
                subsample_size=subsample_size,
                fixed_bootstrap_indices=fixed_indices
            )


# =============================================================================
# REWEIGHTING ANALYSIS
# =============================================================================

def _optimize_sn2_vacuum_geometry(initial_positions, npz_file_path=None):
    """Optimize the geometry with MP2.

    Parameters
    ----------
    initial_positions : Quantity
        Initial position for the optimization.
    npz_file_path : str
        If given, the optimized positions are saved in npz format at
        this path (in units of angstrom).

    Returns
    -------
    optimized_positions : Quantity or False
        Optimized positions or False if the optimization failed.

    """
    ureg = initial_positions._REGISTRY

    # Check if we are resuming.
    if npz_file_path is not None and os.path.isfile(npz_file_path):
        data = np.load(npz_file_path)
        optimized_positions = data['positions']
        optimized_energy = float(data['energy'])
        return optimized_positions, optimized_energy

    # Set psi4 global options.
    import psi4
    # psi4.set_options({'geom_maxiter': 1, **SN2_VACUUM_PSI4_KWARGS['psi4_global_options']})
    psi4.set_options({'geom_maxiter': 2000, **SN2_VACUUM_PSI4_KWARGS['psi4_global_options']})

    # Set the position of the molecule.
    molecule = copy.deepcopy(SN2_VACUUM_MOLECULE)
    molecule.geometry = initial_positions.to('angstrom')

    # Optimize the geometry.
    psi4_molecule = molecule.to_psi4(reorient=False, translate=False)
    try:
        optimized_energy = psi4.optimize('mp2', molecule=psi4_molecule)
    except Exception:
        if npz_file_path is not None:
            print('FAILED OPTIMIZATION! For frame {}'.format(npz_file_path), flush=True)
        return False, False

    # The molecule stores the optimized geometry in atomic units.
    optimized_positions = (psi4_molecule.geometry().to_array() * ureg.bohr).to('angstrom')
    optimized_energy = (optimized_energy * ureg.hartree * ureg.avogadro_constant).to('kJ/mol')

    # Save the positions.
    if npz_file_path is not None:
        np.savez_compressed(npz_file_path, positions=optimized_positions.magnitude,
                            energy=optimized_energy.magnitude)

    return optimized_positions.magnitude, optimized_energy.magnitude


def optimize_geometries(equilibration_time, stride_time, n_optimized_geometries,
                        output_dir_path=None, process_pool=None):
    """Optimize multiple geometries taken from the subsampled trajectory.

    For each metastable state, the optimized geometries, and the optimized
    energies are saved in two numpy arrays (and in pdb format) in the output
    path.

    Parameters
    ----------
    equilibration_time : pint.Quantity
        The initial equilibration time of the simulation to discard.
    train_stride_time : pint.Quantity
        The stride time to use to subsample the trajectory to obtain frames
        to optimize.
    n_optimized_geometries : int
        The number of geometries to optimize.
    output_dir_path : str, optional
        The path to the output directory where to save the output.
    process_pool : multiprocessing.Pool, optional
        A pool of processes to use to parallelize the MP2 optimization.

    """
    from modules.data import TrajectorySubsampler, TrajectorySubset
    from modules.functions.geometry import to_batch_atom_3_shape

    # Open the dataset.
    cv_ranges = {
        'basin1': (-1.5, -0.5),
        'basin2': (0.8, 1.5),
    }

    # Function to save positions.
    def _save_data(batch_positions, energies, prefix):
        unitless_positions = batch_positions.to('angstrom').magnitude
        unitless_energies = energies.to('kJ/mol').magnitude

        np.save(prefix + 'positions.npy', unitless_positions)
        np.save(prefix + 'energies.npy', unitless_energies)

        trajectory_subset.save(
            topology_file_path=SN2_VACUUM_PRMTOP_FILE_PATH,
            output_file_path=prefix + 'positions.pdb',
            positions=unitless_positions,
            multiframe=True
        )

    for basin_name, cv_range in cv_ranges.items():
        if output_dir_path is None:
            basin_dir_path = None
        else:
            basin_dir_path = os.path.join(output_dir_path, basin_name)
            os.makedirs(basin_dir_path, exist_ok=True)

        subsampler = TrajectorySubsampler(
            equilibration_time=equilibration_time,
            stride_time=stride_time,
            cv_range=cv_range
        )
        with open_trajectory_sn2_vacuum(subsampler, remove_unmatched_frames=True) as trajectory_dataset:
            # Take 24 samples for computing average distances.
            subset_indices = np.linspace(0, len(trajectory_dataset)-1, num=n_optimized_geometries, dtype=np.int)
            trajectory_subset = TrajectorySubset(
                trajectory_dataset,
                indices=subset_indices
            )

            # The trajectory might have in total less than n_optimized_geometries samples.
            print(basin_name, 'subset indices:', subset_indices, flush=True)
            print(basin_name, 'trajectory indices:', trajectory_subset.trajectory_indices, flush=True)
            print(basin_name, 'effective total number of samples:', len(trajectory_subset), flush=True)

            # Get original positions.
            original_positions = to_batch_atom_3_shape(np.array([s['positions'].detach().numpy()
                                  for s in trajectory_subset])) * global_unit_registry.angstrom

            # Get the paths where to save the optimized positions.
            optimized_positions_dir_path = os.path.join(basin_dir_path, 'optimized_frames')
            os.makedirs(optimized_positions_dir_path, exist_ok=True)
            npz_file_paths = [os.path.join(optimized_positions_dir_path, 'frame{}.npz'.format(i))
                                              for i in trajectory_subset.trajectory_indices]

            # Optimize the geometries.
            starmap_args = [args for args in zip(original_positions, npz_file_paths)]
            if process_pool is None:
                optimized_positions = [_optimize_sn2_vacuum_geometry(*args) for args in starmap_args]
            else:
                optimized_positions = process_pool.starmap(_optimize_sn2_vacuum_geometry, starmap_args)

            # optimized_positions is a list of tuples (positions, energy).
            optimized_positions, optimized_energies = zip(*optimized_positions)

            # Remove and log failed optimization.
            failed_indices = [i for i, x in enumerate(optimized_positions) if x is False]
            if len(failed_indices) > 0:
                print('THE OPTIMIZATION OF THE FOLLOWING TRAJECTORY FRAMES FAILED:', flush=True)
                print('\tfailed subset indices:', failed_indices, flush=True)
                print('\tfailed trajectory indices:', trajectory_subset.trajectory_indices[failed_indices], flush=True)

            optimized_indices = [i for i, x in enumerate(optimized_positions) if x is not False]
            trajectory_subset = TrajectorySubset(trajectory_subset, indices=optimized_indices)
            optimized_positions = [optimized_positions[i] for i in optimized_indices]
            original_positions = original_positions[optimized_indices]
            optimized_energies = [optimized_energies[i] for i in optimized_indices]

            # Convert list of array to array. _optimize_sn2_vacuum_geometry
            # returns the positions and energies in angstrom and kJ/mol.
            optimized_positions = np.array(optimized_positions) * global_unit_registry.angstrom
            optimized_energies = np.array(optimized_energies) * global_unit_registry.kJ / global_unit_registry.mol

            # Save original and optimized positions.
            _save_data(original_positions, optimized_energies, os.path.join(basin_dir_path, 'original_'))
            _save_data(optimized_positions, optimized_energies, os.path.join(basin_dir_path, 'optimized_'))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # ---------------------- #
    # Command line arguments #
    # ---------------------- #

    # The arguments passed through the command line overwrite the SLURM variables.
    import argparse
    parser = argparse.ArgumentParser()

    # Options for both standard and targeted reweighting.
    parser.add_argument('-d', '--amberdir', type=str, dest='amber_dir',
                        help='The name of the directory containing AMBER/PLUMED output files.')
    parser.add_argument('-o', '--outputdir', type=str, dest='output_dir',
                        help='The path to the output directory.')
    parser.add_argument('-t', '--stride', type=float, dest='stride',
                        help='The stride time (in femtoseconds) used to subsample the trajectory to create the training dataset.')
    parser.add_argument('-s', '--seed', type=int, dest='seed',
                        help='The random seed to generate the training dataset, bootstrap datasets and NN parameters. If not given 0 is used.')
    parser.add_argument('--cvrange', type=str, dest='cv_range', default='all',
                        help='One between "all", "basin1", "basin2". This controls whether all samples or only those in a particular metastable states are used for sampling.')
    parser.add_argument('--method', type=str, dest='method', default='trp',
                        help='One between "trp" and "tfep". This controls whether the CV-preserving condition is enforced.')
    parser.add_argument('--aepochs', type=str, dest='analyzed_epochs',
                        help='A list such as 3,4,5 with the index (1-based) of the analyzed epochs. The job ID refers to this list.')

    # Options controlling the NN architecture for targeted reweighting.
    parser.add_argument('-a', '--architecture', type=str, dest='architecture',
                        help='One of the pre-defined architectures supported by the SN2VacuumFlow and CVPreservingSN2VacuumFlow classes. '
                             'Supported values are: "split", "cheap", "expensive", "expensive2", "sossplit", "soscheap", "sosexpensive", and "splineexpensive2"')
    parser.add_argument('-y', '--maflayers', type=int, dest='n_maf_layers',
                        help='The number of MAF layers in the normalizing flow.')
    parser.add_argument('-u', '--batchnorm', dest='batch_norm', action='store_true',
                        help='If given, batch normalization is used between MAF layers.')
    parser.add_argument('-v', '--weightnorm', dest='weight_norm', action='store_true',
                        help='If given, weight normalization is used for the conditioners.')
    parser.add_argument('--gradcovprecond', dest='grad_cov_precond', action='store_true', default=False,
                        help='If given, the gradient is preconditioned with the inverse covariance matrix of the gradient.')

    # Training parameters for the NN in targeted reweighting.
    parser.add_argument('-l', '--loss', type=str, dest='loss_func_name', default='kl',
                        help='One between "kl", "boltzmannkl", or "metadkl" (ignored for standard reweighting)')
    parser.add_argument('--optimizer', type=str, dest='optimizer', default='adam',
                        help='One between "adam" and "sgd"')
    parser.add_argument('--lr', type=float, dest='learning_rate',
                        help='The learning rate of the optimizer. If cyclical LR is used, this is the maximum LR.')
    parser.add_argument('--rlronpf', type=float, dest='reduce_lr_on_plateau_factor',
                        help='The multiplicative factor used to reduce the learning rate on plateau. If not given, learning rate is constant.')
    parser.add_argument('--clrmin', type=float, dest='cyclical_learning_rate_min',
                        help='The minimum learning rate with cyclical LR is used. If this is not given, CLR is not used.')
    parser.add_argument('--clrepoch', type=int, dest='cyclical_learning_epochs_per_step',
                        help='Number of epochs in each step of the cycle (composed by an ascending and descending step).')
    parser.add_argument('--multlr', type=float, dest='multiplicative_lr_gamma',
                        help='The multiplicative factor used at each step to change the learning rate.')
    parser.add_argument('-b', '--batchsize', type=int, dest='batch_size',
                        help='The batch size used for training.')
    parser.add_argument('-e', '--nepochs', type=int, dest='n_epochs', help='Number of training epochs.')
    parser.add_argument('-w', '--weightdecay', type=float, dest='weight_decay',
                        help='The weight decay controlling the L2 regularization during training.')
    parser.add_argument('--damping', type=float, dest='damping',
                        help='The damping of the curvature information when using gradcovprecond.')
    parser.add_argument('-c', '--cachewavefunctions', dest='cache_wavefunctions', default=False,
                        action='store_true', help='Parallelize over batch samples rather than Psi4 threads.')

    parser.add_argument('--droplast', dest='drop_last', action='store_true',
                        help='Flag to drop last batch if shorter than batchsize. This is the default behavior.')
    parser.add_argument('--no-droplast', dest='drop_last', action='store_false',
                        help='Flag to avoid dropping last batch if shorter than batchsize.')
    parser.set_defaults(drop_last=True)

    # Options for parallelization.
    parser.add_argument('-m', '--memory', type=str, dest='memory',
                        help='Memory available to Psi4 (overrides SLURM_MEM_PER_CPU)')
    parser.add_argument('-n', '--ncpus', type=int, dest='n_cpus',
                        help='Number of cpus (overrides SLURM_CPUS_PER_TASK).')
    parser.add_argument('-p', '--multiprocessing', dest='multiprocessing', action='store_true',
                        help='Parallelize over batch samples with multiprocessing rather than with Psi4 threads.')
    parser.add_argument('-j', '--jobid', type=str, dest='job_id',
                        help='Job ID for targeted reweighting (overrides SLURM_ARRAY_TASK_ID).')

    args = parser.parse_args()

    # ------------------------- #
    # Path to AMBER/PUMED files #
    # ------------------------- #

    # Simulation output.
    SN2_VACUUM_AMBER_DIR_PATH = os.path.join(SN2_VACUUM_DIR_PATH, args.amber_dir)
    SN2_VACUUM_COLVAR_FILE_PATH = os.path.join(SN2_VACUUM_AMBER_DIR_PATH, 'colvar')
    SN2_VACUUM_HILLS_FILE_PATH = os.path.join(SN2_VACUUM_AMBER_DIR_PATH, 'hills')
    SN2_VACUUM_CRD_FILE_PATH = os.path.join(SN2_VACUUM_AMBER_DIR_PATH, 'sn2_vacuum.crd')

    # --------- #
    # Utilities #
    # --------- #

    # Remove Pint warning about change in behavior after NEP 18.
    # https://github.com/hgrecco/pint/pull/905
    import pint
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pint.Quantity([])

    # Subsampling parameters to compute the delta FES.
    if args.stride is not None:
        STRIDE_TIME = args.stride * global_unit_registry.femtoseconds
    else:
        STRIDE_TIME = None
    EQUILIBRATION_TIME = 5000000*global_unit_registry.femtoseconds

    # ---------------------------- #
    # Analysis of the metadynamics #
    # ---------------------------- #

    # # First compute the metadynamics FES (every 60000 depositions).
    # # This saves files in a 'fes' directory in the same folder as hills.
    # from modules.plumedwrapper import sumhills
    # sumhills.run_plumed_sum_hills(
    #     metad_file_path=SN2_VACUUM_HILLS_FILE_PATH,
    #     min_to_zero=True,
    #     stride=210000,
    #     cv_min=FES_CV_BOUNDS[0],
    #     cv_max=FES_CV_BOUNDS[1],
    #     n_bins=FES_GRID_N_BINS-1,  # Results in FES_GRID_N_BINS grid points.
    #     overwrite='warning'
    # )

    # ------------------------- #
    # Reweighting configuration #
    # ------------------------- #

    # Configure threads and memory, and direct PSI4 output to stdout.
    if args.n_cpus is None:
        n_cpus = os.getenv('SLURM_CPUS_PER_TASK')
        if n_cpus is not None:
            n_cpus = int(n_cpus)
    else:
        n_cpus = args.n_cpus

    if args.memory is None:
        memory = os.getenv('SLURM_MEM_PER_CPU')
        if memory is not None:
            # This is the memory for a single CPU. It should be
            # a little less than the total memory available.
            memory = int(memory) - 128
            if not args.multiprocessing:
                # The memory is the total memory available to all Psi4 threads.
                memory *= n_cpus
            memory = str(memory) + 'MiB'
    else:
        memory = args.memory

    from modules.functions.qm.psi4 import configure_psi4
    if args.multiprocessing:
        if n_cpus is None:
            raise ValueError('Flagged multiprocessing but no number of CPU.')
        configure_psi4(memory, 1, psi4_output_file_path=None)
        N_PROCESSES = n_cpus
    else:
        # The memory is the total memory available to all Psi4 threads.
        configure_psi4(memory, n_cpus, psi4_output_file_path=None)
        N_PROCESSES = 1

    # Make sure the training/reweighting split permutation
    # and the shuffling are reproducible.
    if args.seed is None:
        np.random.seed(0)
        torch.manual_seed(0)
        randomize_training_dataset = False
    else:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        randomize_training_dataset = True

    # Use double throughout since the NN is not the bottleneck in memory/speed.
    torch.set_default_dtype(torch.float64)

    # -------------------- #
    # Standard reweighting #
    # -------------------- #

    with create_global_process_pool(N_PROCESSES) as process_pool:
        run_merged_standard_reweighting_sn2_vacuum(
            EQUILIBRATION_TIME,
            output_dir_path=args.output_dir,
            train_stride_time=STRIDE_TIME,
            process_pool=process_pool
        )

    assert args.output_dir is not None

    # Compute the Delta FES.
    with create_global_process_pool(N_PROCESSES) as process_pool:
        run_standard_reweighting_sn2_vacuum(
            EQUILIBRATION_TIME,
            output_dir_path=args.output_dir,
            train_stride_time=STRIDE_TIME,
            process_pool=process_pool
        )
        analyze_bootstrap_standard_reweighting_sn2_vacuum(
            EQUILIBRATION_TIME,
            output_dir_path=args.output_dir,
            train_stride_time=STRIDE_TIME,
            process_pool=process_pool
        )

    # -------------------- #
    # Targeted reweighting #
    # -------------------- #

    # Fix the CV range to analyze.
    def _grid_n_bins(cv_range):
        ratio = FES_CV_BOUNDS[1] - FES_CV_BOUNDS[0]
        ratio = (cv_range[1] - cv_range[0]) / ratio
        return int(FES_GRID_N_BINS * ratio)

    if args.cv_range == 'basin1':
        cv_range = BASIN1_CV_BOUNDS
    elif args.cv_range == 'basin2':
        cv_range = BASIN2_CV_BOUNDS
    elif args.cv_range != 'all':
        cv_min, cv_max = args.cv_range.split(',')
        cv_range = (float(cv_min), float(cv_max))
    else:
        cv_range = None

    if args.cv_range != 'all':
        FES_GRID_N_BINS = _grid_n_bins(cv_range)
        FES_CV_BOUNDS = cv_range

    # Decide which quantities to compute.
    if args.method == 'tfep':
        COMPUTE_DFES = False
        COMPUTE_DF_BASINS_FROM_FES = False
        COMPUTE_DF_BASINS_FROM_H = {args.cv_range: cv_range}
    else:
        COMPUTE_DFES = True
        COMPUTE_DF_BASINS_FROM_FES = True
        COMPUTE_DF_BASINS_FROM_H = {}

    # Set the type of flow for targeted reference potential or free energy perturbation.
    flow_classes = {
        'trp': CVPreservingSN2VacuumFlow,
        'tfep': SN2VacuumFlow
    }
    flow_cls = flow_classes[args.method]

    # Set loss function for training.
    from modules.loss import (KLDivLoss, BoltzmannKLDivLoss, MetaDKLDivLoss,
                              CVBoltzmannKLDivLoss, CVMetaDKLDivLoss, CVAlphaDiv2Loss,
                              VariationalLoss)
    loss_functions = {
        'kl': KLDivLoss,
        'boltzmannkl': BoltzmannKLDivLoss,
        'metadkl': MetaDKLDivLoss
    }
    loss_func = loss_functions[args.loss_func_name]

    if args.batch_size is not None:
        DROP_LAST = args.drop_last
    if args.batch_size is not None:
        TRAINING_BATCH_SIZE = args.batch_size
    if args.n_epochs is not None:
        TRAINING_N_EPOCHS = args.n_epochs
    if args.optimizer is not None:
        OPTIMIZER_NAME = args.optimizer
    if args.learning_rate is not None:
        LEARNING_RATE = args.learning_rate
    if args.reduce_lr_on_plateau_factor is not None:
        REDUCE_LR_ON_PLATEAU_FACTOR = args.reduce_lr_on_plateau_factor
    if args.cyclical_learning_rate_min is not None:
        CYCLICAL_LEARNING_RATE_MIN = args.cyclical_learning_rate_min
    if args.cyclical_learning_epochs_per_step is not None:
        CYCLICAL_LEARNING_EPOCHS_PER_STEP = args.cyclical_learning_epochs_per_step
    if args.multiplicative_lr_gamma is not None:
        MULTIPLICATIVE_LR_GAMMA = args.multiplicative_lr_gamma
    if args.weight_decay is not None:
        WEIGHT_DECAY = args.weight_decay

    # Job-IDs start from one while we need the 0-based index.
    if args.job_id is None:
        job_id = os.getenv('SLURM_ARRAY_TASK_ID')
        if job_id is not None:
            job_id = int(job_id) - 1
    elif args.job_id == 'training':
        job_id = args.job_id
    else:
        job_id = int(args.job_id) - 1

    # Check if we need to analyze only a subset of epochs.
    if args.analyzed_epochs is None:
        analyzed_epochs = None
    else:
        analyzed_epochs = [int(x)-1 for x in args.analyzed_epochs.split(',')]

    flow_kwargs = {}
    if args.architecture is not None:
        flow_kwargs['architecture_name'] = args.architecture
    if args.n_maf_layers is not None:
        flow_kwargs['n_maf_layers'] = args.n_maf_layers
    if args.batch_norm is not None:
        flow_kwargs['batch_norm'] = args.batch_norm
    if args.weight_norm is not None:
        flow_kwargs['weight_norm'] = args.weight_norm

    with create_global_process_pool(N_PROCESSES) as process_pool:
        run_targeted_reweighting_sn2_vacuum(
            EQUILIBRATION_TIME,
            STRIDE_TIME,
            output_dir_path=args.output_dir,
            flow_cls=flow_cls,
            loss_func=loss_func,
            randomize_training_dataset=randomize_training_dataset,
            cv_range=cv_range,
            cl_h_restraint='-cl' in SN2_VACUUM_AMBER_DIR_PATH,
            f_h_restraint='-f' in SN2_VACUUM_AMBER_DIR_PATH,
            grad_cov_precond=args.grad_cov_precond,
            damping=args.damping,
            cache_wavefunctions=args.cache_wavefunctions,
            analyzed_epochs=analyzed_epochs,
            job_id=job_id,
            process_pool=process_pool,
            **flow_kwargs
        )

    # --------- #
    # Analysis  #
    # --------- #

    # Compute the distances.
    optimized_geom_dir_path = os.path.join(SN2_VACUUM_AMBER_DIR_PATH, 'psi4_optimization')
    with create_global_process_pool(N_PROCESSES) as process_pool:
        optimize_geometries(
            EQUILIBRATION_TIME, STRIDE_TIME, n_optimized_geometries=512,
            output_dir_path=optimized_geom_dir_path, process_pool=process_pool
        )
