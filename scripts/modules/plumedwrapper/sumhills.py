#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Python wrapper for the plumed sum_hills program.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import logging
import os
import subprocess

from . import _utils
from . import io as plumedio


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_plumed_sum_hills(
        metad_file_path,
        output_file_path=None,
        stride=None,
        min_to_zero=None,
        cv_min=None,
        cv_max=None,
        n_bins=None,
        overwrite=False,
        create_time_file=True
):
    """
    Run the plumed sum_hills command.

    Parameters
    ----------
    metad_file_path : str
        The path to the file defined in the 'METAD' action of the PLUMED
        script where the hills are saved. Corresponds to the '--hills'
        option in the ``plumed sum_hills`` program.
    output_file_path : str, optional
        The path to the output file with the free energy surface. By default,
        the function generates a fes.dat file in the same directory as
        the metad file or, if ``stride`` is passed, all the files for
        each stride are saved in a sub directory called 'fes'.
    min_to_zero : bool, optional
        If ``True``, the minimum free energy will be shifted to zero.
        The default is identical to the program.
    stride : int, optional
        If given, the free energy surface is estimated every ``stride``
        Gaussian depositions. The default is identical to the program.
    cv_min : float, optional
        The minimum value of the CV for the grid. It corresponds to the
        --min option. The default is identical to the program.
    cv_max : float, optional
        The maximum value of the CV for the grid. It corresponds to the
        --max option. The default is identical to the program.
    n_bins : int, optional
        The number of bins in the grid used to represent the free energy
        surface. The default is identical to the program.
    overwrite : bool or 'warning', optional
        If the output file exist, if ``True``, the output file is
        overwritten; If ``False`` an exception is raised; If the string
        'warning', the file is not overwritten, but only a warning is
        raised; If the string 'ignore', the output file is not
        overwritten and the function returns silently.
    create_time_file : bool, optional
        By default, when ``stride`` is passed, the function creates an
        extra 'fes_time.dat' file with a single column with the time of
        each 'fes.dat' file. Set this to ``False`` to turn off this feature.

    """
    # Check that plumed is installed.
    _utils.check_plumed_is_installed()

    # Handle default output path.
    if output_file_path is None:
        # Check if we need to create a subdirectory.
        output_dir_path = os.path.dirname(metad_file_path)
        if stride is None:
            output_file_path = os.path.join(output_dir_path, 'fes.dat')
        else:
            output_dir_path = os.path.join(output_dir_path, 'fes')
            os.makedirs(output_dir_path, exist_ok=True)

            # With stride, PLUMED will use the --outfile as a prefix.
            output_file_path = os.path.join(output_dir_path, 'fes_')

    # Check if we need to overwrite the file. If stride is set, we
    # check if the first fes has been computed.
    overwritten_file_path = output_file_path
    if stride is not None:
        overwritten_file_path += '0.dat'

    # Raise exception/warning if needed.
    if os.path.exists(overwritten_file_path) and overwrite is not True:
        if overwrite == 'warning':
            logger.warning(f'{output_file_path} already exist. Skipping the '
                            'calculation of the FES.')
        elif overwrite != 'ignore':
            raise RuntimeError(f'{output_file_path} already exist. Please '
                                'delete the file or set overwrite=True.')
        return

    # Check if we need to create a time file and read the number of depositions.
    # TODO: When the number of depositions is divisible by stride, it seems that
    # TODO: PLUMED creates twice the final FES for some reason, which result in
    # TODO: an extra fes_*.dat file. Ask about this behavior on the issue tracker.
    if stride is not None and create_time_file:
        # Read the time information.
        time_data = plumedio.read_table(metad_file_path, col_names=['time'])

        # Slice time array according to stride.
        time_data['time'] = time_data['time'][stride-1::stride]

        # Write time data.
        time_file_path = os.path.join(output_file_path + 'time.dat')
        plumedio.write_table(time_data, time_file_path)

    # Command to invoke.
    sum_hills_cmd = [
        'plumed', 'sum_hills',
        '--hills', metad_file_path,
    ]

    # Add optional commands.
    if output_file_path is not None:
        sum_hills_cmd.extend(['--outfile', output_file_path])
    if min_to_zero is not None:
        sum_hills_cmd.append('--mintozero')
    if stride is not None:
        sum_hills_cmd.extend(['--stride', str(stride)])
    if cv_min is not None:
        sum_hills_cmd.extend(['--min', str(cv_min)])
    if cv_max is not None:
        sum_hills_cmd.extend(['--max', str(cv_max)])
    if n_bins is not None:
        sum_hills_cmd.extend(['--bin', str(n_bins)])

    # Run plumed sum_hills.
    subprocess.check_output(sum_hills_cmd)

