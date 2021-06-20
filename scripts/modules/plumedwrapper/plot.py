#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Provide utility functions to read output files generated by PLUMED.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from matplotlib import pyplot as plt

from . import plumed_unit_registry


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def plot_trajectory(
        data,
        col_names=None,
        time_unit=None,
        stride=1,
        axes=None,
        plot_kwargs=None
):
    """Plot the trajectory of all the given columns in time.

    Parameters
    ----------
    data : Dict[str, numpy.ndarray]
        The named columns of the PLUMED output file. For the expected
        format, see ``plumedwrapper.io.read_columns``. There must be a
        'time' column in the data.
    col_names : str or List[str], optional
        A single name or a list of the column names to plot. If not given,
        all columns are plotted.
    time_unit : str, optional
        The unit of time (e.g., 'ps', or 'ns') of the time dimension
        used for plotting.
    stride : int, optional
        Only data points every ``stride`` entries are plotted. Default is 1.
    axes : matplotlib.pyplot.Axes, optional
        Optionally, an existing Axes object can be passed, otherwise
        this function will create a new figure.
    plot_kwargs : Dict, optional
        Other keyword arguments to pass to matplotlib.pyplot.plot.

    Returns
    -------
    axes : matplotlib.pyplot.Axes
        The Axes object use for plotting.

    """
    # Instantiate mutable defaults.
    if plot_kwargs is None:
        plot_kwargs = {}

    # Create a new Figure if no Axes is passed.
    if axes is None:
        fig, axes = plt.subplots()

    # If no column names are given, we plot all of them.
    if col_names is None:
        col_names = list(data.keys())
    elif isinstance(col_names, str):
        # Make sure col_names is a list.
        col_names = [col_names]

    # Convert time dimension. Plumed plot time in femtoseconds.
    if time_unit is None or time_unit == 'fs':
        time_unit = 'fs'
        time = data['time']
    else:
        time = (data['time'] * plumed_unit_registry.fs).to(time_unit).magnitude

    # Plot all the trajectories.
    for col_name in col_names:
        axes.plot(time[::stride], data[col_name][::stride],
                  label=col_name, **plot_kwargs)

    # Fix labels.
    axes.set_xlabel(f'simulation time [{time_unit}]')

    # If there are multiple trajectories, use a legend rather than a label.
    if len(col_names) == 1:
        axes.set_ylabel(col_names[0])
    else:
        axes.legend()

    # There's no point in making the x axis start from negative numbers.
    axes.set_xlim((0, time[-1]))

    return axes