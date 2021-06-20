#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
An ``AuxReader`` for MDAnalysis capable of reading PLUMED-generated files.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os

import MDAnalysis

from .io import read_table


# =============================================================================
# AUXILIARY READER
# =============================================================================

class PLUMEDReader(MDAnalysis.auxiliary.XVG.XVGReader):
    """Auxiliary reader for PLUMED files.

    The class extends the ``MDAnalysis.auxiliary.XVG.XVGReader`` class
    to allow filtering by column name, converting to MDAnalysis internal
    unit system, and solving a few recurring problems with the PLUMED
    output (e.g., when the record at time 0.0 is duplicated).

    Parameters
    ----------
    file_path : str
        The path to the PLUMED output file.
    col_names : List[str], optional
        A list of column names to read. These names correspond to those
        provided in the initial '#! FIELDS ...' header record. If not
        given, the function reads all the columns. Time is always read
        and it will always be moved as the first column.
    units : Dict[str, str]
        ``units[col_name]`` is the unit used by PLUMED (e.g. 'fs') to
        for values in column ``col_name``. This information is used to
        convert the data into the internal unit system used by MDAnalysis.
        If not given, no conversion is performed, which is equivalent
        to assume that the PLUMED output is given in the MDAnalysis unit
        system. Note that the default unit for length is different in
        PLUMED (nm) and MDAnalysis (Angstrom).
    **kwargs
        Other keyword arguments for ``XVGReader``.

    Attributes
    ----------
    col_names : Tuple[str]
        The order of the columns in the auxiliary information. Depending
        on if and where the 'time' column was given on initialization,
        this can be different than the input parameter.

    See Also
    --------
    MDAnalysis.auxiliary.XVG.XVGReader
        All other parameters, methods, and attributes exposed by ``PLUMEDReader``.

    """

    def __init__(self, file_path, col_names=None, units=None, **kwargs):
        self._auxdata = os.path.abspath(file_path)

        # If col_names is None, all of them are read, including 'time'.
        if col_names is not None:
            # We always read the 'time' column as it's necessary for the
            # AuxReader to work.
            if 'time' not in col_names:
                col_names.insert(0, 'time')

            # Make sure 'time' is always the first column.
            time_col_idx = col_names.index('time')
            if time_col_idx != 0:
                col_names[0], col_names[time_col_idx] = 'time', col_names[0]

        # We store the actual order of the column names that was used
        # to read the file so that the user can match the names of the
        # columns to the columns of the array.
        self.col_names = col_names

        self._auxdata_values = read_table(file_path, col_names=col_names, as_array=True)

        # In some cases, PLUMED at the beginning has a double entry
        # with time == 0.0 which cause AuxReader to crash.
        if self._auxdata_values[0][0] == self._auxdata_values[1][0]:
            self._auxdata_values = self._auxdata_values[1:]

        # Convert units.
        if units is not None:
            self._convert_units(col_names, units)

        self._n_steps = len(self._auxdata_values)

        # Skip call to XVGReader.__init__ which reads the file in without
        # making possible the features above and call directly XVGReader's
        # parent class.
        super(MDAnalysis.auxiliary.XVG.XVGReader, self).__init__(**kwargs)

    def _convert_units(self, col_names, units):
        """Convert all values in _auxdata_values to internal coordinate system."""
        for col_name, plumed_unit_name in units.items():
            conversion_factor = None
            # Try all possible conversion factors exposed by MDAnalysis.
            for unit_type, conversion_factors in MDAnalysis.units.conversion_factor.items():
                try:
                    conversion_factor = conversion_factors[plumed_unit_name]
                except KeyError:
                    pass

            if conversion_factor is None:
                raise ValueError(f'Cannot find a conversion factor for units {plumed_unit_name}')

            col_idx = col_names.index(col_name)
            self._auxdata_values[:,col_idx] = self._auxdata_values[:,col_idx] / conversion_factor

    def get_column_idx(self, col_name):
        """Return the column index associated to the column name."""
        return self.col_names.index(col_name)
