#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility functions for the plumedwrapper package.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import shutil


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_plumed_is_installed():
    """Check that the plumed executable can be found and raise an error otherwise.

    Raises
    ------
    RuntimeError
        If the executable could not be found.

    """
    if shutil.which('plumed') is None:
        raise RuntimeError(f'Cannot find the plumed program installed. Please check '
                            'that it is installed correctly and the $PLUMED_KERNEL is set.')
