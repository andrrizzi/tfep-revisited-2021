#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility functions for the nets package.
"""


# =============================================================================
# FUNCTIONS
# =============================================================================

def generate_block_sizes(n_features, blocks, shorten_last_block=False):
    """Divides the features into blocks.

    In case a constant block size is requested, the function can automatically
    make the last block smaller if the number of features is not divisible
    by the block size. The function also raises errors if it detects inconsistencies
    between the number of features and the blocks parameters.

    Parameters
    ----------
    n_features : int
        The number of features to be divided into blocks.
    blocks : int or List[int]
        The size of the blocks. If an integer, the features are divided
        into blocks of equal size (except eventually for the last block).
        If a list, it is interpreted as the return value, and the function
        simply checks that the block sizes divide exactly the number of
        features.
    shorten_last_block : bool, optional
        If ``blocks`` is an integer that is not a divisor of the number
        of features, this option controls whether the last block is
        shortened (``True``) or an exception is raised (``False``).
        Default is ``False``.

    Returns
    -------
    blocks : List[int]
        The features can be divided into ``len(blocks)`` blocks, with
        the i-th block having size ``blocks[i]``.

    """
    # If blocks is an int, divide in blocks of equal size.
    if isinstance(blocks, int):
        if n_features % blocks != 0 and not shorten_last_block:
            raise ValueError('The parameter "n_features" must be '
                             f'divisible by "blocks" ({blocks})')

        div, mod = divmod(n_features, blocks)
        blocks = [blocks] * div
        if mod != 0:
            blocks += [mod]
    elif n_features != sum(blocks):
        raise ValueError('The sum of the block sizes must be equal to '
                         f'"n_features" ({n_features}).')

    return blocks
