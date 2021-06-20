#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Functions to perform standard and targeted free energy perturbation.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import abc
import collections
import os

import numpy as np
import scipy.special


# =============================================================================
# LOW-LEVEL REWEIGHTING FACILITY
# =============================================================================

def _compute_bin_delta_f(minus_work, rbias, det_dcv=None):
    """Compute the Delta f(s) for a single bin given the list of work and MetaD bias values.

    If det_dcv is given, the contribution to the geometric Delta f(s) is computed as well.

    The free energy is returned in reduced units.
    """
    if len(minus_work) == 0:
        if det_dcv is None:
            return np.nan
        return np.nan, np.nan

    rbias_minus_work = rbias + minus_work
    logsumexp_rbias_minus_work = scipy.special.logsumexp(rbias_minus_work)
    delta_f = - (logsumexp_rbias_minus_work - scipy.special.logsumexp(rbias))

    if det_dcv is not None:
        log_mean_det_dcv = scipy.special.logsumexp(rbias_minus_work + np.log(det_dcv)) - logsumexp_rbias_minus_work
        return delta_f, log_mean_det_dcv
    return delta_f


def _find_integration_bin_indices(cvs, cv_bounds):
    """Find the indices of the CV bin used for the integration.

    We prefer integration bounds a bit larger than smaller since
    the wall of the basins should contribute less to the integral.
    """
    indices = np.array([
        np.searchsorted(cvs, cv_bounds[0], side='right')-1,
        np.searchsorted(cvs, cv_bounds[1], side='left')
    ])
    # If cv_bounds[0] is less than any value in cvs, indices[0] is -1.
    if indices[0] < 0:
        indices[0] = 0
    return indices


def _logtrapzexp(y, dx, axis=-1):
    """Compute the logarithm of the integral of the exponential of y using the trapezoidal rule.

    The function works only with uniformly-spaced grid (with spacing dx).

    """
    n_samples = y.shape[axis]

    # Make sure axis is in the positive format.
    if axis < 0:
        axis = len(y.shape) + axis

    # Build weights for the logsumexp to implement trapezoidal rule.
    shape = [1] * axis + [n_samples] + [1] * (len(y.shape) - axis - 1)
    weights = np.ones(shape)
    weights[:, 0] = 0.5
    weights[:, -1] = 0.5
    weights *= dx

    # Perform exponential integral.
    return scipy.special.logsumexp(y, axis=axis, b=weights)


def _moving_average_2d(arr, window_size):
    """Create moving averages of a set of bootstrapped arrays."""
    n_samples = arr.shape[-1]

    # Create a new array to store the result to avoid modifying input argument.
    avg_arr = np.empty_like(arr)

    # Pad the arrays symmetrically to contain boundary effects.
    padding = window_size // 2
    arr = np.concatenate(
        (np.flip(arr[:, :padding], axis=-1), arr, np.flip(arr[:, n_samples-padding:], axis=-1)),
        axis=-1
    )

    # Convolve each array. The 'valid' mode removes the padding.
    for i, x in enumerate(arr):
        avg_arr[i] = np.convolve(x, np.ones(window_size) / window_size, 'valid')
    return avg_arr


def compute_DF_metastable_states(
        cvs,
        f_s,
        basin1_bounds,
        basin2_bounds,
        geo_f_s=None,
        compute_barrier=True,
        smooth_window_size=0
):
    """Integrate f(s) in the two basin and return the free energy difference.

    If a geometric free energy surface is given, the free energy barrier is
    also calculated and returned. The transition state is determined as the
    CV corresponding to the maximum value of ``geo_f_s`` in the interval
    ``(basin1_bounds[1], basin2_bounds[0])``.

    Parameters
    ----------
    cvs : numpy.ndarray
        An array of CV values corresponding to the free energy data
        points in ``f_s``.
    f_s : numpy.ndarray
        Either a 1D of shape ``(m,)`` or a 2D array of shape ``(n, m)``
        where ``n`` is the number of free energy profiles and ``m`` is
        the lengths of the ``cvs`` argument. The free energies must be
        passed in reduced units (i.e., divided by kT).
    basin1_bounds : Tuple[float] or Tuple[int]
        A pair ``(lower_bound, upper_bound)`` representing the integration
        bounds. If floats, these are interpreted as values of the CV
        within which to integrate. If integers, these are interpreted
        as indices of ``cvs`` and ``f_s`` last dimension. Both
        indices are included in the integration.
    basin2_bounds : Tuple[float] or Tuple[int]
        Same as ``basin1_bounds`` but for the basin 2.
    geo_f_s : numpy.ndarray, optional
        The geometric free energy surface with the same shae as ``f_s``.
        If given the free energy barrier between basin1 and basin2 is
        computed.
    compute_barrier : bool, optional
        If ``True``, the (geometric) free energy barrier is computed as
        well.
    smooth_window_size : int, optional
        If greater than 0, a moving average with the given window size
        is applied to the free energy profile(s) (including the geometric
        FES) before integrating.

    Returns
    -------
    delta_f_data : Dict
        A dictionary with the following keys.

        'delta_f' : float or numpy.ndarray
            A single free energy difference between the two basins or an
            array of free energy differences of shape ``(n,)`` depending on
            the shape of ``f_s`` in reduced units.
        'f_basin1' : float or numpy.ndarray
            The free energy (or bootstrap free energies) of basin1.
        'f_basin2' : float or numpy.ndarray
            The free energy (or bootstrap free energies) of basin2.
        'f_barrier' : float or numpy.ndarray
            The free energy difference (or bootstrap free energy differences)
            between the transition state and basin1.
        'gf_barrier' : float or numpy.ndarray
            The difference (or bootstrap differences) between the geometric
            free energy of the transition state and the free energy of basin1.

    """
    # Transform CV bounds into indices.
    basin_indices = [basin1_bounds, basin2_bounds]
    for i, indices in enumerate(basin_indices):
        if not np.issubdtype(type(indices[0]), np.integer):
            basin_indices[i] = _find_integration_bin_indices(cvs, indices)

    # Make sure we are dealing with a 2D array to handle standard
    # and bootstrap estimates without too many if-else clauses.
    is_bootstrap = len(f_s.shape) == 2
    f_s = np.atleast_2d(f_s)
    all_fes = [f_s]
    if geo_f_s is not None:
        geo_f_s = np.atleast_2d(geo_f_s)
        all_fes.append(geo_f_s)

    # Remove the nans by linearly interpolating between points.
    # In each cycle, we interpolate a separate bootstrap free energy profile.
    for fes in all_fes:
        for bootstrap_idx, nan_mask in enumerate(np.isnan(fes)):
            nan_indices = nan_mask.nonzero()[0]
            if len(nan_indices) == 0:
                continue
            not_nan_indices = (~nan_mask).nonzero()[0]
            fes[bootstrap_idx, nan_indices] = np.interp(
                cvs[nan_indices],
                cvs[not_nan_indices],
                fes[bootstrap_idx, not_nan_indices])

    # Create a moving average of the free energy profile.
    if smooth_window_size > 0:
        f_s = _moving_average_2d(f_s, window_size=smooth_window_size)
        if geo_f_s is not None:
            geo_f_s = _moving_average_2d(geo_f_s, window_size=smooth_window_size)
        all_fes = [f_s, geo_f_s]

    # Compute the grid spacing used for integration.
    d_cv = cvs[1] - cvs[0]

    # Initialize return value.
    n_bootstrap_cycles = f_s.shape[0]
    delta_f_data = {'delta_f': np.zeros(n_bootstrap_cycles)}

    # Compute integrals.
    for basin_idx, (sign, indices) in enumerate(zip([-1, 1], basin_indices)):
        integrated_f_s = f_s[:, indices[0]:indices[1]+1]

        # This implements the trapezoidal integration rule for exponential integration.
        basin_f = - _logtrapzexp(-integrated_f_s, d_cv)
        delta_f_data['delta_f'] += sign * basin_f
        delta_f_data['f_basin' + str(basin_idx+1)] = basin_f

    # Compute the free energy barrier both with the standard and geometric FES.
    # TODO: Add the term connecting it to the actual Eyring free energy barrier (possibly in the outer function since we have no info about units here).
    if compute_barrier:
        for fes, prefix in zip(all_fes, ['', 'g']):
            transition_state_f = np.nanmax(fes[:, basin_indices[0][1]:basin_indices[1][0]], axis=1)
            delta_f_data[prefix + 'f_barrier'] = transition_state_f - delta_f_data['f_basin1']

    # Go back to 1D if there was no batch dimension.
    if not is_bootstrap:
        for k, v in delta_f_data.items():
            delta_f_data[k] = v[0]

    return delta_f_data


def fep(
        temperature,
        reference_potentials,
        target_potentials,
        metad_rbias,
        n_bootstrap_cycles=0,
        subsample_size=0,
        fixed_bootstrap_indices=None
):
    """Compute the total free energy difference between the reference and target hamiltonian.

    Parameters
    ----------
    temperature : pint.Quantity
        The temperature of the system.
    reference_potentials : pint.Quantity or numpy.ndarray
        reference_potentials[i] is the potential energy of the system
        at the i-th frame of the simulation in units of energy/mol.
    target_potentials : pint.Quantity or numpy.ndarray
        target_potentials[i] is the potential energy of the system at
        the i-th frame of the simulation re-evaluated at the target
        Hamiltonian in units of energy/mol.
    metad_rbias : pint.Quantity or numpy.ndarray
        metad_rbias[i] is the normalized bias (i.e. V(s,t) - c(t))
        at the i-th frame of the simulation with s=cvs[i] in units of
        energy/mol.
    n_bootstrap_cycles : int, optional
        The number of bootstrap cycles.
    subsample_size : int, optional
        If between 1 and len(reference_potentials), the data is subsampled
        to use a dataset of this size. The true prediction, is subsampled
        using a constant interval to keep the observations as less correlated
        as possible, while bootstrap samples are subsampled randomly.
    fixed_bootstrap_indices : numpy.ndarray or None, optional
        If given, these indices are always added to the bootstrap datasets
        and only the bootstrap is performed only on the indices that are
        not present in this list.

    Returns
    -------
    delta_f : pint.Quantity
        The free energy difference f_target - f_reference in units of
        energy/mol.
    bootstrap_ci : pint.Quantity, optional
        If ``n_bootstrap_cycles > 0``, this is a pair (lower_bound, upper_bound)
        defining the 95% confidence interval in units of energy/mol.
    bootstrap_mean : pint.Quantity, optional
        If ``n_bootstrap_cycles > 0``, this is the mean bootstrap statistic
        in units of energy/mol.
    bootstrap_median : pint.Quantity, optional
        If ``n_bootstrap_cycles > 0``, this is the median bootstrap statistic
        in units of energy/mol.

    """
    # All the trajectories must have the same number of samples.
    n_samples = len(reference_potentials)
    for traj in [target_potentials, metad_rbias]:
        if len(traj) != n_samples:
            raise ValueError("All trajectories must have the same length.")

    # Extract unit registry.
    ureg = temperature._REGISTRY

    # Convert all to consistent units.
    kT = (temperature * ureg.molar_gas_constant).to('kJ/mol')
    unitless_kT = kT.magnitude
    reference_potentials = reference_potentials.to('kJ/mol').magnitude
    target_potentials = target_potentials.to('kJ/mol').magnitude
    metad_rbias = metad_rbias.to('kJ/mol').magnitude

    # Compute and return the free energy.
    delta_f = _fep_helper(target_potentials, reference_potentials, metad_rbias,
                          unitless_kT, n_bootstrap_cycles, subsample_size, fixed_bootstrap_indices)

    # Add units.
    if n_bootstrap_cycles > 0:
        delta_f, bootstrap_ci, bootstrap_mean, bootstrap_median = delta_f  # Unpack
        return delta_f*kT, bootstrap_ci*kT, bootstrap_mean*kT, bootstrap_median*kT
    return delta_f*kT


def _fep_helper(target_potentials, reference_potentials, metad_rbias, kT,
                n_bootstrap_cycles=0, subsample_size=0, fixed_bootstrap_indices=None):
    """Same as fep() but it works with unitless arguments (and unitless return values)."""
    n_samples = len(target_potentials)

    # Compute the reduced work and weights.
    metad_rbias = metad_rbias / kT
    minus_work = (- target_potentials + reference_potentials) / kT

    # Compute and return the free energy.
    if 0 < subsample_size < n_samples:
        indices = np.linspace(0, n_samples-1, num=subsample_size, dtype=np.int)
        rbias = metad_rbias[indices]
        log_weights = rbias - scipy.special.logsumexp(rbias)
        delta_f = - scipy.special.logsumexp(minus_work[indices] + log_weights)
    else:
        log_weights = metad_rbias - scipy.special.logsumexp(metad_rbias)
        delta_f = - scipy.special.logsumexp(minus_work + log_weights)

    if n_bootstrap_cycles > 0:
        # Initialize bootstrap distribution of Df.
        boot_delta_f = np.empty(n_bootstrap_cycles)

        for boot_cycle_idx, bootstrap_indices in enumerate(_iterate_bootstrap_indices(
            n_bootstrap_cycles, n_samples, subsample_size, fixed_bootstrap_indices
        )):
            rbias = metad_rbias[bootstrap_indices]
            log_weights = rbias - scipy.special.logsumexp(rbias)
            boot_delta_f[boot_cycle_idx] = - scipy.special.logsumexp(minus_work[bootstrap_indices] + log_weights)

        # Compute the 95% bootstrap CI.
        bootstrap_ci = np.nanpercentile(boot_delta_f, [2.5, 97.5], axis=0)

        # Compute the mean and median bootstrap statistic.
        bootstrap_mean = np.mean(boot_delta_f, axis=0)
        bootstrap_median = np.median(boot_delta_f, axis=0)

        return delta_f, bootstrap_ci, bootstrap_mean, bootstrap_median
    return delta_f


def _iterate_bootstrap_indices(n_bootstrap_cycles, n_samples, subsample_size=0, fixed_indices=None):
    """Iterate over bootstrap indices.

    The bootstrap indices selection keeps the ``fixed_indices`` fixed
    (e.g., the samples in the training dataset), and then return the
    indices of a subset of size ``subsample_size`` from a total of
    ``n_samples``.

    """
    with_fixed_indices = fixed_indices is not None

    # Determine the number of samples in the bootstrap cycle.
    if 0 < subsample_size < n_samples:
        n_bootstrap_samples = subsample_size
    else:
        n_bootstrap_samples = n_samples

    # Determine the non-fixed indices.
    if with_fixed_indices:
        n_nonfixed_bootstrap_samples = n_bootstrap_samples - len(fixed_indices)
        assert n_nonfixed_bootstrap_samples > 0
        fixed_bootstrap_indices_set = set(fixed_indices)
        nonfixed_indices = np.array([i for i in range(n_samples) if i not in fixed_bootstrap_indices_set])

    # Iterate over bootstrap indices.
    for boot_idx in range(n_bootstrap_cycles):
        if with_fixed_indices:
            # First select the non fixed indices.
            bootstrap_indices = np.random.choice(nonfixed_indices, n_nonfixed_bootstrap_samples, replace=True)
            bootstrap_indices = np.concatenate([fixed_indices, bootstrap_indices])
        else:
            bootstrap_indices = np.random.choice(n_samples, n_bootstrap_samples, replace=True)
        yield bootstrap_indices


# =============================================================================
# LOW-LEVEL REWEIGHTING FACILITY
# =============================================================================

class MetadReweighting:
    """Perform reweighting from a low-level metadynamics ensemble to a high-level potential.

    This is applicable both to standard reweighting according to Piccini
    et al. [1], and to targeted reweighting if the ``reference_potentials``
    are the mapped potentials.

    Parameters
    ----------
    temperature : pint.Quantity
        The temperature of the system.
    cvs : List[float]
        cvs[i] is the value of the CV at the i-th frame of the simulation.
    reference_potentials : pint.Quantity
        reference_potentials[i] is the potential energy of the system
        at the i-th frame of the simulation in units of energy/mol.
    target_potentials : pint.Quantity
        target_potentials[i] is the potential energy of the system at
        the i-th frame of the simulation re-evaluated at the target
        Hamiltonian in units of energy/mol.
    metad_rbias : pint.Quantity
        metad_rbias[i] is the normalized bias (i.e. V(s,t) - c(t))
        at the i-th frame of the simulation with s=cvs[i] in units of
        energy/mol.
    n_bins : int
        The total number of bins used for the grid representing the FES.
        This discretization is always used for reweighting a sample to
        compute the normalized weights used for the averages, even when
        compute the free energy between two free energy basins.
    cv_bounds : Tuple[float], optional
        The bounds of the bins (cv_min, cv_max). If not given, the
        minimum and maximum values in ``cvs`` will be used. This
        discretization is always used for reweighting a sample to
        compute the normalized weights used for the averages, even when
        compute the free energy between two free energy basins.
    det_dcv : numpy.ndarray, optional
        det_dcv[i] is the determinant of the CV gradient matrix used
        to compute the geometric FES.
    n_bootstrap_cycles : int, optional
        The number of bootstrap samples used to compute confidence
        intervals. Default is 10000. Set to 0 to deactivate
        bootstrapping. If bootstrap confidence intervals were previously
        cached, this is ignored.
    subsample_size : int, optional
        If between 1 and len(reference_potentials), the data is subsampled
        to use a dataset of this size. The true prediction, is subsampled
        using a constant interval to keep the observations as less correlated
        as possible, while bootstrap samples are subsampled randomly.
    fixed_bootstrap_indices : numpy.ndarray or None, optional
        If given, these indices are always added to the bootstrap datasets
        and only the bootstrap is performed only on the indices that are
        not present in this list.
    fes_file_path : str, optional
        The file path where to cache the free energy surface in XVG
        format.
    bootstrap_fes_file_path : str, optional
        If given, the bootstrap-generated FES will be cached at this
        path in numpy npz format.
    process_pool : multiprocessing.Pool, optional
        Optionally, a pool of processes used to parallelize the calculation
        of the bootstrap delta FES.

    References
    ----------
    [1] Piccini G, Parrinello M. Accurate quantum chemical free energies
        at affordable cost. The journal of physical chemistry letters.
        2019 Jun 21;10(13):3727-31.

    """

    def __init__(
            self,
            temperature,
            cvs,
            reference_potentials,
            target_potentials,
            metad_rbias,
            n_bins,
            cv_bounds=None,
            det_dcv=None,
            n_bootstrap_cycles=10000,
            subsample_size=0,
            fixed_bootstrap_indices=None,
            fes_file_path=None,
            bootstrap_fes_file_path=None,
            process_pool=None
    ):
        # All the trajectories must have the same number of samples.
        n_samples = len(cvs)
        for traj in [reference_potentials, target_potentials, metad_rbias]:
            if len(traj) != n_samples:
                raise ValueError("All trajectories must have the same length.")

        self._temperature = temperature
        self._cvs = cvs
        self._n_bins = n_bins
        self._cv_bounds = cv_bounds
        self.n_bootstrap_cycles = n_bootstrap_cycles
        self.subsample_size = subsample_size
        self.fixed_bootstrap_indices = fixed_bootstrap_indices

        # Keep internally all data into consistent units (kJ/mol).
        self._reference_potentials = reference_potentials.to('kJ/mol').magnitude
        self._target_potentials = target_potentials.to('kJ/mol').magnitude
        self._metad_rbias = metad_rbias.to('kJ/mol').magnitude
        self._det_dcv = det_dcv

        # File paths to cache data.
        self.fes_file_path = fes_file_path
        self.bootstrap_fes_file_path = bootstrap_fes_file_path
        self.process_pool = process_pool

    @property
    def unit_registry(self):
        """The Pint UnitRegistry compatible with this object."""
        return self._temperature._REGISTRY

    @property
    def kT(self):
        """The thermal energy in units of energy/mol."""
        return self._temperature * self.unit_registry.molar_gas_constant

    @property
    def n_samples(self):
        """Total number of samples without accounting for cv_bounds."""
        return len(self._cvs)

    @property
    def cv_bounds(self):
        """Minimum and maximum values allowed for the cv."""
        if self._cv_bounds is None:
            cv_min, cv_max = np.min(self._cvs), np.max(self._cvs)
        else:
            cv_min, cv_max = self._cv_bounds
        return cv_min, cv_max

    @property
    def bins_centers(self):
        """These are the centers of the n_bins CV histogram bins."""
        cv_min, cv_max = self.cv_bounds  # Unpack.
        return np.linspace(cv_min, cv_max, self._n_bins)

    @property
    def _compute_gfes(self):
        """Whether the geometric FES must be computed."""
        return self._det_dcv is not None

    def fep(self, cv_range=None):
        """Compute the total difference in free energy between target and reference.

        Parameters
        ----------
        cv_range : Tuple[float]
            A pair (lower_bound, upper_bound) limiting the values of the
            CV that defines the particular metastable state for which
            the Delta f must be computed.

        Returns
        -------
        delta_f : pint.Quantity
            The free energy difference f_target - f_reference in units of
            energy/mol.
        bootstrap_ci : pint.Quantity, optional
            If ``self.n_bootstrap_cycles > 0``, this is a pair (lower_bound, upper_bound)
            defining the 95% confidence interval in units of energy/mol.
        bootstrap_mean : pint.Quantity, optional
            If ``self.n_bootstrap_cycles > 0``, this is the mean bootstrap statistic
            in units of energy/mol.
        bootstrap_median : pint.Quantity, optional
            If ``self.n_bootstrap_cycles > 0``, this is the median bootstrap statistic
            in units of energy/mol.
        """
        # Select all the data within cv_range.
        if cv_range is not None:
            samples_to_keep = (self._cvs >= cv_range[0]) & (self._cvs <= cv_range[1])
            target_potentials = self._target_potentials[samples_to_keep]
            reference_potentials = self._reference_potentials[samples_to_keep]
            metad_rbias = self._metad_rbias[samples_to_keep]
        else:
            target_potentials = self._target_potentials
            reference_potentials = self._reference_potentials
            metad_rbias = self._metad_rbias

        # Compute the free energy difference between the two Hamiltonians.
        delta_f = _fep_helper(target_potentials, reference_potentials,
                              metad_rbias, self._unitless_kT,
                              self.n_bootstrap_cycles, self.subsample_size,
                              self.fixed_bootstrap_indices)

        if self.n_bootstrap_cycles > 0:
            delta_f, bootstrap_ci, bootstrap_mean, bootstrap_median = delta_f  # Unpack
            return delta_f*self.kT, bootstrap_ci*self.kT, bootstrap_mean*self.kT, bootstrap_median*self.kT
        return delta_f*self.kT

    def reweight_fes(self):
        """Reweight the entire free energy profile.

        Returns
        -------
        delta_fes_data : Dict
            A dictionary with the following keys

            'cv' : numpy.ndarray[float]
                The bin centers of the CV histogram used to compute the FES.
            'delta_fes' : pint.Quantity
                The values of the Delta f for each bin center in kJ/mol.
            'bootstrap_delta_fes_ci_lb' : pint.Quantity
                If self.n_bootstrap_cycles > 0, the lower bound of the
                bootstrap CI for each bin center in kJ/mol.
            'bootstrap_delta_fes_ci_hb' : pint.Quantity
                If self.n_bootstrap_cycles > 0, the upper bound of the
                bootstrap CI for each bin center in kJ/mol.
            'delta_gfes' : pint.Quantity
                The values of the geometric Delta f for each bin center in kJ/mol.
            'bootstrap_delta_gfes_ci_lb' : pint.Quantity
                If self.n_bootstrap_cycles > 0, the lower bound of the
                bootstrap CI of the geometric FES for each bin center in kJ/mol.
            'bootstrap_delta_gfes_ci_hb' : pint.Quantity
                If self.n_bootstrap_cycles > 0, the upper bound of the
                bootstrap CI of the geometric FES for each bin center in kJ/mol.

        """
        from .plumedwrapper import io as plumedio

        sparse_histograms = None

        # Check if the cache exist or compute the FES from scratch.
        if self.fes_file_path is not None and os.path.exists(self.fes_file_path):
            update_cache = False
            delta_fes_data = plumedio.read_table(self.fes_file_path)
        else:
            update_cache = True

            # We might need to recycle the sparse histogram for the bootstrap calculations.
            sparse_histograms = self._build_sparse_histograms()

            # Compute the delta fes.
            delta_fes_data = {'cv': self.bins_centers}
            delta_fes_data.update(self._compute_delta_fes(sparse_histograms))

        # Check if we need to compute bootstrap CIs.
        if self.n_bootstrap_cycles > 0 and 'bootstrap_delta_fes_ci_lb' not in delta_fes_data:
            update_cache = True

            bootstrap_delta_fes_data = self._compute_bootstrap_delta_fes(
                sparse_histograms=sparse_histograms)

            # Compute 95%-percentile CI.
            for delta_fes_name, boostrap_delta_fes in bootstrap_delta_fes_data.items():
                bootstrap_ci = np.nanpercentile(boostrap_delta_fes, [2.5, 97.5], axis=0)
                delta_fes_data['bootstrap_' + delta_fes_name + '_ci_lb'] = bootstrap_ci[0]
                delta_fes_data['bootstrap_' + delta_fes_name + '_ci_hb'] = bootstrap_ci[1]

        # Update cached data.
        if self.fes_file_path is not None and update_cache:
            plumedio.write_table(delta_fes_data, file_path=self.fes_file_path)

        # Attach units before returning the value.
        energy_units = self.unit_registry.kJ / self.unit_registry.mol
        for k, v in delta_fes_data.items():
            if k == 'cv':
                continue
            delta_fes_data[k] = v * energy_units
        return delta_fes_data

    def reweight_DF_metastable_states(
            self,
            reference_f_s,
            basin1_cv_bounds,
            basin2_cv_bounds,
            smooth_window_size=0
    ):
        """Computes the difference in free energy between two basins.

        Compute and reweight the low-level potential data to obtain the
        difference in free energy between two ranges of CV at the high-level
        potential.

        Parameters
        ----------
        reference_f_s : pint.Quantity
            The reference free energy as a function of the CV evaluated
            for points ``self.bins_centers``.
        basin1_cv_bounds : List[float]
            The minimum and maximum CV used for the integration of the
            first basin.
        basin2_cv_bounds : List[float]
            The minimum and maximum CV used for the integration of the
            second basin.
        smooth_window_size : bool, optional
            If greater than 0, a moving average with the given window size
            is applied to the free energy profile(s) before integrating.

        Returns
        -------
        data : Dict
            A dictionary with the following keys:

            'delta_f' : pint.Quantity
                The difference in free energy between the two basins f_2 - f_1
                after integration.
            'f_basin1' : pint.Quantity
                The free energy of basin1.
            'f_basin2' : pint.Quantity
                The free energy of basin1.
            'f_barrier' : pint.Quantity
                The free energy difference between the transition state
                and basin1.
            'gf_barrier' : pint.Quantity, optional
                The between the the geometric free energy of the transition state
                and the free energy of basin1.
            'bootstrap_delta_f_ci' : List[pint.Quantity], optional
                A pair with the lower and upper bounds of the delta_f. This
                is returned only if bootstrapping is enabled.
            'bootstrap_f_basin1_ci' : List[pint.Quantity], optional
                The lower and upper bounds of f_basin1.
            'bootstrap_f_basin2_ci' : List[pint.Quantity], optional
                The lower and upper bounds of f_basin2.
            'bootstrap_f_barrier_ci' : List[pint.Quantity], optional
                The lower and upper bounds of f_barrier.
            'bootstrap_gf_barrier_ci' : List[pint.Quantity], optional
                The lower and upper bounds of gf_barrier.

        """
        # Compute the FES computed using all the data.
        delta_f_s_data = self.reweight_fes()
        cvs = delta_f_s_data['cv']
        delta_f_s = delta_f_s_data['delta_fes']

        # Transform everything in reduced units for compute_DF_metastable_states().
        reference_f_s = reference_f_s.to('kJ/mol').magnitude / self._unitless_kT
        delta_f_s = delta_f_s.to('kJ/mol').magnitude / self._unitless_kT

        # The reweighted free energy profile.
        f_s = reference_f_s + delta_f_s

        # Check if we need to compute the geometric free energy barrier as well.
        if self._compute_gfes:
            delta_geo_f_s = delta_f_s_data['delta_gfes'].to('kJ/mol').magnitude / self._unitless_kT
            geo_f_s = reference_f_s + delta_geo_f_s
        else:
            geo_f_s = None

        # Find the integration bounds corresponding to CV limits.
        basin1_integration_indices = _find_integration_bin_indices(
            cvs, basin1_cv_bounds)
        basin2_integration_indices = _find_integration_bin_indices(
            cvs, basin2_cv_bounds)

        # Compute the difference in free energy between basins using all the data.
        data = compute_DF_metastable_states(
            cvs, f_s, basin1_integration_indices, basin2_integration_indices,
            geo_f_s=geo_f_s, smooth_window_size=smooth_window_size)

        # Compute bootstrap confidence intervals for the difference in free energy.
        if self.n_bootstrap_cycles > 0:
            # Compute bootstrap free energy profile.
            bootstrap_delta_f_s_data = self._compute_bootstrap_delta_fes()
            bootstrap_f_s = reference_f_s + bootstrap_delta_f_s_data['delta_fes'] / self._unitless_kT

            if self._compute_gfes:
                geo_f_s = reference_f_s + bootstrap_delta_f_s_data['delta_gfes'] / self._unitless_kT

            bootstrap_delta_f_s_data = compute_DF_metastable_states(
                cvs, bootstrap_f_s, basin1_integration_indices, basin2_integration_indices,
                geo_f_s=geo_f_s, smooth_window_size=smooth_window_size)

            # Compute 95%-percentile bootstrap confidence interval and add units.
            for k, v in bootstrap_delta_f_s_data.items():
                data['bootstrap_' + k + '_ci'] = np.nanpercentile(v, [2.5, 97.5])
                data['bootstrap_' + k + '_mean'] = np.mean(v)
                data['bootstrap_' + k + '_median'] = np.median(v)

        # Convert reduced units to kJ/mol.
        energy_units = self._unitless_kT * self.unit_registry.kJ / self.unit_registry.mol
        for k, v in data.items():
            data[k] = v * energy_units
        return data

    @property
    def _unitless_kT(self):
        return self.kT.to('kJ/mol').magnitude

    def _compute_delta_fes(self, sparse_histograms=None):
        """Compute the reweighted Delta FES.

        Parameters
        ----------
        sparse_histograms : Tuple[scipy.sparse.csr_matrix], optional
            A pair of previously cached histogram in sparse matrix form
            binnin respectively the work and MetaD normalized bias values
            (in reduced units) by CV.

        Returns
        -------
        delta_fes_data : dict[str, numpy.ndarray]
            delta_fes_data['delta_fes'][i] is the value of Delta f(s) for the i-th
            CV bin in kJ/mol. If self._compute_gfes is True, the dictionary also
            has an array delta_fes_data['delta_gfes'][i] with the values of
            Delta f(s) - log<\nabla s> for the i-th CV bin in kJ/mol.

        """
        # Create the histograms of work/MetaD bias in sparse matrix form.
        if sparse_histograms is None:
            sparse_histograms = self._build_sparse_histograms()
        return self._compute_delta_fes_static(
            sparse_histograms, self._compute_gfes, self._unitless_kT, self.subsample_size)

    @staticmethod
    def _compute_delta_fes_static(sparse_histograms, compute_gfes, unitless_kT, subsample_size=0):
        """Static version of _compute_delta_fes that can be used with parallelization."""
        # Subsample the dataset.
        n_samples = len(sparse_histograms[0].data)
        if 0 < subsample_size < n_samples:
            indices = np.linspace(0, n_samples-1, num=subsample_size, dtype=np.int)
            sparse_histograms = [h[:, indices] for h in sparse_histograms]
            n_samples = len(indices)

        # Compute the FES. Each bin_values returned by iterator_by_cv_bin is
        # a w, r pair (or w, r, dcv triple if the det_dcv histogram is returned
        # as well) with the list of work values and MetaD normalized bias (and det(dcv))
        # for each CV bin.
        histograms = [np.split(h.data, h.indptr[1:-1]) for h in sparse_histograms]
        iterator_by_cv_bin = zip(*histograms)
        delta_fes = [_compute_bin_delta_f(*bin_values) for bin_values in iterator_by_cv_bin]

        delta_fes_data = {}

        # delta_fes might contains both the delta_fes and the contribution to the geometric fes.
        if compute_gfes:
            delta_fes, log_mean_det_dcv = [np.array(x) for x in zip(*delta_fes)]
            delta_fes_data['delta_gfes'] = delta_fes - log_mean_det_dcv
        else:
            delta_fes = np.array(delta_fes)
        delta_fes_data['delta_fes'] = delta_fes

        # Add units before returning.
        for k, v in delta_fes_data.items():
            delta_fes_data[k] = unitless_kT * delta_fes_data[k]
        return delta_fes_data

    def _compute_bootstrap_delta_fes(
            self,
            sparse_histograms=None
    ):
        """Compute many Delta FES from bootstrapping the trajectory samples.

        Parameters
        ----------
        sparse_histograms : Tuple[scipy.sparse.csr_matrix], optional
            A pair of previously cached histogram in sparse matrix form
            binnin respectively the work and MetaD normalized bias values
            (in reduced units) by CV.

        Returns
        -------
        bootstrap_delta_fes_data : dict[str, numpy.ndarray]
            bootstrap_delta_fes_data['delta_fes'][i][j] is the value of Delta f(s)
            for the i-th bootstrap sample and the j-th CV bin in kJ/mol.
            If self._compute_gfes is True, the dictionary also has an array
            bootstrap_delta_fes_data['delta_gfes'][i][j] with the value of
            Delta f(s) - ln<\nabla s> for the i-th bootstrap sample and the
            j-th CV bin in kJ/mol.

        """
        # Return previously cached data.
        if self.bootstrap_fes_file_path is not None and os.path.exists(self.bootstrap_fes_file_path):
            # The file was saved as a compressed archive.
            bootstrap_delta_fes_data = {k: v for k, v in np.load(self.bootstrap_fes_file_path).items()}
            if len(bootstrap_delta_fes_data['delta_fes']) != self.n_bootstrap_cycles:
                raise ValueError('Requested {} bootstrap cycles but found '
                                 'cached data with {} samples.'.format(
                    self.n_bootstrap_cycles, len(bootstrap_delta_fes_data['delta_fes'])))
            return bootstrap_delta_fes_data

        # Create the histograms of work/MetaD bias(/det_dcv) in sparse matrix form.
        if sparse_histograms is None:
            sparse_histograms = self._build_sparse_histograms()

        # Initialize returned value. If we need to compute the geometric FES
        # as well, we need an extra row.
        bootstrap_delta_fes_data = {'delta_fes': np.empty(shape=(self.n_bootstrap_cycles, self._n_bins))}
        if self._compute_gfes:
            bootstrap_delta_fes_data['delta_gfes'] = np.empty(shape=(self.n_bootstrap_cycles, self._n_bins))

        # Compute all bootstrap FES.
        if self.n_bootstrap_cycles > 0:
            if self.process_pool is not None:
                # Parallel implementation.
                starmap_args = [(sparse_histograms, self._compute_gfes, self._unitless_kT, self.subsample_size, self.fixed_bootstrap_indices)
                                for _ in range(self.n_bootstrap_cycles)]
                all_bootstrap_fes = self.process_pool.starmap(
                    self._compute_bootstrap_sample_delta_fes_static, starmap_args)

                for boot_idx, bootstrap_fes in enumerate(all_bootstrap_fes):
                    for k, v in bootstrap_fes.items():
                        bootstrap_delta_fes_data[k][boot_idx] = v
            else:
                # Serial implementation.
                for boot_idx in range(self.n_bootstrap_cycles):
                    if boot_idx+1 % 10 == 0:
                        print('\rRunning bootstrap cycle number', boot_idx+1, end='')
                    bootstrap_fes = self._compute_bootstrap_sample_delta_fes_static(
                        sparse_histograms, self._compute_gfes, self._unitless_kT, self.subsample_size, self.fixed_bootstrap_indices
                    )
                    for k, v in bootstrap_fes.items():
                        bootstrap_delta_fes_data[k][boot_idx] = v

        # Update cache.
        if self.bootstrap_fes_file_path is not None:
            os.makedirs(os.path.dirname(self.bootstrap_fes_file_path), exist_ok=True)
            np.savez_compressed(self.bootstrap_fes_file_path, **bootstrap_delta_fes_data)

        # Return all the bootstrap FES.
        return bootstrap_delta_fes_data

    @staticmethod
    def _compute_bootstrap_sample_delta_fes_static(
            sparse_histograms, compute_gfes, unitless_kT, subsample_size, fixed_bootstrap_indices
    ):
        """Compute a the delta FES for a single bootstrap sample."""
        # Select a subset of samples.
        # bootstrap_sample_indices = np.random.choice(n_samples, n_bootstrap_samples, replace=True)
        bootstrap_sample_indices = next(_iterate_bootstrap_indices(
            n_bootstrap_cycles=1, n_samples=sparse_histograms[0].shape[1],
            subsample_size=subsample_size, fixed_indices=fixed_bootstrap_indices))
        histograms = [h[:, bootstrap_sample_indices] for h in sparse_histograms]

        # Compute the FES for this sample.
        return MetadReweighting._compute_delta_fes_static(
            histograms, compute_gfes, unitless_kT)

    def _build_sparse_histograms(self):
        """Create the histograms of reduced work/MetaD bias in sparse matrix form.

        If self._compute_gfes is True, the histogram of the determinant of the
        CV gradient is returned as well.

        This is a clever way to histogram and store the values among using
        a sparse matrix. See https://stackoverflow.com/a/26888164.

        """
        import scipy.sparse

        # Remove all indices that are outside the bounds.
        cv_min, cv_max = self.cv_bounds
        samples_to_keep = (self._cvs >= cv_min) & (self._cvs <= cv_max)
        cvs = self._cvs[samples_to_keep]
        n_samples = len(cvs)

        # Keep the work and MetaD normalized bias values in reduced units.
        kT = self._unitless_kT
        rbias_values = self._metad_rbias[samples_to_keep] / kT
        minus_work_values = (self._reference_potentials[samples_to_keep] - self._target_potentials[samples_to_keep]) / kT

        # cv_hist_indices[i] is the bin index for the i-th sample.
        # The first and last histogram bins are centered on cv_min/max.
        # d_cv is the difference between two bin centers.
        d_cv = (cv_max - cv_min) / (self._n_bins - 1)
        bin_min = cv_min - d_cv / 2
        bin_max = cv_max + d_cv / 2
        cv_hist_indices = ((cvs - bin_min) / (bin_max - bin_min) * self._n_bins).astype(int)

        # Check if we need to build a histogram of the determinant of the CV gradient matrix as well.
        all_values = [minus_work_values, rbias_values]
        if self._compute_gfes:
            all_values.append(self._det_dcv[samples_to_keep])

        # This is a clever way to histogram the values among bins using
        # a sparse matrix. See https://stackoverflow.com/a/26888164.
        sparse_hist = []
        for values in all_values:
            sparse_hist.append(scipy.sparse.csr_matrix(
                (values, [cv_hist_indices, np.arange(n_samples)]),
                shape=(self._n_bins, n_samples)))

        return sparse_hist


# =============================================================================
# STANDARD REWEIGHTING
# =============================================================================

class DatasetReweighting(abc.ABC, MetadReweighting):
    """Helper class to perform the standard/targeted reweighting from a dataset.

    This is an abstract class that requires the implementation of the
    methods ``get_traj_info`` and ``compute_potentials`` (see the docs
    of each method for more info).

    The class can cache the potential energies computed at the target
    Hamiltonian on disk in numpy format. This is used to restart in case the
    calculation is interrupted or if a different subset of the trajectory
    frames is used. See ``load_cached_potential`` for how to recover the
    computed potentials.

    Furthermore, the class supports merging independent dataset and perform
    the analysis as if it was one large dataset.

    Parameters
    ----------
    datasets : data.TrajectoryDataset or List[data.TrajectoryDataset]
        A trajectory dataset containing the frames to be used for
        reweighting as well as CV and metadynamics bias information.
        If a list of datasets, they will be concatenated and treated
        as a single dataset.
    n_bins : int
        The total number of bins in the for the grid representing the FES.
        The bounds of the grid are determined by min(cvs) and max(cvs).
    temperature : pint.Quantity
        The temperature of the system.
    cv_bounds : Tuple[float], optional
        The bounds of the bins (cv_min, cv_max). If not given, the minimum
        and maximum values in ``cvs`` will be used.
    map : Callable, optional
        A function taking a batch of positions as input and returning a
        pair ``( mapped_positions, log(det(Jacobian)) )``. This is optional
        and used to implement targeted reweighting.
    n_bootstrap_cycles : int, optional
        The number of bootstrap samples used to compute confidence intervals.
        Default is 10000. Set to 0 to deactivate bootstrapping. If bootstrap
        confidence intervals were previously cached, this is ignored.
    subsample_size : int, optional
        If between 1 and len(reference_potentials), the data is subsampled
        to use a dataset of this size. The true prediction, is subsampled
        using a constant interval to keep the observations as less correlated
        as possible, while bootstrap samples are subsampled randomly.
    fes_file_path : str, optional
        If given, the FES is saved in an XVG file at this path. The file
        has two columns: the CV and difference in free energy between the
        reference and target Hamiltonians (in kJ/mol).
    bootstrap_fes_file_path : str, optional
        If given, the bootstrap FES generated are cached at this path
        in numpy format.
    potentials_file_paths : str or List[str], optional
        If given, the potential at the target Hamiltonian are
        cached in a file at this path in numpy npz format as an array of
        shape ``(2, n_frames)``, where ``n_frames`` is the total number
        of frames in the trajectory (which is generally greater than the
        total number of samples in the dataset). The first row stores
        the time entries (in ps), and the second one the corresponding
        potential energies (in kJ/mol) computed at the target Hamiltonian.
        Frames for which no potential was computed (because not in the
        give dataset) have valid time entries but ``np.nan`` as potential
        energy. If dataset is a list, this must be a list of equal length.
    potentials_batch_size : int, optional
        The batch size used to compute the target potentials. This
        controls how many parallel instances of ``compute_potentials``
        are invoked.
    potentials_write_interval : int, optional
        The number of batches to compute before caching the results
        on disk (if ``potentials_file_path`` is given).
    process_pool : multiprocessing.Pool, optional
        Optionally, a pool of processes used to parallelize the calculation
        of the bootstrap delta FES.

    References
    ----------
    [1] Piccini G, Parrinello M. Accurate quantum chemical free energies
        at affordable cost. The journal of physical chemistry letters.
        2019 Jun 21;10(13):3727-31.

    """

    @staticmethod
    @abc.abstractmethod
    def get_traj_info(dataset):
        """Return the CV, reference potential, and MetaD normalized bias of the dataset samples.

        Parameters
        ----------
        dataset : data.TrajectoryDataset
            A single trajectory dataset containing the frames to be used for
            reweighting as well as CV and metadynamics bias information.

        Returns
        -------
        cvs : numpy.ndarray
            ``cvs[i]`` is the CV of the i-th sample in ``self.dataset``.
        reference_potentials : pint.Quantity
            ``reference_potentials[i]`` is the potential energy of the
            i-th sample in the ``self.dataset`` in units compatible with
            energy/mol.
        metad_rbias : pint.Quantity
            ``metad_rbias[i]`` is the metadynamics normalized bias
            (i.e., V(s, t) - c(t)) for the i-th sample in the ``self.dataset``
            in units compatible with energy/mol.

        """
        pass

    @abc.abstractmethod
    def compute_potentials(self, batch_positions):
        """Compute the potential at the target Hamiltonian for the given positions.

        Parameters
        ----------
        batch_positions : pint.Quantity
            An array of shape ``(batch_size, n_atoms, 3)`` with the
            positions for which to compute the potential energies.

        Returns
        -------
        batch_potentials : pint.Quantity
            An array of shape ``(batch_size,)`` with the potentials
            of each configuration in units compatible with energy/mol.

        """
        pass

    @abc.abstractmethod
    def compute_det_dcv(self, batch_positions):
        """Compute the determinant of the nxn matrix of the CV gradient used to compute the geometric FES.

        Parameters
        ----------
        batch_positions : np.array or torch.Tensor
            An array of shape ``(batch_size, n_atoms, 3)`` with the
            positions for which to compute the potential energies.

        Returns
        -------
        det_dcv : np.array
            An array of shape ``(batch_size,)`` with the determinants
            of the matrix D_ij = \grad s_i . \grad s_j. The units are
            implicitly given by batch_positions.

        """
        pass

    def __init__(
            self,
            datasets,
            n_bins,
            temperature,
            cv_bounds=None,
            map=None,
            compute_geometric_fes=False,
            n_bootstrap_cycles=10000,
            subsample_size=0,
            fixed_bootstrap_indices=None,
            fes_file_path=None,
            bootstrap_fes_file_path=None,
            potentials_file_paths=None,
            det_dcv_file_paths=None,
            potentials_batch_size=1,
            potentials_write_interval=1,
            process_pool=None
    ):
        # Make sure the dataset and potentials file paths are iterables.
        if not (isinstance(datasets, list) or isinstance(datasets, tuple)):
            datasets = [datasets]
            potentials_file_paths = [potentials_file_paths]
            det_dcv_file_paths = [det_dcv_file_paths]

        # These are needed by compute_dataset_potentials() and get_traj_info()
        self.datasets = datasets
        self.potentials_file_paths = potentials_file_paths
        self.det_dcv_file_paths = det_dcv_file_paths
        self._temperature = temperature
        self.map = map
        # process_pool is also initialized in super().__init__ but it's needed by functions before it.
        self.process_pool = process_pool

        # Compute the target potentials for each dataset.
        n_datasets = len(datasets)
        target_potentials = [None for _ in range(n_datasets)]
        cvs = [None for _ in range(n_datasets)]
        reference_potentials = [None for _ in range(n_datasets)]
        metad_rbias = [None for _ in range(n_datasets)]

        if compute_geometric_fes:
            det_dcv = [None for _ in range(n_datasets)]
        else:
            det_dcv = None

        for i, (dataset, potentials_file_path, det_dcv_file_path) in enumerate(zip(datasets, potentials_file_paths, det_dcv_file_paths)):
            # Compute potentials and determinant of the CV gradient matrix.
            temp = self.compute_dataset_potentials(
                dataset=dataset, potentials_file_path=potentials_file_path,
                det_dcv_file_path=det_dcv_file_path, return_det_dcv=compute_geometric_fes,
                batch_size=potentials_batch_size, write_interval=potentials_write_interval)
            if compute_geometric_fes:
                target_potentials[i], det_dcv[i] = temp
            else:
                target_potentials[i] = temp

            # Read other values necessary for reweighting.
            temp = self.get_traj_info(dataset)
            cvs[i] = temp[0]
            reference_potentials[i] = temp[1]
            metad_rbias[i] = temp[2]

        # Concatenate the data in a single dataset.
        if n_datasets == 1:
            cvs = cvs[0]
            target_potentials = target_potentials[0]
            reference_potentials = reference_potentials[0]
            metad_rbias = metad_rbias[0]
            if compute_geometric_fes:
                det_dcv = det_dcv[0]
        else:
            cvs = np.concatenate(cvs)

            # The potentials are lists of pint.Quantity so we need to maintain the units.
            energy_units = self.unit_registry.kJ / self.unit_registry.mol
            target_potentials = np.concatenate([x.to('kJ/mol').magnitude for x in target_potentials]) * energy_units
            reference_potentials = np.concatenate([x.to('kJ/mol').magnitude for x in reference_potentials]) * energy_units
            metad_rbias = np.concatenate([x.to('kJ/mol').magnitude for x in metad_rbias]) * energy_units
            if compute_geometric_fes:
                det_dcv = np.concatenate(det_dcv)

        # Initialize MetaDReweighting facility.
        super().__init__(
            temperature=temperature,
            cvs=cvs,
            reference_potentials=reference_potentials,
            target_potentials=target_potentials,
            metad_rbias=metad_rbias,
            n_bins=n_bins,
            cv_bounds=cv_bounds,
            det_dcv=det_dcv,
            n_bootstrap_cycles=n_bootstrap_cycles,
            subsample_size=subsample_size,
            fixed_bootstrap_indices=fixed_bootstrap_indices,
            fes_file_path=fes_file_path,
            bootstrap_fes_file_path=bootstrap_fes_file_path,
            process_pool=process_pool
        )

    def compute_dataset_potentials(
            self,
            dataset,
            potentials_file_path=None,
            det_dcv_file_path=None,
            return_det_dcv=False,
            batch_size=1, write_interval=1
    ):
        """
        Compute the MP2 potentials for the standard reweighting of the SN2 reaction in vacuum.

        Parameters
        ----------
        dataset : modules.data.TrajectoryDataset
            The dataset to analyze.
        potentials_file_path : str, optional
            Output file path where to save the energies at the MP2 level of
            theory used for reweighting. If not given, restarting won't be
            possible.
        det_dcv_file_path : str, optional
            Output file path where to save the determinants. If not given,
            restarting won't be possible.
        return_det_dcv : bool, optional
            If True, the det_dcv array is also returned.
        batch_size : int, optional
            The number of samples to be passed to the ``compute_potentials``
            function in each iteration. Depending on the function, this
            can enable the parallelization of the calculation over samples.
            Default is 1.
        write_interval : int, optional
            How many iterations should pass between two updates of the cache
            file on disk. Default is 1.


        Returns
        -------
        potentials : pint.Quantity
            ``potentials[i]`` is the potential energy of the i-th sample
            in the dataset.

        """
        import torch

        # First load the potentials.
        try:
            data = self.load_cached_data_from_file(potentials_file_path, as_tuple=False)
        except (FileNotFoundError, TypeError):
            # Initialize one potential for each (not subsampled) frame.
            if self.map is None:
                n_rows = 2
            else:
                # We need to cache also the log(det(J)) of the map.
                n_rows = 3
            data = np.full((n_rows, len(dataset.trajectory)), np.nan)

            # Create directory if it doesn't exist already.
            os.makedirs(os.path.dirname(potentials_file_path), exist_ok=True)
        else:
            # Check that the cached data is compatible with this dataset.
            assert len(dataset.trajectory) == data.shape[1]

        # Then the CV gradient matrix determinant.
        if return_det_dcv:
            try:
                det_dcv = self.load_cached_data_from_file(det_dcv_file_path, as_tuple=False)
            except (FileNotFoundError, TypeError):
                det_dcv = np.full(len(dataset.trajectory), np.nan)
            else:
                assert len(dataset.trajectory) == det_dcv.shape[0]

        # This might create a copy so we invoke the property only once.
        dataset_trajectory_indices = dataset.trajectory_indices

        # Slice the entries required for this dataset.
        log_det_J = 0
        if dataset_trajectory_indices is None:
            potentials = data[1]
            if self.map is not None:
                log_det_J = data[2]
        else:
            potentials = data[1, dataset_trajectory_indices]
            if self.map is not None:
                log_det_J = data[2, dataset_trajectory_indices]

        # This function add units to all the entries in the returned value.
        def _add_units_to_return_value():
            target_potentials = potentials - self._unitless_kT * log_det_J
            target_potentials = target_potentials * self.unit_registry.kJ / self.unit_registry.mol
            if return_det_dcv:
                return target_potentials, det_dcv[dataset_trajectory_indices]
            else:
                return target_potentials

        # Find the dataset indices that still needs to be computed.
        dataset_indices_to_compute = np.where(np.isnan(potentials))[0]
        if return_det_dcv:
            det_dcv_indices_to_compute = np.where(np.isnan(det_dcv[dataset_trajectory_indices]))[0]
            dataset_indices_to_compute = np.array(sorted(set(dataset_indices_to_compute).union(set(det_dcv_indices_to_compute))))

        # Check if we have already computed the potential for all frames.
        if len(dataset_indices_to_compute) == 0:
            return _add_units_to_return_value()

        # Initialize the arrays used to collect positions in batches.
        n_batches = int(np.ceil(len(dataset_indices_to_compute) / batch_size))
        batch_times = np.zeros(batch_size)
        batch_positions = np.zeros((batch_size, dataset.trajectory.n_atoms, 3))

        # Process the samples in batches so that we can trivially
        # parallelize the calculation of the potentials.
        for batch_idx in range(n_batches):
            # Determine the samples that belong to this batch
            start_sample_idx = batch_idx * batch_size
            end_sampled_idx = start_sample_idx + batch_size
            sample_indices = dataset_indices_to_compute[start_sample_idx:end_sampled_idx]

            try:
                batch_trajectory_indices = dataset_trajectory_indices[sample_indices]
            except TypeError:
                # The trajectory was not subsampled (trajectory_indices == None).
                batch_trajectory_indices = sample_indices

            # The last batch might be shorter.
            if len(sample_indices) < batch_size:
                batch_times = batch_times[:len(sample_indices)]
                batch_positions = batch_positions[:len(sample_indices)]

            # Collect times and positions.
            for i, sample_idx in enumerate(sample_indices):
                ts = dataset.get_ts(sample_idx)
                batch_positions[i] = ts.positions
                batch_times[i] = ts.time

            # Map the positions.
            if self.map is None:
                mapped_batch_positions = batch_positions
            else:
                # Convert batch positions to flat tensor.
                mapped_batch_positions = np.reshape(batch_positions, (batch_positions.shape[0], -1))
                mapped_batch_positions = torch.tensor(mapped_batch_positions, dtype=torch.get_default_dtype())

                # Compute mapped positions and log(det(J)).
                with torch.no_grad():
                    mapped_batch_positions, batch_log_det_J = self.map(mapped_batch_positions)

                mapped_batch_positions = mapped_batch_positions.detach().numpy()
                batch_log_det_J = batch_log_det_J.detach().numpy()

            # Check if we need to compute the potentials or only the det_dcv.
            if np.any(np.isnan(data[1, batch_trajectory_indices])):
                # Compute potentials.
                batch_potentials = self.compute_potentials(
                    batch_positions=mapped_batch_positions*self.unit_registry.angstrom)

                # Make sure the potential is in kJ/mol before storing it.
                batch_potentials = batch_potentials.to('kJ/mol').magnitude
                potentials[sample_indices] = batch_potentials
                if self.map is not None:
                    log_det_J[sample_indices] = batch_log_det_J

                # Update main result that we will cache.
                if potentials_file_path is not None:
                    data[0, batch_trajectory_indices] = batch_times
                    data[1, batch_trajectory_indices] = batch_potentials
                    if self.map is not None:
                        data[2, batch_trajectory_indices] = batch_log_det_J

                    # Update the cache on disk.
                    if (batch_idx + 1) % write_interval == 0:
                        np.savez_compressed(potentials_file_path, data)

            # Compute the determinant of the gradient matrix and store it.
            if return_det_dcv and np.any(np.isnan(det_dcv[batch_trajectory_indices])):
                det_dcv[batch_trajectory_indices] = self.compute_det_dcv(mapped_batch_positions)
                if (det_dcv_file_path is not None) and ((batch_idx + 1) % write_interval == 0):
                    np.savez_compressed(det_dcv_file_path, det_dcv)

        # Make sure we update the cache with the latest data even if
        # the number of batches is not divisible by write_interval.
        if (potentials_file_path is not None) and ((batch_idx + 1) % write_interval != 0):
            np.savez_compressed(potentials_file_path, data)
        if return_det_dcv and (det_dcv_file_path is not None) and ((batch_idx + 1) % write_interval != 0):
            np.savez_compressed(det_dcv_file_path, det_dcv)

        return _add_units_to_return_value()

    def load_cached_potentials(self, dataset, potentials_file_path, as_tuple=True):
        """Load the potentials associated to the dataset samples.

        Parameters
        ----------
        dataset : modules.data.TrajectoryDataset
            The dataset to analyze.
        potentials_file_path : str, optional
            Output file path where to save the energies at the MP2 level of
            theory used for reweighting. If not given, restarting won't be
            possible.
        as_tuple : bool, optional
            If ``True``, two separates arrays for times and potentials are
            returned instead of a single matrix.

        Returns
        -------
        data : numpy.ndarray
            An array of shape ``(2, n_frames)``, where ``data[0][i]`` and
            ``data[1][i]`` are respectively the simulation time (in ps) and
            potential energy (in kJ/mol) of the i-th configuration.

        """
        return self.load_cached_data_from_file(
            potentials_file_path, dataset.trajectory_indices, as_tuple=as_tuple)

    @staticmethod
    def load_cached_data_from_file(file_path, trajectory_indices=None, as_tuple=True):
        """Load the data (e.g., potential energies) cached in a file.

        Parameters
        ----------
        file_path : str
            The file where the potentials are stored.
        trajectory_indices : List[int], optional
            If given, only the potentials of the trajectory frames (at the
            time of writing) given by these indices are loaded. If not given,
            all frames are loaded.
        as_tuple : bool, optional
            If ``True``, two separates arrays for times and potentials are
            returned instead of a single matrix.

        Returns
        -------
        data : numpy.ndarray
            An array of shape ``(2, n_frames)``, where ``data[0][i]`` and
            ``data[1][i]`` are respectively the simulation time (in ps) and
            potential energy (in kJ/mol) of the i-th configuration.

        """
        # Load the archive file.
        data = np.load(open(file_path, 'rb'))['arr_0']

        # Subsample.
        if trajectory_indices is not None:
            data = data[:, trajectory_indices]

        if as_tuple:
            return tuple(*data)
        return data
