#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
This script applies the targeted reweighting method to a double-well potential
system.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.special

import torch

from modules.functions.restraints import lower_walls_plumed, upper_walls_plumed


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

RESULT_DIR = os.path.join('..', '')
PAPER_FIG_DIR_PATH = os.path.join('..', 'paper_figures', 'fig_toy_problem')

PAPER_COL_WIDTH = 3.375  # inches


# =============================================================================
# SQUARE ROOT OF A MATRIX
# =============================================================================

def sqrtm(input, inverse=False, epsilon=1e-6):
    """Matrix square root with SVD decomposition."""
    u, s, v = torch.svd(input)
    s = torch.sqrt(s)
    if inverse:
        s = 1 / (s + epsilon)
    return torch.mm(torch.mm(u, torch.diag(s)), v.t())


# =============================================================================
# HELPER MIX-IN
# =============================================================================

class DistrMixin:
    """A mix-in class augumenting the interface of a probability distribution class with potential/free energy methods."""

    @property
    def Z(self):
        """The value of the partion function."""
        return np.exp(-self.f)

    def prob(self, x):
        """The value of the probability."""
        return torch.exp(self.log_prob(x))

    def u(self, x):
        """The value of the potential energy."""
        return - self.log_prob(x) + self.f

    def generalized_work(self, samples, p2, map=None, p_sample=None, force_return_log_weights=False):
        """The instantaneous work of moving the samples from this distribution to p2.

        Parameters
        ----------
        samples : torch.Tensor
            A tensor with all the samples with size (n_samples, dim), where
            dim is the dimension of each sample.
        p2 : DistrMixin
            A probability distribution implementing this mixin interface.
        map : torch.nn.modules, optional
            A normalizing flow mapping the samples. If passed, the
            generalized work is evaluated instead of the standard one.
        p_sample : DistrMixin, optional
            A different distribution from where the samples comes from.
            If this is given the log weights associated to each work
            value is also returned. These can be used to reweight the
            free energy.
        force_return_log_weights : bool, optional
            If True, the log weights are returned even if p_sample is not
            passed as None.

        Returns
        -------
        work : torch.Tensor
            A tensor of size (n_samples,) with the work values for each sample.
        log_weights : torch.Tensor or None, optional
            If p_sample is given, this a tensor of size (n_samples,) with the
            log weights that can be used for reweighting. Otherwise, this is
            not returned or, if force_return_log_weights is True, it is None.

        """
        u_1 = self.u(samples)
        if map is None:
            log_det_J = 0.0
            u_2 = p2.u(samples)
        else:
            mapped_samples, log_det_J = map(samples)
            u_2 = p2.u(mapped_samples)
        work = u_2 - u_1 - log_det_J

        # Compute also the log weights for each work sample if p_sample is given.
        if p_sample is not None:
            u_sample = p_sample.u(samples)
            u_diff = u_sample - u_1
            log_weights = u_diff - scipy.special.logsumexp(u_diff)

            return work, log_weights

        if force_return_log_weights:
            return work, None
        return work

    def df(self, p2, input_bounds=None, cv_idx=0, *args, **kwargs):
        """Difference in free energy between this distribution and p2.

        Parameters
        ----------
        p2 : DistrMixin
            The target distribution.
        input_bounds : numpy.ndarray, optional
            Array of shape (sample_dim, 2), where input_bounds[i][0] and
            input_bounds[i][1] are the lower and upper bound that can be
            sampled for the i-th dimension. If given, only the Df restricted
            to the CV bounds are computed.
        cv_idx : int, optional
            The index of the CV bounds in ``input_bounds``.
        *args
        **kwargs
            Further arguments to pass to ``f_basin()``.

        Returns
        -------
        df : torch.Tensor
            The free energy difference.

        """
        if input_bounds is None:
            # Total free energy difference.
            return p2.f - self.f

        # Compute the free energy difference with the domain restricted to the given cv bounds.
        cv_bounds = input_bounds[cv_idx]
        cvs = torch.linspace(*cv_bounds)
        integral_limits = input_bounds[(cv_idx+1) % 2]
        f_self = self.f_basin(cvs, *args, cv_idx=cv_idx, int_lims=integral_limits, **kwargs)
        f_2 = p2.f_basin(cvs, *args, cv_idx=cv_idx, int_lims=integral_limits, **kwargs)
        return torch.tensor(f_2 - f_self)

    def tfep(self, p2, samples=1000, p_sample=None, input_bounds=None, map=None):
        """Run FEP or TFEP.

        Parameters
        ----------
        p2 : DistrMixin
            The target distribution.
        samples : int or torch.Tensor
            Wither a number of samples to sample from this distribution or
            pre-generated samples as a tensor.
        p_sample : DistrMixin, optional
            A different distribution from where the samples comes from.
            If this is given the log weights associated to each work
            value is used to reweight the free energy.
        input_bounds : numpy.ndarray, optional
            Array of shape (sample_dim, 2), where input_bounds[i][0] and
            input_bounds[i][1] are the lower and upper bound that can be
            sampled for the i-th dimension. If given, only samples within
            these bounds are computed.
        map : torch.nn.Module, optional
            The normalizing flow. If given, the TFEP estimator is used
            instead of standard FEP.

        Returns
        -------
        Df : torch.Tensor
            The free energy difference estimate.

        """
        samples = _process_samples(samples, self, p_sample, input_bounds=input_bounds)

        # Compute the generalized work.
        work, log_weights = self.generalized_work(
            samples, p2, map, p_sample=p_sample, force_return_log_weights=True)

        # Compute Df.
        return fep(work, log_weights=log_weights)

    def ps(self, s, cv_idx=0, int_lims=(-5, 5), n_int_nodes=200):
        """Returns the marginal probability p(s) as a function of the CV.

        Parameters
        ----------
        s : torch.Tensor
            The values of the CV at which to evaluate the marginal probability.
        cv_idx : int, optional
            The index of the degree of freedom that is considered the CV.
        int_lims : Tuple, optional
            The lower and upper integration bounds.
        n_int_nodes : int, optional
            The number of nodes used to perform the numerical integration.

        Returns
        -------
        ps : torch.Tensor
            The marginal probability with the same size as ``s``.

        """
        if self.event_shape != (2,):
            raise ValueError('Cannot evaluate marginal probabilities for >2D distributions.')

        # The x used for evaluating the probability and integrating.
        int_x = torch.linspace(int_lims[0], int_lims[1], n_int_nodes)
        prob_x = torch.empty((n_int_nodes, 2))
        not_cv_idx = (cv_idx + 1) % 2
        prob_x[:, not_cv_idx] = int_x

        # Compute the profile.
        ps = torch.empty_like(s)
        for i, ss in enumerate(s):
            prob_x[:, cv_idx] = ss
            ps[i] = torch.trapz(self.prob(prob_x), int_x)

        return ps

    def fs(self, *args, **kwargs):
        """Returns the free energy profile.

        Takes all the input parameters as ``self.ps()``.
        """
        return -torch.log(self.ps(*args, **kwargs)) - torch.log(self.Z)

    def f_basin(self, s, *args, **kwargs):
        """Computes the free energy of a basin.

        This simply computes the free energy profile along s and computes
        the total free energy by integration.

        Parameters
        ----------
        s : torch.Tensor
            The values of the CV at which to evaluate the free energy profile.
            The first and last value define the limits free energy basin.
            All s are assumed to be equidistant.
        *args
        **kwargs
            Values forwarded to ``self.fs()``.

        Returns
        -------
        df : torch.Tensor
            The free energy of the basin.

        """
        from modules.reweighting import _logtrapzexp
        fs = self.fs(s, *args, **kwargs)

        # Add fake batch dimension since _logtrapzexp expects batch.
        fs = fs.unsqueeze(0)

        fs = fs.detach().numpy()
        ds = (s[1] - s[0]).detach().numpy()

        return - _logtrapzexp(y=-fs, dx=ds)[0]

    def df_basins(self, basin_A_bounds, basin_B_bounds, *args, **kwargs):
        """Computes the difference in free energy between two basins.

        Parameters
        ----------
        basin_A_bounds : Tuple[float]
            The limits of the CV defining the first basin.
        basin_B_bounds : Tuple[float]
            The limits of the CV defining the second basin.
        *args
        **kwargs
            Further arguments to pass to ``self.f_basin()``.

        """
        cv_basin_A = torch.linspace(*basin_A_bounds)
        cv_basin_B = torch.linspace(*basin_B_bounds)
        f_A = self.f_basin(cv_basin_A, *args, **kwargs)
        f_B = self.f_basin(cv_basin_B, *args, **kwargs)
        return f_B - f_A

    def trp(self, p2=None, samples=1000, p_sample=None, work=None, log_weights=None, map=None, cv_idx=0, **kwargs):
        """Compute the free energy profile wih (targeted) free energy perturbation.

        Parameters
        ----------
        p2 : DistrMixin
            The target distribution.
        samples : int or torch.Tensor
            Either a number of samples to sample from this distribution or
            pre-generated samples as a tensor.
        p_sample : DistrMixin, optional
            A different distribution from where the samples comes from.
            If this is given the log weights associated to each work
            value is used to reweight the free energy.
        work : torch.Tensor
            Alternatively to ``p2``, one can directly passed pre-cached work
            values for each sample.
        log_weights : torch.Tensor
            Alternatively to ``p2``, one can directly passed pre-cached work
            and log_weights values for each sample.
        map : torch.nn.Module, optional
            The normalizing flow. If given, the TFEP estimator is used
            instead of standard FEP.
        cv_idx : int, optional
            The index of the degree of freedom that is considered the CV.
        **kwargs
            Further arguments to pass to ``rp()``.

        Returns
        -------
        cvs : torch.Tensor
        fes : torch.Tensor
            fes[i] is the value of the free energy at CV cvs[i].

        """
        if (p2 is None) == (work is None):
            raise ValueError('One and only one between p2 and work must be passed.')

        # Create samples.
        samples = _process_samples(samples, self, p_sample)

        # Compute the generalized work.
        if work is None:
            work, log_weights = self.generalized_work(
                samples, p2, map, p_sample=p_sample, force_return_log_weights=True)

        # Compute Df.
        cvs = samples[:, cv_idx]
        return rp(cvs, work, log_weights, **kwargs)

    def trp_df_basins(self, basin_A_bounds, basin_B_bounds, *args, ref_fs=None, **kwargs):
        """Compute the free energy difference between two basins from the FES.

        Parameters
        ----------
        basin_A_bounds : Tuple[float]
            The limits of the CV defining the first basin.
        basin_B_bounds : Tuple[float]
            The limits of the CV defining the second basin.
        ref_fs : torch.Tensor, optional
            A pre-cached reference free energy difference. If not passed,
            it is computed with ``self.fs()``.
        *args
        **kwargs
            Further arguments to pass to ``self.f_basin()``.

        Returns
        -------
        df : torch.Tensor
            The free energy difference.

        """
        cvs, delta_fes = self.trp(*args, **kwargs)
        if ref_fs is None:
            ref_fs = self.fs(cvs)
        fs = (ref_fs + delta_fes).detach().numpy()
        return df_basins(cvs.detach().numpy(), fs, basin_A_bounds, basin_B_bounds)

    def metropolis_step(self, x, u_x, max_displacement=0.2):
        """Takes a Metropolis step after proposing a random displacement with uniform probability."""
        displacement = torch.rand(*x.shape, dtype=x.dtype) * max_displacement
        y = x + displacement

        # Generate acceptance probabilities.
        u_y = self.u(y)
        p_accept = torch.exp(-u_y + u_x)
        p_accept = torch.min(p_accept, torch.tensor(1.0))

        # Accept/reject moves.
        thresholds = torch.rand(*p_accept.shape, dtype=p_accept.dtype)
        rejected_moves_indices = thresholds > p_accept
        y[rejected_moves_indices] = x[rejected_moves_indices]
        return y


# =============================================================================
# GAUSSIAN DISTRIBUTION WITH ANALYTICAL CAPABILITIES
# =============================================================================

class Gaussian(torch.distributions.MultivariateNormal, DistrMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._determinant_covariance = None
        self._f = None

    @property
    def determinant_covariance(self):
        """Determinant of the covariance matrix."""
        if self._determinant_covariance is None:
            self._determinant_covariance = torch.det(self.covariance_matrix)
        return self._determinant_covariance

    @property
    def f(self):
        """Free energy."""
        if self._f is None:
            Z = torch.rsqrt((self.determinant_covariance * (2.0*np.pi)**self.event_shape[0]))
            self._f = - torch.log(Z)
        return self._f

    def generate_perfect_mapping(self, p):
        """Generate the perfect Jarzynski mapping from this distribution to p."""
        inv_sqrt_cov1 = sqrtm(self.covariance_matrix, inverse=True)
        sqrt_cov2 = sqrtm(p.covariance_matrix)
        scale_t = torch.t(torch.mm(sqrt_cov2, inv_sqrt_cov1))
        log_det_J = torch.log(torch.sqrt(p.determinant_covariance / self.determinant_covariance))

        def _perfect_map(x):
            y = x - self.loc
            y = torch.addmm(p.loc, y, scale_t)
            return y, torch.full((y.shape[0],), log_det_J)

        return _perfect_map


# =============================================================================
# A GAUSSIAN MIXTURE WITH SEMI-ANALYTICAL RESULTS
# =============================================================================

class GaussianMixture(torch.distributions.MixtureSameFamily, DistrMixin):
    """A Gaussian mixture with an associated free energy.

    The free energy is arbitrary (and thus we know its exact value for reference)
    and it is passed in the constructor as a float.

    See Also
    --------
    torch.distributions.MixtureSameFamily

    """

    def __init__(self, p_mixture, means, covs, f=0):
        mix = torch.distributions.Categorical(torch.tensor(p_mixture))
        comp = torch.distributions.MultivariateNormal(torch.tensor(means), torch.tensor(covs))
        super().__init__(mix, comp)

        self.f = torch.tensor(f)

    def sample(self, sample_shape, input_bounds=None, flow=None):
        """Sample from the distribution, optionally constraining the CV between two values.

        Parameters
        ----------
        sample_shape : numpy.ndarray
            Array of shape (n_samples, sample_dim).
        input_bounds : numpy.ndarray, optional
            Array of shape (sample_dim, 2), where input_bounds[i][0] and
            input_bounds[i][1] are the lower and upper bound that can be
            sampled for the i-th dimension. If not given, no bounds are
            applied.
        flow : torch.nn.Module, optional
            If given and input_bounds is given as well, the input is sampled
            so that the mapped samples are also within the boundaries given
            by input_bounds.

        """
        samples = super().sample(sample_shape)

        # If samples must be within the some bound, keep sampling until
        # the shape is filled with samples within the requested region.
        if input_bounds is not None:
            outliers_indices = self._get_outlier_indices(samples, input_bounds, flow)
            while len(outliers_indices) > 0:
                temp_samples = super().sample((len(outliers_indices),))
                samples[outliers_indices] = temp_samples

                temp_outliers_indices = self._get_outlier_indices(temp_samples, input_bounds, flow)
                outliers_indices = outliers_indices[temp_outliers_indices]

        return samples

    @staticmethod
    def _get_outlier_indices(samples, input_bounds, flow):
        # Find all samples outside the boundaries.
        samples = samples.detach().numpy()
        lower_bounds = input_bounds[:, 0]
        upper_bounds = input_bounds[:, 1]

        # is_outside_boundary[i][d] is True if the d-th dimension
        # of the i-th sample is below/above the lower/upper bound.
        is_outside_boundary = (samples < lower_bounds) | (upper_bounds < samples)

        # outliers_indices[i] is the index of the i-th sample to discard.
        outliers_indices = np.where(np.bitwise_or(*is_outside_boundary.T))[0]

        # Find also all samples that are outside the boundaries once mapped.
        if flow is not None:
            mapped_samples = flow(samples)[0].detach().numpy()
            is_outside_boundary = (mapped_samples < lower_bounds) | (upper_bounds < mapped_samples)
            mapped_outliers_indices = np.where(np.bitwise_or(*is_outside_boundary.T))[0]

            if len(outliers_indices) > 0:
                return np.array(sorted(set(outliers_indices).union(set(mapped_outliers_indices))))
            return mapped_outliers_indices
        return outliers_indices


# =============================================================================
# UTILITY CLASS TO MODEL THE MAPPED PROBABILITY DISTRIBUTION
# =============================================================================

class PMapped:
    """Model the probability distribution ``p`` after being transformed by the
    change of variable implemented by ``flow``."""

    def __init__(self, p, flow):
        self.p = p
        self.flow = flow

    def log_prob(self, x):
        y, log_det_J = self.flow(x.reshape((x.shape[0]*x.shape[1], x.shape[2])))
        y = y.reshape((x.shape[0], x.shape[1], x.shape[2]))
        log_det_J = log_det_J.reshape((x.shape[0], x.shape[1]))
        return self.p.log_prob(y) + log_det_J

    def prob(self, x):
        return torch.exp(self.log_prob(x))


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class WorkLoss(torch.nn.Module):
    def __init__(self, p1, p2, p_sample=None):
        super().__init__()
        self.p1 = p1
        self.p2 = p2

    @staticmethod
    def mean(work, log_weights):
        if log_weights is None:
            return torch.mean(work)
        weights = torch.nn.functional.softmax(log_weights)
        return torch.sum(work * weights)


class MLLoss(WorkLoss):
    def __call__(self, work, log_weights=None):
        return self.mean(work, log_weights)

class KLPg(WorkLoss):
    def __call__(self, work, log_weights=None):
        return self.mean(work, log_weights) - self.p1.df(self.p2)


class KLPw(WorkLoss):
    def __call__(self, work, log_weights=None):
        return self.mean(work, log_weights) - fep(work)


# =============================================================================
# NORMALIZING FLOW
# =============================================================================

class ToyFlow(torch.nn.Module):
    """A normalizing flow used for the double-well potential example.

    Parameters
    ----------
    dimension : int, optional
        The dimension of the samples.
    dimension_conditioning : int, optional
        Controls how many degrees of freedom are not mapped but instead
        simply used to conditioning the mapping. For example, to preserve
        the first degree of freedom, simply set dimension_conditions=1
        in the constructor.
    n_maf_layers : int, optional
        Number of MAF layers in the network.
    transformer : torch.Module or List[torch.Module]
        Either 'affine' or 'neuralspline'.
    input_bounds : List[Tuple[float]]
        This is used in combination with the NeuralSplineTransformer, and
        it defines the limits of the DOF to be mapped.
    **maf_kwargs
        More keyword arguments to pass to the MAF object constructor.

    See Also
    --------
    from modules.nets.modules.flows import MAF

    """

    def __init__(
            self,
            dimension=2,
            dimension_conditioning=0,
            n_maf_layers=4,
            transformer='affine',
            input_bounds=None,
            **maf_kwargs
    ):
        from modules.nets.modules.flows import MAF, NormalizingFlow
        from modules.nets.modules.flows import NeuralSplineTransformer

        super().__init__()

        if transformer == 'affine':
            # The affine transformer is the default in MAF.
            transformer = None
        else:
            x0 = torch.tensor(input_bounds[dimension_conditioning:, 0])
            xf = torch.tensor(input_bounds[dimension_conditioning:, 1])
            transformer = NeuralSplineTransformer(x0, xf, n_bins=3)

        flows = []
        for maf_layer_idx in range(n_maf_layers):
            degrees_in = 'input' if (maf_layer_idx%2 == 0) else 'reversed'
            flows.append(MAF(
                dimension=dimension,
                dimension_conditioning=dimension_conditioning,
                degrees_in=degrees_in,
                weight_norm=False,
                transformer=transformer,
                **maf_kwargs
            ))
        self.flow = NormalizingFlow(*flows)

    def forward(self, x):
        return self.flow(x)

    def inv(self, x):
        return self.flow.inv(x)


def train_step(batch, flow, loss_func, optimizer, target_p, inverse=False, cv_bounds=None, cv_idx=0):
    """Perform a single step of training.

    Parameters
    ----------
    batch : torch.Tensor
        The batch data.
    flow : torch.nn.Module
        The normalizing flow to train.
    loss_func :
        The loss function.
    optimizer :
        The PyTorch optimizer (e.g., Adam).
    target_p : DistrMixin
        The target distribution.
    inverse : bool, optional
        Whether the flow must perform the inverse mapping.
    cv_bounds : Tuple[float], optional
        If given, a quadratic penalty is added to the work values
        to penalize escaping these bounds.
    cv_idx : int, optional
        The index of the degree of freedom that is considered the CV.

    Returns
    -------
    y : torch.Tensor
        The mapped samples.
    u_y : torch.Tensor
        The potential of the mapped samples at the target distribution.
    loss : torch.Tensor
        The value of the loss function for this batch.

    """
    # Unpack.
    try:
        x, u_x = batch
    except ValueError:
        x, u_x, log_weights = batch
    else:
        log_weights = None

    # Map the positions.
    if inverse:
        y, log_det_J = flow.inv(x)
    else:
        y, log_det_J = flow(x)

    # Compute the work.
    u_y = target_p.u(y)
    work = u_y - u_x - log_det_J

    # Add restraint on CV.
    if cv_bounds is not None:
        cvs = y[:, cv_idx]
        work += lower_walls_plumed(cvs, at=cv_bounds[0], kappa=1000.0)
        work += upper_walls_plumed(cvs, at=cv_bounds[1], kappa=1000.0)

    # Compute the loss.
    loss = loss_func(work, log_weights=log_weights)

    # Backpropagation.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return y, u_y, loss


def _sample_util(p, p_sample, n_samples, input_bounds=None):
    if p_sample is None:
        return p1.sample((n_samples,), input_bounds=input_bounds)
    return p_sample.sample((n_samples,), input_bounds=input_bounds)


def _process_samples(samples, p, p_sample=None, file_path=None, input_bounds=None):
    """If samples is an int, this will sample ``samples`` samples from p1 and cache it in file_path.

    Otherwise, samples is simply returned. If the file already exists,
    the samples are loaded from disk rather than being generated.
    """
    if isinstance(samples, int):
        # We generate and store on disk the samples.
        if file_path is not None and os.path.isfile(file_path):
            samples = torch.load(file_path)
        else:
            samples = _sample_util(p, p_sample, n_samples=samples, input_bounds=input_bounds)
            if file_path is not None:
                torch.save(samples, file_path)
    return samples


def _get_reference_fs(p, basin_A_bounds, basin_B_bounds, n_cv_bins, cv_idx=0, int_lims=(-5, 5)):
    """Return a free energy profile along the CV."""
    cv_bounds = (basin_A_bounds[0], basin_B_bounds[1])
    cvs = _get_cv_linspace(cv_bounds, n_cv_bins)
    return p.fs(cvs, cv_idx=cv_idx, int_lims=int_lims)


def train_flow(
        p1, p2, flow, train_samples=1000, p_sample=None, batch_size=256, n_epochs=20,
        loss_func=KLPg, lr=0.001, bidirectional=False, eval_samples=10000,
        return_df_traj=True, return_loss_traj=True, return_eval_stats=True,
        basin_A_bounds=None, basin_B_bounds=None, input_bounds=None, cv_idx=0
):
    """Train the normalizing flow saving some evaluation metrics in the process.

    The function returns the trained flow and a dictionary of metrics whose
    keywords (which can be configured from the function arguments) include

    'mean_loss_traj':
        The mean loss for each epoch computed as the mean of all the batch loss.
    'train_df_traj':
        The free energy evaluated at the end of each epoch on the training dataset.
    'eval_df_traj':
        The free energy evaluated at the end of each epoch on the evaluation dataset.
    'train_loss_traj':
        The total loss computed at the end of each epoch on the full training dataset.
    'eval_loss_traj':
        The total loss computed at the end of each epoch on the full evaluation dataset.

    By 'free energy' above, we mean either the total free energy between
    the target and reference distributions, or the free energy between
    two basins defined by ``basin_A_bounds`` and ``basin_B_bounds`` at the
    target level. This behavior is controlled by the flow. If the flow
    ``dimension_conditioning`` is 0 the first is computed. Otherwise,
    the function assumes the CV is preserved by the flow and it computes
    the free energy profile instead.

    Parameters
    ----------
    p1 : DistrMixin
        The reference distribution.
    p2 : DistrMixin
        The target distribution.
    flow : torch.nn.Module
        The normalizing flow to train.
    train_samples : int or torch.Tensor
        Either a number of samples to sample from this distribution or
        pre-generated samples as a tensor.
    p_sample: DistrMixin, optional
        A different distribution from where the samples comes from.
        If this is given the log weights associated to each work
        value is used to reweight the free energy.
    batch_size : int
        The batch size used for training.
    n_epochs : int
        The total number of training epochs.
    loss_func : torch.nn.Module
        The loss function.
    lr : float
        The learning rate.
    bidirectional : bool
        If True, the neural network is trained alternatively in the
        forward and backward direction (at each batch). The samples
        from p2 are sampled exactly unless ``bidirectional`` is
        'simulated', in which case the samples from p2 for the backward
        training are generated from a single step of Metropolis MC
        starting from the mapped samples from p1.
    eval_samples : int or torch.Tensor
        Either a number of samples to sample from this distribution or
        pre-generated samples as a tensor. These samples are used to
        evaluate the statistics on an independent dataset.
    return_df_traj : bool
        If True, 'train_df_traj' is added to the computed metrics. If
        ``return_eval_stats`` is also True, 'eval_df_traj' is also computed.
    return_loss_traj=True : bool
        If True, 'train_loss_traj' is added to the computed metrics. If
        ``return_eval_stats`` is also True, 'eval_loss_traj' is also computed.
    return_eval_stats : bool
        Controls whether 'eval_df_traj' and 'eval_loss_traj' are computed.
    basin_A_bounds : Tuple[float]
        The limits of the CV defining the first basin. Ignored if
        ``flow.dimension_conditioning`` is 0.
    basin_B_bounds : Tuple[float]
        The limits of the CV defining the second basin. Ignored if
        ``flow.dimension_conditioning`` is 0.
    input_bounds : numpy.ndarray, optional
        Array of shape (sample_dim, 2), where input_bounds[i][0] and
        input_bounds[i][1] are the lower and upper bound that can be
        sampled for the i-th dimension. If given, only samples within
        these bounds are computed.
    cv_idx : int, optional
        The index of the degree of freedom that is considered the CV.

    Returns
    -------
    flow : torch.nn.Module
        The trained normalizing flow.
    metrics : Dict
        The dictionary of evaluated metrics.

    """
    from torch.utils.data import TensorDataset, DataLoader

    # Check if this is a tfep or trp flow.
    is_tfep = flow.flow.flows[0].dimension_conditioning == 0
    if input_bounds is None:
        cv_bounds = None
    else:
        cv_bounds = input_bounds[cv_idx]

    # Check which method to use for bidirectionality.
    is_simulated_bidirectional = bidirectional == 'simulated'
    is_true_bidirectional = bidirectional is True
    bidirectional = is_true_bidirectional or is_simulated_bidirectional

    with torch.no_grad():
        # Create dataset of samples and associated energies.
        p1_samples = _process_samples(train_samples, p1, p_sample, input_bounds=input_bounds)
        p1_u1 = p1.u(p1_samples)
        if p_sample is None:
            p1_dataset = TensorDataset(torch.Tensor(p1_samples), torch.Tensor(p1_u1))
        else:
            u_diff = p_sample.u(p1_samples) - p1_u1
            p1_log_weights = u_diff - scipy.special.logsumexp(u_diff)
            p1_dataset = TensorDataset(torch.Tensor(p1_samples), torch.Tensor(p1_u1), torch.Tensor(p1_log_weights))
        p1_data_loader = DataLoader(p1_dataset, batch_size=batch_size, shuffle=True)

        # Build the reference free energy profile to compute the delta f between basins.
        if not is_tfep:
            N_CV_BINS = 400
            ref_fs = _get_reference_fs(p1, basin_A_bounds, basin_B_bounds, N_CV_BINS, cv_idx=cv_idx)

        # Create the dataset with p2 samples if we need bidirectional training.
        if is_true_bidirectional:
            p2_samples = p2.sample((len(p1_samples),), input_bounds=input_bounds)
            p2_u2 = p2.u(p2_samples)
            p2_dataset = TensorDataset(torch.Tensor(p2_samples), torch.Tensor(p2_u2))
            p2_data_loader = DataLoader(p2_dataset, batch_size=batch_size, shuffle=True)

        # Create data used for validation.
        if (return_df_traj or return_loss_traj) and return_eval_stats:
            p1_eval_samples = _process_samples(eval_samples, p1, p_sample, input_bounds=input_bounds)

    # Initialize flow and optimizer.
    flow.train()
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler = None

    # This function is used to assign and inizialize metrics  lazily.
    metrics = {'mean_loss_traj': torch.empty(n_epochs, dtype=p1_samples.dtype)}
    def assign_metric(name, epoch_idx, value):
        try:
            metrics[name][epoch_idx] = value
        except:
            metrics[name] = torch.empty(n_epochs, dtype=p1_samples.dtype)
            metrics[name][epoch_idx] = value

    for epoch_idx in range(n_epochs):
        if epoch_idx % 10 == 9:
            print('\rEpoch {}/{}'.format(epoch_idx+1, n_epochs), end='', flush=True)

        if is_true_bidirectional:
            p2_data_loader_iter = iter(p2_data_loader)

        for batch_idx, batch in enumerate(p1_data_loader):
            # Forward step.
            y, u_y, loss = train_step(batch, flow, loss_func, optimizer, target_p=p2,
                                      cv_bounds=cv_bounds, cv_idx=cv_idx)
            # Update loss traj.
            metrics['mean_loss_traj'][epoch_idx] += loss.detach()

            # Backward step.
            if bidirectional:
                if is_true_bidirectional:
                    batch = next(p2_data_loader_iter)
                else:
                    with torch.no_grad():
                        propagated_y = p2.metropolis_step(y, u_y)
                        propagated_y_u_2 = p2.u(propagated_y)
                        batch = (propagated_y, propagated_y_u_2)

                # Backward step.
                _, _, loss = train_step(batch, flow, loss_func, optimizer, target_p=p1, inverse=True,
                                        cv_bounds=cv_bounds, cv_idx=cv_idx)
                metrics['mean_loss_traj'][epoch_idx] += loss.detach()

        # Compute metrics.
        metrics['mean_loss_traj'][epoch_idx] /= batch_idx + 1
        if bidirectional:
            metrics['mean_loss_traj'][epoch_idx] /= 2

        with torch.no_grad():
            if return_df_traj or return_loss_traj:
                flow.eval()
                work, log_weights = p1.generalized_work(
                    p1_samples, p2, map=flow, p_sample=p_sample, force_return_log_weights=True)
                if return_eval_stats:
                    eval_work, eval_log_weights = p1.generalized_work(
                        p1_eval_samples, p2, map=flow, p_sample=p_sample, force_return_log_weights=True)
                flow.train()

            if return_df_traj:
                if is_tfep:
                    assign_metric('train_df_traj', epoch_idx, fep(work, log_weights))
                    if return_eval_stats:
                        assign_metric('eval_df_traj', epoch_idx, fep(eval_work, eval_log_weights))
                else:
                    # TRP.
                    df = p1.trp_df_basins(
                        basin_A_bounds, basin_B_bounds, samples=p1_samples, p_sample=p_sample,
                        ref_fs=ref_fs, work=work, log_weights=log_weights, cv_idx=cv_idx, n_bins=N_CV_BINS)
                    assign_metric('train_df_traj', epoch_idx, df)
                    if return_eval_stats:
                        df = p1.trp_df_basins(basin_A_bounds, basin_B_bounds, samples=p1_eval_samples, p_sample=p_sample,
                                              ref_fs=ref_fs, work=eval_work, log_weights=eval_log_weights, cv_idx=cv_idx, n_bins=N_CV_BINS)
                        assign_metric('eval_df_traj', epoch_idx, df)

            if return_loss_traj:
                assign_metric('train_loss_traj', epoch_idx, loss_func(work, log_weights))
                if return_eval_stats:
                    assign_metric('eval_loss_traj', epoch_idx, loss_func(eval_work, eval_log_weights))

        # Update learning rate for next epoch.
        if scheduler is not None:
            scheduler.step(metrics['mean_loss_traj'][-1])

    print(' ... DONE!', flush=True)

    # Return
    return flow, metrics


# =============================================================================
# FREE ENERGY DIFFERENCE
# =============================================================================

def fep(work, log_weights=None):
    """Estimate the free energy difference from the work and (optionally) the log weights.

    The log_weights are used to reweight the exponential average if
    the samples have been generated by another distribution than the
    reference (e.g., a biased distribution).

    """
    if log_weights is None:
        n_samples = len(work)
        log_n_samples = torch.log(torch.tensor(n_samples, dtype=work.dtype))
        return - torch.logsumexp(- work - log_n_samples, dim=0)
    return - torch.logsumexp(-work + log_weights, dim=0)


def _get_cv_linspace(cv_bounds, n_bins):
    cvs = torch.linspace(*cv_bounds, n_bins+1)
    return (cvs[:-1] + cvs[1:]) / 2


def rp(cvs, work, log_weights=None, cv_bounds=(-4, 4), n_bins=400):
    """Estimate the free energy profile from the work and (optionally) the log weights.

    The log_weights are used to reweight the exponential average if
    the samples have been generated by another distribution than the
    reference (e.g., a biased distribution).

    Parameters
    ----------
    cvs : torch.Tensor
        The values of the CV for each sample.
    work : torch.Tensor
        The work values for each sample.
    log_weights : torch.Tensor, optional
        The log_weights values for each sample.
    cv_bounds : Tuple[float], optional
        The limits over which to compute the FES.
    n_bins : int
        The number of bins for the CV histogram used to compute the FES.

    Returns
    -------
    cvs : torch.Tensor
        The values of the CV bins.
    delta_fes : torch.Tensor
        The value of the free energy for each CV bin.

    """
    delta_fes = torch.tensor(np.full(n_bins, np.nan), dtype=work.dtype)

    # Determine indices.
    cv_min, cv_max = cv_bounds
    bin_indices = np.array((cvs.detach().numpy() - cv_min) / (cv_max - cv_min) * (n_bins - 1), dtype=np.integer)

    # This is set to something != None inside the loop only if log_weights is given.
    bin_log_weights = None

    for bin_idx in range(n_bins):
        bin_work_indices = np.where(bin_indices == bin_idx)[0]
        if len(bin_work_indices) > 0:
            bin_work = work[bin_work_indices]
            if log_weights is not None:
                bin_log_weights = log_weights[bin_work_indices]
            delta_fes[bin_idx] = fep(bin_work, bin_log_weights)

    cvs = _get_cv_linspace(cv_bounds, n_bins)
    return cvs, delta_fes


def df_basins(s, fs, basin_A_bounds, basin_B_bounds):
    """Computes the difference in free energy between two basins.

    Parameters
    ----------
    s : numpy.ndarray
        An array of CV values corresponding to the free energy data
        points in ``f_s``.
    f_s : numpy.ndarray
        Either a 1D of shape ``(m,)`` or a 2D array of shape ``(n, m)``
        where ``n`` is the number of free energy profiles and ``m`` is
        the lengths of the ``cvs`` argument. The free energies must be
        passed in reduced units (i.e., divided by kT).
    basin_A_bounds : Tuple[float]
        The limits of the CV defining the first basin.
    basin_B_bounds : Tuple[float]
        The limits of the CV defining the second basin.

    Returns
    -------
    delta_f : float
        The free energy difference.

    """
    """Assumes that all s are equidistant."""
    from modules.reweighting import compute_DF_metastable_states
    return compute_DF_metastable_states(
        cvs=s,
        f_s=fs,
        basin1_bounds=basin_A_bounds,
        basin2_bounds=basin_B_bounds,
        compute_barrier=False
    )['delta_f']


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_2d_p(p, x_lims, y_lims, ax=None, p_diff=None, samples=None, cmap=None):
    if ax is None:
        fig, ax = plt.subplots()
    x, y = np.mgrid[x_lims[0]:x_lims[1]:.01, y_lims[0]:y_lims[1]:.01]
    pos = torch.tensor(np.dstack((x, y)))

    f = p.prob(pos)
    if p_diff:
        f = f - p_diff.prob(pos)

    ax.contourf(x, y, f.detach().numpy(), cmap=cmap)

    if samples is not None:
        samples = samples.detach().numpy()
        ax.scatter(samples[:, 0], samples[:, 1], color='black', s=1)
    return ax


def read_ml_df_data(n_train_samples, analytic_df, subdir_pattern, eval_file_path):
    """Read the DF and avg work evaluated on the training/eval set for the ML network.

    Parameters
    ----------
    n_train_samples : List[int]
        The list of number of training samples for each network (e.g.,
        [50, 100, 250, ..., 2500]).

    Returns
    -------
    data : Dict[str, Dict]
        A dict with 1 key-value per per metric. Each with the
        following subkeys: "repeats" (all repeat values), "mean" the
        mean value across repeats, "lb" and "hb" (the lowest and
        greatest value across repeats). For evaluation set data,
        "lb" and "hb" are instead 95% CIs.
    subdir_pattern : str
        Pattern for the directory holding the results for a specific
        number of training samples (e.g., "tfep-double-gaussian-klpg-train{}").
    eval_file_path : str
        Path to the "eval.npz" generated by compute_eval_results().

    """
    df_tfep_train_repeats = None

    # Collect data for all repeats.
    for i, n_samples in enumerate(n_train_samples):
        # Find all repeats.
        repeats_path_pattern = os.path.join(TOY_PROBLEM_DIR_PATH, subdir_pattern.format(n_samples), 'repeat*')
        repeats_dir_paths = list(glob.glob(repeats_path_pattern))
        n_nets = len(repeats_dir_paths)

        # Initialize data array (requires knowing n_repeats).
        if df_tfep_train_repeats is None:
            df_tfep_train_repeats = np.empty((n_nets, len(n_train_samples)))
            avg_w_repeats = np.empty((n_nets, len(n_train_samples)))

        for repeat_idx, repeat_dir_path in enumerate(repeats_dir_paths):
            # We assume, the ML map is the one with minimum loss.
            repeat_metrics = torch.load(os.path.join(repeat_dir_path, 'metrics.pt'))
            loss = repeat_metrics['train_loss_traj'].detach().numpy()
            ml_map_idx = np.argmin(loss)
            df_tfep_train_repeats[repeat_idx, i] = repeat_metrics['train_df_traj'][ml_map_idx].detach().numpy()
            # This assumes the KLPg loss was used.
            avg_w_repeats[repeat_idx, i] = loss[ml_map_idx] + analytic_df

    # Collect training and evaluation data.
    all_data = {
        'df_tfep_train': {'repeats': df_tfep_train_repeats},
        'avg_w': {'repeats': avg_w_repeats}
    }
    eval_data = np.load(eval_file_path)
    for k in eval_data:
        all_data[k] = {'repeats': eval_data[k]}

    # Compute mean and average statistics across the repeats.
    for data in all_data.values():
        df_repeats = data['repeats']
        df_mean = np.nanmean(df_repeats, axis=0)
        df_std = np.nanstd(df_repeats, axis=0, ddof=1)

        data.update({
            'mean': np.nanmean(df_repeats, axis=0),
            # 'lb': df_mean - 2 * df_std / np.sqrt(df_repeats.shape[0]),  # Confidence interval of the mean.
            # 'ub': df_mean + 2 * df_std / np.sqrt(df_repeats.shape[0]),  # Confidence interval of the mean.
            'lb': df_mean - df_std / 2,  # This simply plots something proportional to the std to compare the precision of different estimates without cluttering the figure.
            'ub': df_mean + df_std / 2,  # This simply plots something proportional to the std to compare the precision of different estimates without cluttering the figure.
            'min': np.amin(df_repeats, axis=0),
            'max': np.amax(df_repeats, axis=0)
        })

    return all_data


# =============================================================================
# PAPER FIGURES
# =============================================================================

def plot_paper_distributions(p1, p2, n_samples=200, output_dir_path=None):
    """Plot the sampled and target distributions used for Figure 1 in the paper."""
    import seaborn as sns
    sns.set_context('paper', font_scale=0.8)

    figsize = (PAPER_COL_WIDTH/2*0.7, 1.2)
    labelpadx = 0.0
    labelpady = -1.0
    p_hist_kwargs = dict(x_lims=(-4, 4), y_lims=(-4, 4), cmap="coolwarm")
    # p_hist_kwargs = dict(x_lims=(-4, 4), y_lims=(-4, 4), cmap=FESSA_PALETTE)
    tick_params_kwargs = dict(axis='both', which='major', left=False, bottom=False, pad=-3)

    # Load flow.
    flow = ToyFlow()
    flow_file_path = os.path.join(TOY_PROBLEM_DIR_PATH, 'tfep-double-gaussian-affine',
                                  'klpg-train2500', 'repeat0', 'flow.pt')
    flow.load_state_dict(torch.load(flow_file_path))

    # Sample.
    p1_samples = p1.sample((n_samples,))
    mapped_samples, _ = flow(p1_samples)

    # Plot the sampled distribution.
    fig1, ax1 = plt.subplots(figsize=figsize)
    plot_2d_p(p1, ax=ax1, samples=p1_samples, **p_hist_kwargs)
    ax1.text(2.0, -2.5, 'p$_A$', color='white')
    ax1.set_xlabel('x$_1$', labelpad=labelpadx)
    ax1.set_ylabel('x$_2$', labelpad=labelpady, rotation='horizontal')
    ax1.tick_params(**tick_params_kwargs)
    # ax1.tick_params(labelbottom=False, labeltop=True, **tick_params_kwargs)
    # ax1.xaxis.set_label_position('top')

    # Plot the mapped distribution.
    fig2, ax2 = plt.subplots(figsize=figsize)
    plot_2d_p(p2, ax=ax2, samples=mapped_samples, **p_hist_kwargs)
    ax2.text(2.0, -2.5, 'p$_B$', color='white')
    ax2.set_xlabel('y$_1$', labelpad=labelpadx)
    ax2.set_ylabel('y$_2$', labelpad=labelpady, rotation='horizontal')
    ax2.tick_params(**tick_params_kwargs)

    for ax in [ax1, ax2]:
        ax.set_xlim((-3, 3))
        ax.set_ylim((-3, 3))

    for fig in [fig1, fig2]:
        fig.tight_layout(pad=0.05)

    if output_dir_path is not None:
        os.makedirs(output_dir_path, exist_ok=True)
        fig1.savefig(os.path.join(output_dir_path, 'pA.pdf'))
        fig2.savefig(os.path.join(output_dir_path, 'pB.pdf'))
    else:
        plt.show()


def plot_paper_tfep_asymptotic_behavior(
        p1, p2, n_train_samples,
        subdir_pattern, eval_file_path,
        output_dir_path=None
):
    """Plot the free energy trajectories used in Figure 1 of the paper."""
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=0.8)
    fig, ax = plt.subplots(figsize=(PAPER_COL_WIDTH/1.85, 1.9))

    lw = 1.0  # Line width

    # Read the Deltaf prediction and average work evaluated on training
    # set of the ML map for each value of n_train_samples and repeat.
    analytic_df = p1.df(p2)
    all_data = read_ml_df_data(n_train_samples, analytic_df, subdir_pattern, eval_file_path)

    # Collect the handles of CIs to include in the legend.
    fill_between_handles = []

    # Plot the DF evaluated on the training and evaluation sets.
    for data_label, plot_label, color, zorder in [
        ('df_tfep_train', '$\Delta \hat{f}_{\mathregular{TFEP}}^{~\mathregular{train}}$', 'C2', 2),
        ('df_tfep_fixed_eval', '$\Delta \hat{f}_{\mathregular{TFEP}}^{~\mathregular{eval}}$', 'C1', 1)
    ]:
        data = all_data[data_label]
        mean, lb, ub, min, max = data['mean'], data['lb'], data['ub'], data['min'], data['max']

        # Plot mean CIs as shaded areas.
        handle = ax.fill_between(n_train_samples, lb, ub, facecolor=color, edgecolor='none', alpha=0.35)
        fill_between_handles.append(handle)

        # Plot full range as errorbars.
        if 'train' in data_label:
            ax.errorbar(n_train_samples, mean, yerr=[-(min-mean), max-mean], color=color,
                        lw=lw, marker='.', zorder=zorder, label=plot_label)
        else:
            ax.plot(n_train_samples, mean, color=color, lw=lw, marker='.', zorder=zorder, label=plot_label)

    # Plot the FEP estimate evaluated on the evaluation set.
    # data = all_data['df_fep_fixed_eval']  # This is used for fixed N_eval
    data = all_data['df_fep_fixed_train']
    data = {k: data[k][:len(n_train_samples)] for k in data}

    mean, lb, ub, color = data['mean'], data['lb'], data['ub'], 'C0'
    handle = ax.fill_between(n_train_samples, lb, ub, facecolor=color, edgecolor='none', alpha=0.35)
    fill_between_handles.append(handle)

    # This is used for fixed N_eval.
    # ax.hlines(mean, xmin=n_train_samples[0], xmax=n_train_samples[-1], color=color,
    #           lw=lw, label='$\Delta \hat{f}_{FEP}^{~\mathregular{eval}}$')
    ax.plot(n_train_samples, mean, color=color, lw=lw, label='$\Delta \hat{f}_{FEP}^{~\mathregular{train}}$')

    # Plot the analytical DF.
    ax.hlines(analytic_df, xmin=n_train_samples[0], xmax=n_train_samples[-1], color='black',
              lw=lw, ls='--', label='$\Delta f$', zorder=2)

    # Configure axes.
    ax.set_xlim((48, 2600))
    ax.set_ylim((-6.25, -4.3))

    # Make labels and ticks closer to axis.
    ax.set_ylabel('$\Delta f$', rotation='horizontal')
    ax.set_xlabel('N$_{\mathregular{tr}}$', labelpad=-3)
    ax.yaxis.set_label_coords(-0.25, 0.55)

    # Set logarithm axes scale.
    ax.set_xscale('log')
    ax.tick_params(axis='x', which='minor', bottom=True, direction='in', color='lightgray', width=0.5)
    ax.tick_params(axis='both', which='major', pad=-3, grid_linewidth=0.5)

    # Fix order of legend entries.
    order = [2, 1, 0, 3]
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]

    fill_between_handles = list(reversed(fill_between_handles))
    handles = [handles[0]] + [(fill_between_handles[i], h) for i, h in enumerate(handles[1:])]
    ax.legend(handles, labels,
              loc='lower right', bbox_to_anchor=(1.05, 0.1), ncol=2, frameon=False,
              labelspacing=0.6, columnspacing=0.8, handletextpad=0.25, handlelength=1.2)

    plt.tight_layout(pad=0.1)#, rect=[0, 0, 1.01, 1])

    if output_dir_path is None:
        plt.show()
    else:
        fig.savefig(os.path.join(output_dir_path, 'tfep_asymptotic_behavior.pdf'))


# =============================================================================
# MAIN
# =============================================================================

def train_targeted_flow(
        p1, p2, train_samples, batch_size, n_epochs, subdir_path,
        p_sample=None, loss_func=None, bidirectional=False, eval_samples=1000,
        flow_type='tfep', transformer='affine', basin_A_bounds=None, basin_B_bounds=None,
        input_bounds=None, cv_idx=0
):
    """Main function responsible for training a TFEP flow and store final model and metrics.

    The function returns the trained flow and a dictionary of metrics whose
    keywords (which can be configured from the function arguments) include

    'mean_loss_traj':
        The mean loss for each epoch computed as the mean of all the batch loss.
    'train_df_traj':
        The free energy evaluated at the end of each epoch on the training dataset.
    'eval_df_traj':
        The free energy evaluated at the end of each epoch on the evaluation dataset.
    'train_loss_traj':
        The total loss computed at the end of each epoch on the full training dataset.
    'eval_loss_traj':
        The total loss computed at the end of each epoch on the full evaluation dataset.

    By 'free energy' above, we mean either the total free energy between
    the target and reference distributions, or the free energy between
    two basins defined by ``basin_A_bounds`` and ``basin_B_bounds`` at the
    target level. This behavior is controlled by the ``flow_type`` keyword.

    Parameters
    ----------
    p1 : DistrMixin
        The reference distribution.
    p2 : DistrMixin
        The target distribution.
    train_samples : int or torch.Tensor
        Either a number of samples to sample from this distribution or
        pre-generated samples as a tensor.
    batch_size : int
        The batch size used for training.
    n_epochs : int
        The total number of training epochs.
    subdir_path : str
        The name of the output subdirectory inside ``TOY_PROBLEM_DIR_PATH``.
    p_sample: DistrMixin, optional
        A different distribution from where the samples comes from.
        If this is given the log weights associated to each work
        value is used to reweight the free energy.
    loss_func : torch.nn.Module
        The loss function.
    bidirectional : bool
        If True, the neural network is trained alternatively in the
        forward and backward direction (at each batch). The samples
        from p2 are sampled exactly unless ``bidirectional`` is
        'simulated', in which case the samples from p2 for the backward
        training are generated from a single step of Metropolis MC
        starting from the mapped samples from p1.
    eval_samples : int or torch.Tensor
        Either a number of samples to sample from this distribution or
        pre-generated samples as a tensor. These samples are used to
        evaluate the statistics on an independent dataset.
    flow_type : str
        Either 'tfep' or 'trp'. If the first, the free energy computed
        for the metrics is the difference between the two distributions.
        If the latter, the free energy profile is computed instead.
    transformer : str
        Either 'affine' or 'neuralspline'.
    basin_A_bounds : Tuple[float]
        The limits of the CV defining the first basin. Ignored if
        ``flow.dimension_conditioning`` is 0.
    basin_B_bounds : Tuple[float]
        The limits of the CV defining the second basin. Ignored if
        ``flow.dimension_conditioning`` is 0.
    input_bounds : numpy.ndarray, optional
        Array of shape (sample_dim, 2), where input_bounds[i][0] and
        input_bounds[i][1] are the lower and upper bound that can be
        sampled for the i-th dimension. If given, only samples within
        these bounds are computed.
    cv_idx : int, optional
        The index of the degree of freedom that is considered the CV.

    Returns
    -------
    train_samples : torch.Tensor
        The samples generated for training.
    eval_samples : torch.Tensor
        The samples generated for evaluation.
    flow : torch.nn.Module
        The trained normalizing flow.
    metrics : Dict
        The dictionary of evaluated metrics.

    """
    train_samples_file_path = os.path.join(TOY_PROBLEM_DIR_PATH, subdir_path, 'train_samples.pt')
    eval_samples_file_path = os.path.join(TOY_PROBLEM_DIR_PATH, subdir_path, 'eval_samples.pt')
    flow_file_path = os.path.join(TOY_PROBLEM_DIR_PATH, subdir_path, 'flow.pt')
    metrics_file_path = os.path.join(TOY_PROBLEM_DIR_PATH, subdir_path, 'metrics.pt')

    # Create subdir.
    os.makedirs(os.path.join(TOY_PROBLEM_DIR_PATH, subdir_path), exist_ok=True)

    # Default loss function.
    if loss_func is None:
        loss_func = KLPg(p1, p2)

    train_samples = _process_samples(train_samples, p1, p_sample, train_samples_file_path, input_bounds=input_bounds)
    eval_samples = _process_samples(eval_samples, p1, p_sample, eval_samples_file_path, input_bounds=input_bounds)

    # Create/load flow.
    if flow_type == 'tfep':
        flow = ToyFlow(transformer=transformer, input_bounds=input_bounds)
    else:
        # Preserve the first dimension.
        flow = ToyFlow(dimension_conditioning=1, transformer=transformer, input_bounds=input_bounds)

    try:
        flow.load_state_dict(torch.load(flow_file_path))
        metrics = torch.load(metrics_file_path)
    except FileNotFoundError:
        # Train the flow.
        flow, metrics = train_flow(
            p1, p2, flow=flow, train_samples=train_samples, p_sample=p_sample, batch_size=batch_size, n_epochs=n_epochs,
            loss_func=loss_func, bidirectional=bidirectional, eval_samples=eval_samples,
            basin_A_bounds=basin_A_bounds, basin_B_bounds=basin_B_bounds,
            input_bounds=input_bounds, cv_idx=cv_idx
        )

        # Update cache files.
        torch.save(flow.state_dict(), flow_file_path)
        torch.save(metrics, metrics_file_path)

    return train_samples, eval_samples, flow, metrics


def compute_eval_results(
        p1, p2, n_train_samples, n_eval_samples,
        fixed_n_train, fixed_n_eval, n_repeats_per_net,
        subdir_pattern, output_file_path,
        p_sample=None, flow_type='tfep', transformer='affine',
        basin_A_bounds=None, basin_B_bounds=None,
        input_bounds=None
):
    """Compute the free energies with FEP and TFEP using the maximum-likelihood network.

    Let ``n_nets`` be the number of independent neural network trained, and
    ``n_repeats_per_net`` be the number of evaluation dataset per trained net.
    The returned (and saved) dictionary includes the following data

        'df_fep_fixed_eval': array of size (n_nets*n_repeats_per_net,)
            The free energy with standard FEP computed on the evaluation dataset
            of size ``fixed_n_eval`` for each repeat.
        'df_fep_fixed_train': array of size (n_nets*n_repeats_per_net, len(n_eval_samples))
            The free energy with standard FEP for an increasing size of the
            evaluation dataset for each repeat.
        'df_tfep_fixed_eval': array of size (n_nets*n_repeats_per_net, len(n_train_samples))
            The free energy with targeted FEP  computed on the evaluation dataset
            of size ``fixed_n_eval``using ML maps trained with increasingly larger
            training datasets for each repeat.
        'df_tfep_fixed_train': array of size (n_nets*n_repeats_per_net, len(n_eval_samples))
            The free energy with targeted FEP for an increasing size of the
            evaluation dataset computed using the ML map trained with a training
            dataset of size ``fixed_n_train`` for each repeat.

    Parameters
    ----------
    p1 : DistrMixin
        The reference distribution.
    p2 : DistrMixin
        The target distribution.
    n_train_samples : List[int]
        The sizes of the training datasets to evaluate.
    n_eval_samples : List[int]
        The sizes of the evaluation datasets to evaluate.
    fixed_n_train : int
        The size of the training dataset used to compute the "_fixed_train" statistics.
    fixed_n_eval : int
        The size of the evaluation dataset used to compute the "_fixed_train" statistics.
    n_repeats_per_net : int
        The number of independent evaluation datasets generated for each
        independently trained neural network.
    subdir_pattern : str
        A string patter (e.g., 'mysubdir{}name'), where the argument in
        brackets will be substituted using the ``format()`` function with
        the training dataset size. This is used to find all the independently
        trained neural networks as a subdirectory of ``TOY_PROBLEM_DIR_PATH``.
    output_file_path : str
        Where to save the data in numpy compressed format.
    p_sample : DistrMixin
        A different distribution used to generate the training samples.
        This is useful to simulated "biased" sampling.
    flow_type : str
        Either 'tfep' or 'trp'.
    transformer : str
        Either 'affine' or 'neuralspline'.
    basin_A_bounds : Tuple[float]
        The limits of the CV defining the first basin. Ignored if
        ``flow.dimension_conditioning`` is 0.
    basin_B_bounds : Tuple[float]
        The limits of the CV defining the second basin. Ignored if
        ``flow.dimension_conditioning`` is 0.
    input_bounds : numpy.ndarray, optional
        Array of shape (sample_dim, 2), where input_bounds[i][0] and
        input_bounds[i][1] are the lower and upper bound that can be
        sampled for the i-th dimension. If given, only samples within
        these bounds are computed.

    """
    is_tfep = flow_type == 'tfep'

    # These require knowing the number of repeats and are initialized in the loop.
    df_fep_fixed_eval = None
    df_fep_fixed_train = None
    df_tfep_fixed_eval = None
    df_tfep_fixed_train = None

    # Build the reference free energy profile to compute the delta f between basins.
    if not is_tfep:
        ref_fs = _get_reference_fs(p1, basin_A_bounds, basin_B_bounds, n_cv_bins=400, cv_idx=0)

    for n_train_idx, n_train in enumerate(n_train_samples):
        # Find all repeats for this number of n_train samples.
        repeats_path_pattern = os.path.join(TOY_PROBLEM_DIR_PATH, subdir_pattern.format(n_train), 'repeat*')
        repeats_dir_paths = list(glob.glob(repeats_path_pattern))
        n_nets = len(repeats_dir_paths)

        # Load all the flows.
        if is_tfep:
            dimension_conditioning = 0
        else:
            dimension_conditioning = 1

        flows = [ToyFlow(dimension_conditioning=dimension_conditioning , transformer=transformer, input_bounds=input_bounds)
                 for _ in range(n_nets)]
        for flow, repeat_dir_path in zip(flows, repeats_dir_paths):
            flow.load_state_dict(torch.load(os.path.join(repeat_dir_path, 'flow.pt')))

        # Initialize FEP and TFEP Delta f arrays (which requires n_nets).
        if n_train_idx == 0:
            df_fep_fixed_eval = np.empty(n_nets*n_repeats_per_net)
            df_fep_fixed_train = np.empty((n_nets*n_repeats_per_net, len(n_eval_samples)))
            df_tfep_fixed_eval = np.empty((n_nets*n_repeats_per_net, len(n_train_samples)))
            df_tfep_fixed_train = np.empty((n_nets*n_repeats_per_net, len(n_eval_samples)))

        # Compute the free energy as a function of n_train and n_eval.
        for flow_idx, flow in enumerate(flows):
            for repeat_idx in range(n_repeats_per_net):
                # As a function of n_train.
                result_idx = flow_idx*n_repeats_per_net + repeat_idx
                samples = _sample_util(p1, p_sample, n_samples=fixed_n_eval, input_bounds=input_bounds)
                if is_tfep:
                    df_tfep_fixed_eval[result_idx, n_train_idx] = p1.tfep(p2, samples, p_sample=p_sample, map=flow).detach().numpy()
                else:
                    df_basins = p1.trp_df_basins(basin_A_bounds, basin_B_bounds, p2, samples, ref_fs=ref_fs, p_sample=p_sample, map=flow)
                    df_tfep_fixed_eval[result_idx, n_train_idx] = df_basins

                # FEP actually doesn't depend on n_train.
                if n_train_idx == 0:
                    if is_tfep:
                        df_fep_fixed_eval[result_idx] = p1.tfep(p2, samples, p_sample=p_sample).detach().numpy()
                    else:
                        df_basins = p1.trp_df_basins(basin_A_bounds, basin_B_bounds, p2, samples, ref_fs=ref_fs, p_sample=p_sample)
                        df_fep_fixed_eval[result_idx] = df_basins

                # If this is the correct n_train, compute Df as a function of n_eval.
                if n_train == fixed_n_train:
                    for n_eval_idx, n_eval in enumerate(n_eval_samples):
                        samples = _sample_util(p1, p_sample, n_samples=n_eval, input_bounds=input_bounds)

                        if is_tfep:
                            df_fep_fixed_train[result_idx, n_eval_idx] = p1.tfep(p2, samples, p_sample=p_sample).detach().numpy()
                            df_tfep_fixed_train[result_idx, n_eval_idx] = p1.tfep(p2, samples, p_sample=p_sample, map=flow).detach().numpy()
                        else:
                            df_basins = p1.trp_df_basins(basin_A_bounds, basin_B_bounds, p2, samples, ref_fs=ref_fs, p_sample=p_sample)
                            df_fep_fixed_train[result_idx, n_eval_idx] = df_basins
                            df_basins = p1.trp_df_basins(basin_A_bounds, basin_B_bounds, p2, samples, map=flow, ref_fs=ref_fs, p_sample=p_sample)
                            df_tfep_fixed_train[result_idx, n_eval_idx] = df_basins

    # Cache everything and return.
    data = {
        'df_fep_fixed_train': df_fep_fixed_train,
        'df_tfep_fixed_train': df_tfep_fixed_train,
        'df_fep_fixed_eval': df_fep_fixed_eval,
        'df_tfep_fixed_eval': df_tfep_fixed_eval,
    }
    np.savez(output_file_path, **data)
    return data


if __name__ == '__main__':

    # --------- #
    # Configure #
    # --------- #

    # Create the main output directory.
    TOY_PROBLEM_DIR_PATH = os.path.join('..', 'toy_problem')
    os.makedirs(TOY_PROBLEM_DIR_PATH, exist_ok=True)

    # Parse the arguments passed through the command line.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--flowtype', type=str, dest='flow_type', default='tfep', help='Either "tfep" or "trp".')
    parser.add_argument('--jobid', type=int, dest='job_id', help='1-based job ID for training (overrides SLURM_ARRAY_TASK_ID)')
    args = parser.parse_args()

    job_id = args.job_id
    if job_id is None:
        job_id = os.getenv('SLURM_ARRAY_TASK_ID')
    if job_id is not None:
        # From 1-based index to 0-based index.
        job_id = int(job_id) - 1

    if args.flow_type not in ['tfep', 'trp']:
        raise ValueError('flowtype must be one between "tfep" and "trp"')
    flow_type = args.flow_type

    # Configure Torch.
    torch.set_default_dtype(torch.float64)
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        n_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        torch.set_num_threads(n_cpus)


    # Configure systems.
    p1 = GaussianMixture(
        p_mixture=[0.7, 0.3],
        means=np.array([
            [-1.0, 0.0],
            [1.3, 1.0]]
        ),
        covs=np.array([
            [[0.4, 0.2], [0.2, 0.2]],
            [[0.6, 0.1], [0.1, 0.1]]
        ]),
        f=5
    )
    p2 = GaussianMixture(
        p_mixture=[0.6, 0.4],
        means=np.array([
            [-1.0, -0.2],
            [1.4, 2.1]]
        ),
        covs=np.array([
            [[0.2, 0.1], [0.1, 0.7]],
            [[0.8, 0.1], [0.1, 0.2]]
        ]),
        f=0
    )
    BASIN_A_BOUNDS = (-4.0, 0.09)
    BASIN_B_BOUNDS = (0.51, 4.0)

    # Paths and hyperparameters for the training and evaluation.
    system_name = 'double-gaussian'
    transformer = 'affine'
    prefix = flow_type + '-' + system_name + '-' + transformer

    cv_idx = 0

    n_train_repeats = 40
    subdir_pattern = os.path.join(prefix, 'klpg-train{}')
    fig_dir_path = os.path.join(TOY_PROBLEM_DIR_PATH, prefix, 'figures')

    n_train_samples = [50, 100, 250, 500, 1000, 2500]
    batch_size = 250
    n_epochs = 10000

    n_eval_samples = [50, 100, 250, 500, 1000, 2500, 5000, 10000]
    n_repeats_per_net = 50
    fixed_n_train = 500
    fixed_n_eval = 500
    eval_file_path = os.path.join(TOY_PROBLEM_DIR_PATH, prefix, 'eval.npz')

    # ----- #
    # Train #
    # ----- #

    jobs = []
    for n_samples in n_train_samples:
        for repeat_idx in range(n_train_repeats):
            jobs.append((repeat_idx, n_samples))

    # Select the job to run.
    jobs = jobs[job_id:job_id+1]

    for repeat_idx, n_samples in jobs:
        subdir_path = os.path.join(subdir_pattern.format(n_samples), 'repeat{}'.format(repeat_idx))
        train_samples, _, flow, metrics = train_targeted_flow(
            p1, p2, n_samples, batch_size, n_epochs, subdir_path,
            flow_type=flow_type, transformer=transformer,
            basin_A_bounds=BASIN_A_BOUNDS, basin_B_bounds=BASIN_B_BOUNDS
        )

    # -------- #
    # Evaluate #
    # -------- #

    compute_eval_results(
        p1, p2, n_train_samples, n_eval_samples,
        fixed_n_train, fixed_n_eval, n_repeats_per_net,
        subdir_pattern, eval_file_path,
        flow_type=flow_type, transformer=transformer,
        basin_A_bounds=BASIN_A_BOUNDS, basin_B_bounds=BASIN_B_BOUNDS,
    )

    # ------------- #
    # Paper figures #
    # ------------- #

    os.makedirs(PAPER_FIG_DIR_PATH, exist_ok=True)

    # Plot the two distributions.
    plot_paper_distributions(p1, p2, output_dir_path=PAPER_FIG_DIR_PATH)

    # Plot asymptotic behavior.
    plot_paper_tfep_asymptotic_behavior(
        p1, p2, n_train_samples,
        subdir_pattern, eval_file_path,
        output_dir_path=PAPER_FIG_DIR_PATH
    )
