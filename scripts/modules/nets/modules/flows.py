#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Normalizing flow layers for PyTorch.

All the layers defined in this module are invertible and implement an
``inv()`` method (not to be comfused with the ``Tensor``'s ``backward()``
method which backpropagate the gradients.

The forward propagation of the modules here return both the transformation
of the input plus the log determinant of the Jacobian.
w
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import functools

import numpy as np
import torch

from .autoregressive import MADE
from ..functions.transformer import (
    affine_transformer, affine_transformer_inv,
    sos_polynomial_transformer, neural_spline_transformer,
    unit_cube_to_inscribed_sphere, mobius_transformer
)


# =============================================================================
# TRANSFORMERS
# =============================================================================

class AffineTransformer(torch.nn.Module):
    """Gated affine transformer module for normalizing flows.

    See Also
    --------
    nets.functions.transformer.affine_transformer

    """
    n_parameters_per_input = 2

    def forward(self, x, parameters, gate=None):
        shift, log_scale = self.split_parameters(parameters)
        return affine_transformer(x, shift, log_scale, gate)

    def inv(self, y, parameters, gate=None):
        shift, log_scale = self.split_parameters(parameters)
        return affine_transformer_inv(y, shift, log_scale, gate)

    def split_parameters(self, parameters):
        return parameters[:, 0], parameters[:, 1]

    def get_identity_conditioner(self, dimension_out):
        return torch.zeros(size=(self.n_parameters_per_input, dimension_out))


class SOSPolynomialTransformer(torch.nn.Module):
    """Sum-of-squares polynomial transformer module for normalizing flows.

    See Also
    --------
    nets.functions.transformer.sos_polynomial_transformer

    """
    def __init__(self, n_polynomials=2):
        super().__init__()
        if n_polynomials < 2:
            raise ValueError('n_polynomials must be strictly greater than 1.')
        self.n_polynomials = n_polynomials

    @property
    def degree_polynomials(self):
        return 1

    @property
    def parameters_per_polynomial(self):
        return self.degree_polynomials + 1

    @property
    def n_parameters_per_input(self):
        return self.parameters_per_polynomial * self.n_polynomials + 1

    def forward(self, x, coefficients, gate=None):
        return sos_polynomial_transformer(x, coefficients, gate)

    def inv(self, y, coefficients, gate=None):
        raise NotImplementedError(
            'Inversion of SOS polynomial transformer has not been implemented yet.')

    def get_identity_conditioner(self, dimension_out):
        id_conditioner = torch.zeros(size=(self.n_parameters_per_input, dimension_out))
        # The sum of the squared linear coefficients must be 1.
        id_conditioner[1::self.parameters_per_polynomial].fill_(np.sqrt(1 / self.n_polynomials))
        return id_conditioner


class NeuralSplineTransformer(torch.nn.Module):
    """Neural spline transformer module for normalizing flows.

    See Also
    --------
    nets.functions.transformer.neural_spline_transformer

    """
    def __init__(self, x0, xf, n_bins, y0=None, yf=None):
        super().__init__()

        # Handle mutable default arguments y_0 and y_final.
        if y0 is None:
            y0 = x0.detach()
        if yf is None:
            yf = xf.detach()

        self.x0 = x0
        self.xf = xf
        self.y0 = y0
        self.yf = yf
        self.n_bins = n_bins

    @property
    def n_parameters_per_input(self):
        # n_bins widths, n_bins heights and n_bins-1 slopes.
        return 3*self.n_bins - 1

    def forward(self, x, parameters, gate=None):
        if gate is not None:
            raise NotImplementedError(
                'NeuralSplineTransformer does not support the "gate" parameter yet.')

        # Divide the parameters in widths, heights and slopes.
        widths = torch.nn.functional.softmax(parameters[:, :self.n_bins], dim=1) * (self.xf - self.x0)
        heights = torch.nn.functional.softmax(parameters[:, self.n_bins:2*self.n_bins], dim=1) * (self.yf - self.y0)
        slopes = torch.nn.functional.softplus(parameters[:, 2*self.n_bins:])
        return neural_spline_transformer(x, self.x0, self.y0, widths, heights, slopes)

    def inv(self, y, parameters, gate=None):
        raise NotImplementedError(
            'Inversion of neural spline transformer has not been implemented yet.')

    def get_identity_conditioner(self, dimension_out):
        # Strictly speaking, this becomes the identity conditioner only if x0 == y0 and xf == yf.
        id_conditioner = torch.empty(size=(self.n_parameters_per_input, dimension_out))

        # Both the width and the height of each bin must be constant.
        # Remember that the parameters go through the softmax function.
        id_conditioner[:self.n_bins].fill_(1 / self.n_bins)
        id_conditioner[self.n_bins:2*self.n_bins].fill_(1 / self.n_bins)

        # The slope must be one in each knot. Remember that the parameters
        # go through the softplus function.
        id_conditioner[2*self.n_bins:].fill_(np.log(np.e - 1))

        return id_conditioner


class MobiusTransformer(torch.nn.Module):
    """Mobius transformer module for normalizing flows.

    See Also
    --------
    nets.functions.transformer.mobius_transformer

    """

    n_parameters_per_input = 1

    def __init__(self, blocks, shorten_last_block=False):
        super().__init__()
        self.blocks = blocks
        self.shorten_last_block = shorten_last_block

    def forward(self, x, w, gate=None):
        w = self._map_to_sphere(w)
        return mobius_transformer(x, w, self.blocks, gate, self.shorten_last_block)

    def inv(self, y, w, gate=None):
        w = self._map_to_sphere(w)
        return mobius_transformer(y, -w, self.blocks, gate, self.shorten_last_block)

    def get_identity_conditioner(self, dimension_out):
        return torch.zeros(size=(self.n_parameters_per_input, dimension_out))

    def _map_to_sphere(self, w):
        """Map w from the real hypervolume to the unit hypersphere."""
        # MAF passes the parameters in shape (batch_size, n_parameters, n_features).
        w = w[:, 0]
        # Tanh maps the real hypervolume to the unit hypercube.
        w = torch.tanh(w)
        # Finally we map the unit hypercube to the unit hypersphere.
        return unit_cube_to_inscribed_sphere(w, self.blocks, self.shorten_last_block)


# =============================================================================
# LAYERS
# =============================================================================

class InvertibleBatchNorm1d(torch.nn.BatchNorm1d):
    """
    Extension of torch.nn.BatchNorm1d to compute the Jacobian and inverse.
    """
    def __init__(self, n_features, n_constant_features=0, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, initialize_identity=True):
        # The identity initialization is data-dependent, but we
        # need a scale and shifting parameters to do it.
        if initialize_identity and not affine:
            raise ValueError('Impossible to initialize the normalization '
                             'to identity without the affine parameters.')

        super().__init__(n_features-n_constant_features, eps, momentum, affine, track_running_stats)

        self.initialize_identity = initialize_identity
        self.n_constant_features = n_constant_features

        # This is used to keep track of the data-dependent
        # initialization of the affine parameters.
        self._initialized = False

    def n_parameters(self):
        """int: The total number of parameters."""
        return sum(len(p) for p in self.parameters())

    def forward(self, input, gate=None):
        if gate is not None:
            raise NotImplementedError('Gating is not supported.')

        # Data-dependent initialization of the affine parameters happens
        # only if we need to initialize this to the identity function.
        if not self._initialized:
            if self.initialize_identity:
                self.weight.data = torch.sqrt(torch.var(input[:, self.n_constant_features:],
                                                        dim=0, unbiased=False).detach() + self.eps)
                self.bias.data = torch.mean(input[:, self.n_constant_features:], dim=0).detach()
            self._initialized = True

        # Make sure the constant features are not modified.
        if self.n_constant_features > 0:
            result = torch.empty_like(input)
            result[:, :self.n_constant_features] = input[:, :self.n_constant_features]
            result[:, self.n_constant_features:] = super().forward(input[:, self.n_constant_features:])
            log_det_J = self._compute_forward_log_det_J(input[:, self.n_constant_features:])
        else:
            result = super().forward(input)
            log_det_J = self._compute_forward_log_det_J(input)

        return result, log_det_J

    def inv(self, input, gate=None):
        if self.n_constant_features > 0:
            raise NotImplementedError('Inversion with constant features is not supported.')
        if gate is not None:
            raise NotImplementedError('Gating is not supported.')

        # We don't support data-dependent initialization starting with an inverse.
        if not self._initialized:
            raise RuntimeError('The batch normalization layer must go '
                               'through a forward pass before inverse.')

        # We don't support inverse training since there is the problem
        # of keeping track of the running mean and var.
        if self._use_batch_stats():
            raise NotImplementedError('Inverse can be used only without keeping '
                                      'track of the running mean and var.')

        # Inverse batch norm.
        result = (input - self.bias) * torch.sqrt(self.running_var + self.eps) / self.weight + self.running_mean
        log_det_J = -self._compute_forward_log_det_J(input)

        return result, log_det_J

    def _use_batch_stats(self):
        return self.training or (not self.track_running_stats)

    def _compute_forward_log_det_J(self, input):
        if self._use_batch_stats():
            var = torch.var(input, dim=0, unbiased=False)
        else:
            var = self.running_var
        log_det_J = - 0.5 * torch.log(var + self.eps)
        if self.affine:
            log_det_J += torch.log(self.weight)
        return torch.sum(log_det_J).repeat(input.shape[0])


class MAF(torch.nn.Module):
    """
    Masked Autoregressive Flow.

    This implements an autoregressive flow in which the :class:`MADE`
    network is used for the conditioner. The class supports arbitrary
    transformers.

    When the transformer is the :class:`AffineTransformer`, this is
    equivalent to MAF and IAF [1-2]. These differ only in the direction
    of the conditional dependence, effectively determining which between
    forward and inverse evaluation is faster.

    Parameters
    ----------
    dimension : int
        The number of features of a single input vector. This includes
        the number of conditioning features.
    dimensions_hidden : int or List[int], optional
        Control the number of layers and nodes of the hidden layers in
        the MADE networks that implements the conditioner. If an int,
        this is the number of hidden layers, and the number of nodes in
        each hidden layer will be set to ``dimension_in - 1) * out_per_dimension``.
        If a list, ``dimensions_hidden[l]`` must be the number of nodes
        in the l-th hidden layer. Default is 1.
    dimension_conditioning : int, optional
        If greater than zero the first ``dimension_conditioning`` input
        features will be used to condition the output of the conditioner,
        but they won't be affected by the normalizing flow.
    degrees_in : str or numpy.ndarray, optional
        The degrees to assign to the input/output nodes. Effectively this
        controls the dependencies between variables in the conditioner.
        If ``'input'``/``'reversed'``, the degrees are assigned in
        the same/reversed order they are passed. If an array, this must
        be a permutation of ``numpy.arange(0, n_blocks)``, where ``n_blocks``
        is the number of blocks passed to the constructor. If blocks are
        not used, this corresponds to the number of non-conditioning features
        (i.e., ``dimension_in - dimension_conditioning``). Default is ``'input'``.
    degrees_hidden_motif : numpy.ndarray, optional
        The degrees of the hidden nodes of the conditioner are assigned
        using this array in a round-robin fashion. If not given, they
        are assigned in the same order used for the input nodes. This
        must be at least as large as the dimension of the smallest hidden
        layer.
    weight_norm : bool, optional
        If True, weight normalization is applied to the masked linear
        modules. Default is False.
    blocks : int or List[int], optional
        If an integer, the non-conditioning input features are divided
        into contiguous blocks of size ``blocks`` that are assigned the
        same degree in the MADE conditioner. If a list, ``blocks[i]``
        must represent the size of the i-th block. The default, ``1``,
        correspond to a fully autoregressive network.
    shorten_last_block : bool, optional
        If ``blocks`` is an integer that is not a divisor of the number
        of non-conditioning  features, this option controls whether the
        last block is shortened (``True``) or an exception is raised
        (``False``). Default is ``False``.
    split_conditioner : bool, optional
        If ``True``, separate MADE networks are used to compute separately
        each parameter of the transformer (e.g., for affine transformers
        which require scale and shift parameters, two networks are used).
        Otherwise, a single network is used to implement the conditioner,
        and all parameters are generated in a single pass.
    transformer : torch.nn.Module
        The transformer used to map the input features. By default, the
        ``AffineTransformer`` is used.
    initialize_identity : bool, optional
        If ``True``, the parameters are initialized in such a way that
        the flow initially performs the identity function.

    References
    ----------
    [1] Kingma DP, Salimans T, Jozefowicz R, Chen X, Sutskever I, Welling M.
        Improved variational inference with inverse autoregressive flow.
        In Advances in neural information processing systems 2016 (pp. 4743-4751).
    [2] Papamakarios G, Pavlakou T, Murray I. Masked autoregressive flow for
        density estimation. In Advances in Neural Information Processing
        Systems 2017 (pp. 2338-2347).
    [3] Papamakarios G, Nalisnick E, Rezende DJ, Mohamed S, Lakshminarayanan B.
        Normalizing Flows for Probabilistic Modeling and Inference. arXiv
        preprint arXiv:1912.02762. 2019 Dec 5.

    """

    def __init__(
            self,
            dimension,
            dimensions_hidden=1,
            dimension_conditioning=0,
            degrees_in='input',
            degrees_hidden_motif=None,
            weight_norm=False,
            blocks=1,
            shorten_last_block=False,
            split_conditioner=True,
            transformer=None,
            initialize_identity=True
    ):
        super().__init__()

        # By default, use an affine transformer.
        if transformer is None:
            transformer = AffineTransformer()
        self._transformer = transformer

        if split_conditioner:
            n_conditioners = self._transformer.n_parameters_per_input
            out_per_dimension = 1
        else:
            n_conditioners = 1
            out_per_dimension = self._transformer.n_parameters_per_input

        # We need two MADE layers for the scaling and the shifting.
        self._conditioners = torch.nn.ModuleList()
        for i in range(n_conditioners):
            self._conditioners.append(MADE(
                dimension_in=dimension,
                dimensions_hidden=dimensions_hidden,
                out_per_dimension=out_per_dimension,
                dimension_conditioning=dimension_conditioning,
                degrees_in=degrees_in,
                degrees_hidden_motif=degrees_hidden_motif,
                weight_norm=weight_norm,
                blocks=blocks,
                shorten_last_block=shorten_last_block,
            ))

        # Initialize the log_scale and shift nets to 0.0 so that at
        # the beginning the MAF layer performs the identity function.
        if initialize_identity:
            dimension_out = dimension - dimension_conditioning

            # Determine the conditioner that will make the transformer the identity function.
            identity_conditioner = self._transformer.get_identity_conditioner(dimension_out)

            # If we have not split the conditioners over multiple networks,
            # there is a single output bias parameter vector so we need to
            # convert from shape (batch_size, n_parameters_per_input, n_features)
            # to (batch_size, n_parameters_per_input * n_features).
            if not split_conditioner:
                identity_conditioner = torch.reshape(
                    identity_conditioner,
                    (self._transformer.n_parameters_per_input * dimension_out,)
                )
                identity_conditioner = [identity_conditioner]

            for net, id_cond in zip(self._conditioners, identity_conditioner):
                # Setting to 0.0 only the last layer suffices.
                if weight_norm:
                    net.layers[-1].weight_g.data.fill_(0.0)
                else:
                    net.layers[-1].weight.data.fill_(0.0)
                net.layers[-1].bias.data = id_cond

    @property
    def dimension_conditioning(self):
        return self._conditioners[0].dimension_conditioning

    @property
    def degrees_in(self):
        return self._conditioners[0].degrees_in

    def n_parameters(self):
        """int: The total number of (unmasked) parameters."""
        return sum(c.n_parameters() for c in self._conditioners)

    def forward(self, x, gate=None):
        parameters = self._run_conditioners(x)

        # Make sure the conditioning dimensions are not altered.
        dimension_conditioning = self.dimension_conditioning
        if dimension_conditioning == 0:
            y, log_det_J = self._transformer(x, parameters, gate=gate)
        else:
            # There are conditioning dimensions.
            y = torch.empty_like(x)
            y[:, :dimension_conditioning] = x[:, :dimension_conditioning]
            y[:, dimension_conditioning:], log_det_J = self._transformer(
                x[:, dimension_conditioning:], parameters, gate=gate)

        return y, log_det_J

    def inv(self, y, gate=None):
        # This is slower because to evaluate x_i we need all x_<i.
        # For algorithm, see Eq 39 in reference [3] above.
        dimension_conditioning = self.dimension_conditioning

        # Initialize x to an arbitrary value.
        x = torch.zeros_like(y)
        if dimension_conditioning > 0:
            # All outputs of the nets depend on the conditioning features,
            # which are not transformed by the MAF.
            x[:, :dimension_conditioning] = y[:, :dimension_conditioning]

        # Isolate the features that are not conditioning.
        y_nonconditioning = y[:, dimension_conditioning:]

        # We need to process each block in the order given
        # by their degree to respect the dependencies.
        blocks = self._conditioners[0].blocks
        degrees_in_nonconditioning = self._conditioners[0].degrees_in[dimension_conditioning:]

        block_start_idx = 0
        blocks_start_indices = []
        blocks_degrees = []
        for block_size in blocks:
            blocks_start_indices.append(block_start_idx)
            blocks_degrees.append(degrees_in_nonconditioning[block_start_idx])
            block_start_idx += block_size

        # Order the block by their degree.
        blocks_order = np.argsort(blocks_degrees)

        # Now compute the inverse.
        for block_idx in blocks_order:
            block_size = blocks[block_idx]
            block_start_idx = blocks_start_indices[block_idx]
            block_end_idx = block_start_idx + block_size

            # Compute the inversion with the current x.
            # Cloning, allows to compute gradients on inverse.
            parameters = self._run_conditioners(x.clone())

            # The log_det_J that we compute with the last pass is the total log_det_J.
            x_temp, log_det_J = self._transformer.inv(y_nonconditioning, parameters, gate=gate)

            # There is no output for the conditioning dimensions.
            input_start_idx = block_start_idx + dimension_conditioning
            input_end_idx = block_end_idx + dimension_conditioning

            # No need to update all the xs, but only those we can update at this point.
            x[:, input_start_idx:input_end_idx] = x_temp[:, block_start_idx:block_end_idx]

            block_start_idx += block_size

        return x, log_det_J

    def _run_conditioners(self, x):
        """Return the conditioning parameters with shape (batch_size, n_parameters, n_features)."""
        batch_size, n_features = x.shape
        n_conditioners = len(self._conditioners)
        returned_shape = (
            batch_size,
            self._transformer.n_parameters_per_input,
            n_features-self.dimension_conditioning
        )

        if n_conditioners == 1:
            # A single conditioner for all parameters. The conditioners
            # return the parameters with shape (batch_size, n_features*n_parameters).
            conditioning_parameters = self._conditioners[0](x)
            conditioning_parameters = torch.reshape(
                conditioning_parameters, shape=returned_shape)
        else:
            # The conditioners are split into independent NNs.
            # conditioning_parameters has shape (batch_size, n_features*n_parameters_per_input).
            conditioning_parameters = torch.empty(
                size=returned_shape, dtype=x.dtype)
            for conditioner_idx, conditioner in enumerate(self._conditioners):
                conditioning_parameters[:, conditioner_idx] = conditioner(x)

        return conditioning_parameters


class NormalizingFlow(torch.nn.Module):
    """A sequence of normalizing flows.

    The module offers utility to concatenate multiple flow layers in a
    sequence and implement multiscale architectures (see for example [1]).

    Parameters
    ----------
    **flows
        One or more flows to execute in sequence in the same order of the
        forward propagation.
    constant_indices : List[int], optional
        Optionally, some of the features might remain constant.
        The indices of the inputs in this list will not go through the
        flows and will not affect the output.
    dimension_gate : int, optional
        The first ``dimension_gate`` inputs are feeded to a network that
        generates the gate parameters for each layer of the flow. If greater
        than zero, all flow layers must accept the ``gate`` keyword argument.
    weight_norm : bool, optional
        If True, weight normalization is applied to the linear layers of
        the dense network used for the gate. Default is False.

    References
    ----------
    [1] Dinh L, Sohl-Dickstein J, Bengio S. Density estimation using Real NVP.
        arXiv preprint arXiv:1605.08803. 2016 May 27.

    """

    # TODO: Convert "constant_indices" argument to "multiscale_architect", which
    # TODO: is a callable multiscale_architect(layer_idx) -> List[int] that return
    # TODO: the indices of the dimensions to factor out at each layer.
    def __init__(self, *flows, constant_indices=None, dimension_gate=0, weight_norm=False):
        super().__init__()
        self.flows = torch.nn.ModuleList(flows)
        self._constant_indices = constant_indices

        # We also need the indices that we are not factoring out.
        # This is initialized lazily in self._pass() because we
        # need the dimension of the input.
        self._propagated_indices = None

        # Create the gating network.
        if dimension_gate > 0:
            for flow in flows:
                if dimension_gate > flow.dimension_conditioning:
                    raise ValueError('The of the input for the gate network '
                                     'cannot be greater than the minimum '
                                     'conditioning dimension of all flows.')

            dimension_out = len(self.flows)
            dimension_hidden = 2 * max(dimension_gate, dimension_out)
            n_hidden_layers = 3

            # Check if we need to apply weight norm.
            if weight_norm:
                apply_norm = functools.partial(torch.nn.utils.weight_norm, name='weight')
            else:
                # Leave the layer intact.
                apply_norm = lambda x: x

            # Hidden layers.
            layers = []
            for i in range(n_hidden_layers):
                dimension1 = dimension_gate if i == 0 else dimension_hidden
                layers.extend([
                    apply_norm(torch.nn.Linear(dimension1, dimension_hidden)),
                    torch.nn.Tanh(),
                ])

            # Output layer.
            layers.extend([
                apply_norm(torch.nn.Linear(dimension_hidden, dimension_out)),
                torch.nn.Sigmoid()
            ])

            self.gate_net = torch.nn.Sequential(*layers)
        else:
            self.gate_net = None

    @property
    def dimension_gate(self):
        """int: The number of features used as input for the gate."""
        if self.gate_net is None:
            return 0
        return self.gate_net[0].in_features

    def n_parameters(self):
        """int: The total number of (unmasked) parameters."""
        n_par = sum(f.n_parameters() for f in self.flows)
        if self.gate_net is not None:
            n_par += sum(p.numel() for p in self.gate_net.parameters())
        return n_par

    def forward(self, x):
        return self._pass(x, inverse=False)

    def inv(self, y):
        return self._pass(y, inverse=True)

    def _pass(self, x, inverse):
        batch_size = x.size(0)
        cumulative_log_det_J = torch.zeros(batch_size, dtype=x.dtype)

        # Check if we need to traverse the flows in forward or inverse pass.
        layer_indices = range(len(self.flows))
        if inverse:
            flow_func_name = 'inv'
            layer_indices = reversed(layer_indices)
        else:
            flow_func_name = 'forward'

        # Take care of the constant dimensions.
        if self._constant_indices is not None:
            # Check that we have already cached the propagated indices.
            if self._propagated_indices is None:
                constant_indices_set = set(self._constant_indices)
                self._propagated_indices =  list(i for i in range(x.size(1))
                                                 if i not in constant_indices_set)

            # This will be the returned tensors.
            final_x = torch.empty_like(x)
            final_x[:, self._constant_indices] = x[:, self._constant_indices]

            # This tensor goes through the flow.
            x = x[:, self._propagated_indices]

        # Compute the gating parameters.
        if self.gate_net is not None:
            gate = self.gate_net(x[:, :self.dimension_gate])

        # Now go through the flow layers.
        for layer_idx in layer_indices:
            flow = self.flows[layer_idx]
            # flow_func_name can be 'forward' or 'inv'.
            if self.gate_net is None:
                x, log_det_J = getattr(flow, flow_func_name)(x)
            else:
                x, log_det_J = getattr(flow, flow_func_name)(x, gate=gate[:, layer_idx])
            cumulative_log_det_J += log_det_J

        # Add to the factored out dimensions.
        if self._constant_indices is not None:
            final_x[:, self._propagated_indices] = x
            x = final_x

        return x, cumulative_log_det_J
