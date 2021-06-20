#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Precondittioners for the nets package.
"""


# =============================================================================
# GLOBAL
# =============================================================================

import warnings

import torch
from torch.optim import Optimizer

from .functions.math import batchwise_outer, cov


# =============================================================================
# PRECONDITIONERS BASE CLASSES
# =============================================================================

class _Preconditioner(Optimizer):
    """Base class exposing facilities to precondition the network gradient."""

    def __init__(self, params, damping=0.0, **defaults):
        super().__init__(params, defaults)
        self.damping = damping

    @property
    def damping(self):
        return self.state['damping']

    @damping.setter
    def damping(self, new_value):
        self.state['damping'] = new_value

    def damp(self, m):
        """Add the damping to the matrix.

        Parameters
        ----------
        m : torch.Tensor
            A tensor of shape ``(n, n)``, where ``n`` is the total number
            of parameters in the network to precondition.

        """
        if self.damping > 0:
            diag_view = torch.diagonal(m)
            diag_view += self.damping

    def precondition_grad(self, m):
        """Precondition the gradient with the given matrix.

        Parameters
        ----------
        m : torch.Tensor
            A tensor of shape ``(n, n)``, where ``n`` is the total number
            of parameters in the network to precondition.

        """
        # Compute conditioned gradient in vectorized form.
        grad_vec = self._get_vectorized_grad()
        conditioned_grad_vec = torch.matmul(m, grad_vec)

        # Now convert back gradient from vectorized to standard form.
        par_idx = 0
        for group in self.param_groups:
            for par in group['params']:
                n_elements = par.numel()

                par_grad_vec = conditioned_grad_vec[par_idx:par_idx+n_elements]
                if len(par.shape) > 1:
                    par_grad_vec = torch.reshape(par_grad_vec, par.shape)
                par.grad.data = par_grad_vec

                par_idx += n_elements

    def _get_vectorized_grad(self):
        """Return the gradient of all parameters in vectorized form.

        Return
        ------
        grad : torch.Tensor
            The gradient as a 1D tensor of length ``n``, where ``n`` is
            the total number of parameters in the network.

        """
        # Collect all gradients in vectorized form.
        all_grads = []
        for group in self.param_groups:
            for par in group['params']:
                grad = par.grad
                if len(par.shape) > 1:
                    grad = torch.reshape(grad, (par.numel(),))
                all_grads.append(grad)

        return torch.cat(all_grads)


class _BatchGradPreconditioner(_Preconditioner):
    """A preconditioner that collects the batch-wise gradient in a single pass.

    Currently this supports only the Linear layer. The batch-wise gradients
    are calculated from the activations and the autograd gradients as
    suggested in [1].

    The class does not implement a ``step()`` method and it is supposed
    to be used as a mixin class.

    The parameters are saved in params_group together with their module,
    which is stored in the key 'mod'.

    Parameters
    ----------
    network : torch.nn.Module
        The neural network to precondition.
    loss_type : str
        Either "sum" or "mean" are supported.
    **defaults : dict
        Other keyword arguments to pass to ``Optimizer.__init__`` as defaults.

    References
    ----------
    [1] Martens J, Grosse R. Optimizing neural networks with kronecker-factored
        approximate curvature. In International conference on machine learning
        2015 Jun 1 (pp. 2408-2417).

    """

    def __init__(self, network, loss_type, damping=0.0, **defaults):
        # Create parameters to pass to optimizers.
        params = []

        # Attach a hook to each parameter to compute the batch-wise gradient.
        # Keep track of the handles so that we can destroy the object correctly.
        self._handles = []
        for mod in network.modules():
            # No need to keep track of modules without trainable parameters.
            mod_parameters = list(mod.parameters(recurse=False))
            if len(mod_parameters) == 0:
                continue

            self._check_is_supported(mod)

            h = mod.register_forward_pre_hook(self._save_input_activation)
            self._handles.append(h)
            h = mod.register_backward_hook(self._save_grad_output)
            self._handles.append(h)

            # Store parameters.
            params.append({'params': mod_parameters, 'mod': mod})

        super().__init__(params, damping, **defaults)

        self.loss_type = loss_type

    @property
    def loss_type(self):
        return self.state['loss_type']

    @loss_type.setter
    def loss_type(self, new_value):
        if new_value not in {'sum', 'mean'}:
            raise ValueError('Unsupported loss_type "' + str(new_value) + '"')
        self.state['loss_type'] = new_value

    def compute_batchwise_grad(self, parameter, module, vectorize=False):
        """Return the batchwise gradient of the parameter.

        The function must be called only after backpropagation or an
        error is raised.

        Parameters
        ----------
        parameter : torch.nn.Parameter
            The parameter requiring batchwise gradients.
        module : torch.nn.Module
            The module owning the parameter.
        vectorize : bool, optional
            If ``True``, the gradients are vectorized. Default is ``False``.

        Return
        ------
        batch_grad : torch.Tensor
            A tensor of size ``(batch_size, *)`` where ``*`` represents
            the dimension of the parameter. If ``as_vector`` is ``True``,
            then ``*`` is the total number of elements in the parameter
            (e.g., ``n_rows * n_cols`` for the weight matrix).

        """
        batch_size = module.input_activation.shape[0]

        if self.loss_type == 'sum':
            grad_output = module.grad_output
        else:  # mean
            grad_output = module.grad_output * batch_size

        # Check if this is a weight matrix or a bias vector.
        if len(parameter.shape) == 1:
            # Bias vector.
            batch_grad = grad_output
        else:
            # Weight matrix.
            batch_grad = batchwise_outer(grad_output, module.input_activation)

            # Check if this is a masked linear layer.
            try:
                mask = module.mask
            except AttributeError:
                pass
            else:
                batch_grad *= mask

            if vectorize:
                n_elem = batch_grad.shape[1] * batch_grad.shape[2]
                batch_grad = torch.reshape(batch_grad, (batch_size, n_elem))

        return batch_grad

    def compute_module_batchwise_grad(self, module, vectorize=False, clear=False):
        """Return the batchwise gradient of the module.

        The function must be called only after backpropagation or an
        error is raised.

        Parameters
        ----------
        module : torch.nn.Module
            The module requiring batchwise gradients for the parameters.
        vectorize : bool, optional
            If ``True``, the weight matrix gradients are vectorized.
            Default is ``False``.
        clear : bool, optional
            If ``True``, after computing the gradients, the function
            deletes the tensors accumulated during forward and back
            propagation required to compute the batchwise gradients.
            Default is ``False``.

        Return
        ------
        batch_grad_w : torch.Tensor
            batch_grad_w[i][j][k] is the gradient of the weight matrix
            element jk for batch i. If ``vectorize`` is ``True``, then
            batch_grad_w[i][j][k]
        batch_grad_b : torch.Tensor, optional
            If the layer has bias parameters, batch_grad_b[i][j] is the
            gradient of the bias vector element j for batch i.

        """
        batch_grad_w = self.compute_batchwise_grad(module.weight, module, vectorize)
        if module.bias is None:
            batch_grad_b = None
        else:
            batch_grad_b = self.compute_batchwise_grad(module.bias, module)

        if clear:
            del module.grad_output
            del module.input_activation

        return batch_grad_w, batch_grad_b

    def compute_all_batchwise_grads(self, clear=False):
        """Compute the batchwise gradients of all parameters in the network in vectorized form.

        Returns
        -------
        batch_grad : torch.Tensor
            A tensor of shape ``(batch_size, n_parameters)``, where
            ``n_parameters`` is the total number of parameters in the
            preconditioned network. The order of the parameters follow
            the order of ``self.params_group``.
        clear : bool, optional
            If ``True``, after computing the gradients, the function
            deletes the tensors accumulated during forward and back
            propagation required to compute the batchwise gradients.
            Default is ``False``.

        """
        # Allocate memory for the output tensor.
        first_module = self.param_groups[0]['mod']
        batch_size = first_module.input_activation.shape[0]
        n_parameters = sum([sum([p.numel() for p in group['params']])
                            for group in self.param_groups])
        batch_grad = torch.empty(batch_size, n_parameters, dtype=first_module.input_activation.dtype)

        # Fill the batch gradient.
        par_idx = 0
        for group in self.param_groups:
            params = group['params']
            module = group['mod']

            for par in params:
                batch_grad_par = self.compute_batchwise_grad(par, module, vectorize=True)
                len_par = batch_grad_par.shape[1]
                batch_grad[:, par_idx:par_idx+len_par] = batch_grad_par
                par_idx += len_par

        # Clear.
        if clear:
            for group in self.param_groups:
                module = group['mod']
                if hasattr(module, 'input_activation'):
                    del module.input_activation
                    del module.grad_output

        return batch_grad

    def _save_input_activation(self, module, input):
        module.input_activation = input[0].detach()

    def _save_grad_output(self, module, grad_input, grad_output):
        module.grad_output = grad_output[0].detach()

    @classmethod
    def _check_is_supported(cls, module):
        if not isinstance(module, torch.nn.Linear):
            raise ValueError('Only the Linear module (and properly-implemented'
                             ' subclasses) are currently supported.')

    def __del__(self):
        for h in self._handles:
            h.remove()


# =============================================================================
# GRADIENT COVARIANCE PRECONDITIONER
# =============================================================================

class GradCovPreconditioner(_BatchGradPreconditioner):
    """Precondition with the inverse covariance matrix of the gradient.

    Parameters
    ----------
    network : torch.nn.Module
        The neural network to precondition.
    loss_type : str
        Either "sum" or "mean" are supported.

    """

    def __init__(self, network, loss_type, damping=0.0):
        super().__init__(network, loss_type, damping)

    def step(self):
        # Get batchwise gradients for all parameters in vector form.
        batch_grad = self.compute_all_batchwise_grads(clear=True)

        # Compute covariance matrix of the gradient.
        grad_cov_matrix = cov(batch_grad, ddof=1, dim_n=0, inplace=True)

        # Add damping.
        self.damp(grad_cov_matrix)

        # The pseudoinverse will be less stable with single precision.
        if grad_cov_matrix.dtype != torch.float64:
            warnings.warn('The pseudoinverse is less stable in single precision. '
                          'Consider setting float64 as default type for pytorch.')

        # Invert covariance matrix. In general, the covariance matrix
        # could be singular because of the masked linear layers.
        grad_precision_matrix = torch.pinverse(grad_cov_matrix)

        # Precondition parameters.
        self.precondition_grad(grad_precision_matrix)

        # Make sure the gradient of the masked parameters is zero as it
        # might have become just very small because of the damping and/or
        # numerical imprecisions.
        for group in self.param_groups:
            for par in group['params']:
                if hasattr(group['mod'], 'mask') and len(par.shape) == 2:
                    par.grad.mul_(group['mod'].mask)
