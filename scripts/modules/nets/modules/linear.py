#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Masked linear layer for PyTorch.

The module also contained a version of weight normalization that handles
weight vectors with zero norm, which just nans in the native PyTorch
implementation.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch
from torch import norm_except_dim
from torch.nn.parameter import Parameter
from torch.nn.utils.weight_norm import WeightNorm

from ..functions import linear


# =============================================================================
# LAYERS
# =============================================================================

class MaskedLinear(torch.nn.Linear):
    r"""Implement the masked linear transformation: :math:`y = x \cdot (M \circ A)^T + b`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        If set to ``False``, the layer will not learn an additive bias.
        Default is ``True``.
    mask : torch.Tensor, optional
        The mask of zeros and ones of shape ``(out_features, in_features)``
        to apply to the scaling matrix. Default is ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weights of the module of shape ``(out_features, in_features)``.
        The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`,
        where :math:`k = \frac{1}{\text{in\_features}}`.
    bias : torch.Tensor
        The learnable bias of the module of shape ``(out_features)``.
        If :attr:`bias` is ``True``, the values are initialized from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
        :math:`k = \frac{1}{\text{in\_features}}`.
    mask : torch.Tensor
        The mask passed during initialization.

    See Also
    --------
    functions.MaskedLinearFunction
        The autograd ``Function`` object used to implement the module.

    Examples
    --------

    >>> in_features, out_features = 8, 5
    >>> # Lower triangular mask.
    >>> mask = torch.tril(torch.ones(out_features, in_features, dtype=torch.bool))
    >>> m = MaskedLinear(in_features, out_features, mask=mask)
    >>> input = torch.randn(20, in_features)
    >>> output = m(input)
    >>> print(output.size())
    torch.Size([20, 8])

    """

    def __init__(self, in_features, out_features, bias=True, mask=None):
        # Let nn.Linear register and initialize the parameters.
        super().__init__(in_features, out_features, bias=bias)

        # We don't need to propagate gradients through the mask so we
        # register it as a buffer.
        self.register_buffer('mask', mask)

        # Set the masked weights to 0.0. This effectively sets the
        # gradient of the masked parameters to zero even when weight
        # normalization (whose gradient has a component that depend
        # on the gradient w.r.t. g) is used.
        self.weight.data = self.weight.data * self.mask

    def n_parameters(self):
        """int: The total number of (unmasked) parameters."""
        if self.mask is None:
            n_parameters = self.weight.numel()
        else:
            n_parameters = (self.mask != 0).sum()
        if self.bias is not None:
            n_parameters += self.bias.numel()
        return n_parameters

    def forward(self, input):
        """
        Performs the forward computation.

        Parameters
        ----------
        input : torch.Tensor
            Input of shape ``(batch_size, *, in_features)`` where ``*``
            means any number of additional dimensions.

        Returns
        -------
        output : torch.Tensor
            Output of shape ``(batch_size, *, in_features)`` where ``*``
            is the same number number of additional dimensions in ``input``.

        """
        # If there is no mask, fall back to normal linear behavior.
        if self.mask is None:
            return super().forward(input)
        return linear.masked_linear(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, mask={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.mask
        )


# =============================================================================
# WEIGHT NORMALIZATION FOR MASKED LINEAR LAYER
# =============================================================================

class _ApplyMask:
    """NaN-safe mask application.

    Parameters
    ----------
    norm : bool, optional
        If True, the mask is applied to a norm vector (i.e., g) rather
        than a matrix (i.e., v or w). Default is False.
    inplace : bool, optional
        If True, the tensor is modified in place when ApplyTask is called.
        Otherwise, a copy is created.

    """

    def __init__(self, mask, dim=0, norm=False, inplace=True):
        # Precompute the masked indices.
        self.inplace = inplace
        self._zero_indices = None
        if mask is not None:
            if norm:
                # For g, we need to zet to zero only those vectors
                # that have zero norm because of the mask.
                self._zero_indices = torch.nonzero(norm_except_dim(mask, 2, dim).flatten() == 0.0)
            else:
                self._zero_indices = mask == 0.0

    def __call__(self, w):
        # An element-wise multiplication doesn't work if there are NaNs.
        if self._zero_indices is not None:
            if not self.inplace:
                w = w.clone()
            w.data[self._zero_indices] = 0.0
            return w
        return None


class MaskedWeightNorm(WeightNorm):
    """NaN-free implementation of weight normalization.

    Applying the normal weight normalization implemented with :func:`torch.nn.utils.weight_norm`
    results in NaN entries in the matrices when the mask covers an entire
    vector (thus making its norm zero). This takes care of this special
    case.

    See Also
    --------
    torch.nn.utils.weight_norm.WeightNorm

    """

    def __init__(self, name, dim, mask):
        super().__init__(name, dim)
        self.apply_mask = _ApplyMask(mask)

    def compute_weight(self, module):
        weight = super().compute_weight(module)
        return self.apply_mask(weight)

    @staticmethod
    def apply(module, name, dim, mask):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, MaskedWeightNorm) and hook.name == name:
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = MaskedWeightNorm(name, dim, mask)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        g = Parameter(norm_except_dim(weight, 2, dim).data)
        v = Parameter(weight.data)
        module.register_parameter(name + '_g', g)
        module.register_parameter(name + '_v', v)
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        # Register hook to zero out gradient in the masked weights.
        g.register_hook(_ApplyMask(mask, dim, norm=True))
        v.register_hook(_ApplyMask(mask))

        return fn


def masked_weight_norm(module, name='weight', dim=0):
    """NaN-free implementation of weight normalization.

    Applying the normal weight normalization implemented with :func:`torch.nn.utils.weight_norm`
    results in NaN entries in the matrices when the mask covers an entire
    vector (thus making its norm zero). This takes care of this special
    case.

    See Also
    --------
    torch.nn.utils.weight_norm.weight_norm

    """
    try:
        mask = module.mask
    except AttributeError:
        mask = None
    MaskedWeightNorm.apply(module, name, dim, mask)
    return module


def remove_masked_weight_norm(module, name='weight'):
    """Remove masked weighed normalization hooks.

    See Also
    --------
    torch.nn.utils.weight_norm.remove_weight_norm

    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, MaskedWeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))
