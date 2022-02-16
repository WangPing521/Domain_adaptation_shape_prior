from torch import nn
from torch import Tensor

from loss._base import LossClass
from loss.entropy import _check_reduction_params, Entropy
from utils.general import assert_list, simplex


class JSD_div(nn.Module, LossClass[Tensor]):
    """
    general JS divergence interface
    :<math>{\rm JSD}_{\pi_1, \ldots, \pi_n}(P_1, P_2, \ldots, P_n) = H\left(\sum_{i=1}^n \pi_i P_i\right) - \sum_{i=1}^n \pi_i H(P_i)</math>


    reduction (string, optional): Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._reduction = reduction
        self._eps = eps
        self._entropy_criterion = Entropy(reduction=reduction, eps=eps)

    def forward(self, *input_: Tensor) -> Tensor:
        assert assert_list(
            lambda x: simplex(x), input_
        ), f"input tensor should be a list of simplex."
        assert assert_list(
            lambda x: x.shape == input_[0].shape, input_
        ), "input tensor should have the same dimension"
        mean_prob = sum(input_) / len(input_)
        f_term = self._entropy_criterion(mean_prob)
        mean_entropy: Tensor = sum(
            list(map(lambda x: self._entropy_criterion(x), input_))
        ) / len(input_)
        assert f_term.shape == mean_entropy.shape
        return f_term - mean_entropy
