from typing import Optional, Union, List
from torch.nn import functional as F

from torch import Tensor
from torch import nn
import torch

from utils.general import simplex, assert_list


def _check_reduction_params(reduction):
    assert reduction in (
        "mean",
        "sum",
        "none",
    ), "reduction should be in ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``, given {}".format(
        reduction
    )

class Entropy(nn.Module):
    r"""General Entropy interface
    the definition of Entropy is - \sum p(xi) log (p(xi))
    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._eps = eps
        self._reduction = reduction

    def forward(self, input_: Tensor) -> Tensor:
        assert input_.shape.__len__() >= 2
        b, _, *s = input_.shape
        assert simplex(input_), f"Entropy input should be a simplex"
        e = input_ * (input_ + self._eps).log()
        e = -1.0 * e.sum(1)
        assert e.shape == torch.Size([b, *s])
        if self._reduction == "mean":
            return e.mean()
        elif self._reduction == "sum":
            return e.sum()
        else:
            return e


class SimplexCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-16) -> None:
        super().__init__()
        _check_reduction_params(reduction)
        self._reduction = reduction
        self._eps = eps

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        if not kwargs.get("disable_assert"):
            assert not target.requires_grad
            assert prob.requires_grad
            assert prob.shape == target.shape
            assert simplex(prob)
            assert simplex(target)
        b, c, *_ = target.shape
        ce_loss = -target * torch.log(prob + self._eps)
        if self._reduction == "mean":
            return ce_loss.mean()
        elif self._reduction == "sum":
            return ce_loss.sum()
        else:
            return ce_loss


class KL_div(nn.Module):
    r"""
    KL(p,q)= -\sum p(x) * log(q(x)/p(x))
    where p, q are distributions
    p is usually the fixed one like one hot coding
    p is the target and q is the distribution to get approached.

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16, weight: Union[List[float], Tensor] = None, verbose=True):
        super().__init__()
        self._eps = eps
        self._reduction = reduction
        self._weight: Optional[Tensor] = weight
        if weight is not None:
            assert isinstance(weight, (list, Tensor)), type(weight)
            if isinstance(weight, list):
                assert assert_list(lambda x: isinstance(x, (int, float)), weight)
                self._weight = torch.Tensor(weight).float()
            else:
                self._weight = weight.float()
            # normalize weight:
            self._weight = self._weight / self._weight.sum()
        if verbose:
            print(f"Initialized {self.__class__.__name__} \nwith weight={self._weight} and reduction={self._reduction}.")

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        if not kwargs.get("disable_assert"):
            assert prob.shape == target.shape
            assert simplex(prob), prob
            assert simplex(target), target
            assert not target.requires_grad
            assert prob.requires_grad
        b, c, *hwd = target.shape
        kl = (-target * torch.log((prob + self._eps) / (target + self._eps)))
        if self._weight is not None:
            assert len(self._weight) == c
            weight = self._weight.expand(b, *hwd, -1).transpose(-1, 1).detach()
            kl *= weight.to(kl.device)
        kl = kl.sum(1)
        if self._reduction == "mean":
            return kl.mean()
        elif self._reduction == "sum":
            return kl.sum()
        else:
            return kl


def jaccard_loss(logits, label, eps=1e-7, activation=True):
    """
    Computes the Jaccard loss, a.k.a the IoU loss.
    :param true: a tensor of shape [B, H, W] or [B, C, H, W] or [B, 1, H, W].
    :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
    :param eps: added to the denominator for numerical stability.
    :param activation: if apply the activation function before calculating the loss.
    :return: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[label.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        probas = F.softmax(logits, dim=1) if activation else logits

    true_1_hot = label.type(probas.type())
    dims = (0,) + tuple(range(2, label.ndimension()))
    probas = probas.contiguous()
    true_1_hot = true_1_hot.contiguous()
    intersection = probas * true_1_hot
    intersection = torch.sum(intersection, dims)
    cardinality = probas + true_1_hot
    cardinality = torch.sum(cardinality, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return 1 - jacc_loss


def batch_NN_loss(x, y):
    """
    calculate the distance loss between two point sets
    :param x: a point sets
    :param y: another point sets
    :return: the loss
    """
    def batch_pairwise_dist(x, y):
        """
        compute batch-wise distances of two point sets
        :param x: a point set
        :param y: another point set
        :return: the distance matrix
        """
        # 32, 2500, 3
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
        rx = xx[:, diag_ind, diag_ind]
        rx = rx.unsqueeze(1)
        rx = rx.expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    bs, num_points, points_dim = x.size()
    dist1 = torch.sqrt(batch_pairwise_dist(x, y) + 0.00001)
    values1, indices1 = dist1.min(dim=2)

    dist2 = torch.sqrt(batch_pairwise_dist(y, x) + 0.00001)
    values2, indices2 = dist2.min(dim=2)
    a = torch.div(torch.sum(values1,1), num_points)
    b = torch.div(torch.sum(values2,1), num_points)
    sum = torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)
    return sum

