import collections
from itertools import repeat

import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)


def _check_head_type(head_type):
    return head_type in ("mlp", "linear")


def _check_pool_name(pool_name):
    return pool_name in ("adaptive_avg", "adaptive_max", "identical", "none")


def init_dense_sub_header(head_type, input_dim, hidden_dim, num_clusters, normalize, T):
    if head_type == "linear":
        return nn.Sequential(
            nn.Conv2d(input_dim, num_clusters, 1, 1, 0),
            Normalize() if normalize else Identical(),
            SoftmaxWithT(1, T=T)
        )
    return nn.Sequential(
        nn.Conv2d(input_dim, hidden_dim, 1, 1, 0),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv2d(hidden_dim, num_clusters, 1, 1, 0),
        Normalize() if normalize else Identical(),
        SoftmaxWithT(1, T=T)
    )

def get_pool_component(pool_name, spatial_size: Tuple[int, int]):
    return {
        "adaptive_avg": nn.AdaptiveAvgPool2d(spatial_size),
        "adaptive_max": nn.AdaptiveMaxPool2d(spatial_size),
        None: Identical(),
        "none": Identical(),
        "identical": Identical(),
    }[pool_name]


class SoftmaxWithT(nn.Softmax):

    def __init__(self, dim, T: float = 1.0) -> None:
        super().__init__(dim)
        self._T = T

    def forward(self, input: Tensor) -> Tensor:
        input /= self._T
        return super().forward(input)


class Normalize(nn.Module):

    def __init__(self, dim=1) -> None:
        super().__init__()
        self._dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self._dim)


class Identical(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return input


class _ProjectorHeadBase(nn.Module):

    def __init__(self, *, input_dim: int, output_dim: int, head_type: str, normalize: bool, pool_name="adaptive_avg",
                 spatial_size=(1, 1)):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        assert _check_head_type(head_type=head_type), head_type
        self._head_type = head_type
        self._normalize = normalize
        assert _check_pool_name(pool_name=pool_name), pool_name
        self._pool_name = pool_name
        self._spatial_size = _pair(spatial_size)

        self._pooling_module = get_pool_component(self._pool_name, self._spatial_size)

    def _record_message(self):
        return f"Initializing {self.__class__.__name__} with {self._head_type} dense head " \
               f"({self._input_dim}:{self._output_dim}), " \
               f"{' normalization ' if self._normalize else ''}" \
               f"{f'{self._pool_name} with {self._spatial_size}' if 'adaptive' in self._pool_name else ''} "


# head for IIC segmentation clustering
device='cuda'
class DenseClusterHead(_ProjectorHeadBase):
    """
    this classification head uses the loss for IIC segmentation, which consists of multiple heads
    """

    def __init__(self, *, input_dim: int, num_clusters=10, hidden_dim=64, num_subheads=10, T=1,
                 head_type: str = "linear", normalize: bool = False):
        super().__init__(input_dim=input_dim, output_dim=num_clusters, head_type=head_type, normalize=normalize,
                         pool_name="none", spatial_size=(1, 1))
        self._T = T

        headers = [
            init_dense_sub_header(head_type=head_type, input_dim=self._input_dim, hidden_dim=hidden_dim,
                                  num_clusters=num_clusters, normalize=self._normalize, T=self._T)
            for _ in range(num_subheads)
        ]
        self._headers = nn.ModuleList(headers).to(device)
        message = self._record_message()
        logger.debug(message)

    def forward(self, features):
        return [x(features) for x in self._headers]