import math
import typing as t
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.nn import functional as F

from loss.barlow_twin_loss import BarlowTwins
from loss.entropy import KL_div, Entropy
from utils.general import average_list, simplex


class RedundancyCriterion(nn.Module):

    def __init__(self, *, eps: float = 1e-5, symmetric: bool = True, lamda: float = 1, alpha: float) -> None:
        super().__init__()
        self._eps = eps
        self.symmetric = symmetric
        self.lamda = lamda
        self.alpha = alpha

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        k = x_out.shape[1]
        p_i_j = compute_joint_2D_with_padding_zeros(x_out=x_out, x_tf_out=x_tf_out, symmetric=self.symmetric)
        p_i_j = p_i_j.view(k, k)
        self._p_i_j = p_i_j
        target = ((self.onehot_label(k=k, device=p_i_j.device) / k) * self.alpha + p_i_j * (1 - self.alpha))
        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)  # p_i should be the mean of the x_out
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric
        constrained = (-p_i_j * (- self.lamda * torch.log(p_j + self._eps)
                                 - self.lamda * torch.log(p_i + self._eps))
                       ).sum()
        pseudo_loss = -(target * (p_i_j + self._eps).log()).sum()
        return pseudo_loss + constrained

    @lru_cache()
    def onehot_label(self, k, device):
        label = torch.eye(k, device=device, dtype=torch.bool)
        return label

    def kl_criterion(self, dist: Tensor, prior: Tensor):
        return -(prior * (dist + self._eps).log() + (1 - prior) * (1 - dist + self._eps).log()).mean()

    def get_joint_matrix(self):
        if not hasattr(self, "_p_i_j"):
            raise RuntimeError()
        return self._p_i_j.detach().cpu().numpy()

    def set_ratio(self, alpha: float):
        """
        0 : entropy minimization
        1 : barlow-twin
        """
        assert 0 <= alpha <= 1, alpha
        if self.alpha != alpha:
            logger.trace(f"Setting alpha = {alpha}")
        self.alpha = alpha


class IIDSegmentationLoss(nn.Module):

    def __init__(self, *, eps: float = 1e-5, symmetric: bool = True, lamda: float = 1) -> None:
        super().__init__()
        self._eps = eps
        self.symmetric = symmetric
        self.lamda = lamda

    def forward(
            self, x_out: Tensor, x_tf_out: Tensor
    ) -> Tensor:
        k = x_out.shape[1]
        p_i_j = compute_joint_2D_with_padding_zeros(x_out=x_out, x_tf_out=x_tf_out, symmetric=self.symmetric)
        p_i_j = p_i_j.view(k, k)
        self._p_i_j = p_i_j
        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

        cluster_loss = (-p_i_j * (
                torch.log(p_i_j + self._eps)
                - self.lamda * torch.log(p_j + self._eps)
                - self.lamda * torch.log(p_i + self._eps))
                        ).sum()
        return cluster_loss

    def get_joint_matrix(self):
        if not hasattr(self, "_p_i_j"):
            raise RuntimeError()
        return self._p_i_j.detach().cpu().numpy()

KL_loss =KL_div()
ent_loss = Entropy()

def compute_joint_2D(x_out: Tensor, x_out_disp: Tensor, *, symmetric: bool = True, padding: int = 0):
    x_out = x_out.swapaxes(0, 1).contiguous()
    x_out_disp = x_out_disp.swapaxes(0, 1).contiguous()

    p_i_j = F.conv2d(
        input=x_out,
        weight=x_out_disp, padding=(int(padding), int(padding))
    )
    p_i_j = p_i_j - p_i_j.min().detach() + 1e-8

    # T x T x k x k
    p_i_j = p_i_j.permute(2, 3, 0, 1)
    p_i_j /= p_i_j.sum(dim=[2, 3], keepdim=True)  # norm

    # symmetrise, transpose the k x k part
    if symmetric:
        p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0
    p_i_j /= p_i_j.sum()  # norm
    return p_i_j.contiguous()


def compute_joint_2D_with_padding_zeros(x_out: Tensor, x_tf_out: Tensor, *, symmetric: bool = True):
    k = x_out.shape[1]
    x_out = x_out.swapaxes(0, 1).reshape(k, -1)
    N = x_out.shape[1]
    x_tf_out = x_tf_out.swapaxes(0, 1).reshape(k, -1)
    p_i_j = (x_out / math.sqrt(N)) @ (x_tf_out.t() / math.sqrt(N))
    # p_i_j = p_i_j - p_i_j.min().detach() + 1e-8

    # T x T x k x k
    # p_i_j /= p_i_j.sum()

    # symmetrise, transpose the k x k part
    if symmetric:
        p_i_j = (p_i_j + p_i_j.t()) / 2.0
    p_i_j = p_i_j.view(1, 1, k, k)
    return p_i_j.contiguous()


def compute_joint_distribution(x_out, displacement_map: (int, int), symmetric=True):
    n, d, h, w = x_out.shape
    padding_max = max(np.abs(displacement_map))
    padding_amount = (
        padding_max, padding_max, padding_max, padding_max)  # pad last dim by (1, 1) and 2nd to last by (2, 2)
    out = F.pad(x_out, padding_amount, "constant", 0)
    after_displacement = out[:, :, padding_max - displacement_map[0]:padding_max - displacement_map[0] + h,
                         padding_max - displacement_map[1]:padding_max - displacement_map[1] + w]

    x_out = x_out.reshape(n,d, h*w)
    after_displacement = after_displacement.reshape(n,d, h*w).transpose(2,1)

    p_i_j = (x_out @ after_displacement).mean(0).unsqueeze(0).unsqueeze(0)
    # p_i_j = (x_out @ after_displacement).unsqueeze(0).transpose(1,0)

    p_i_j = p_i_j - p_i_j.min().detach() + 1e-8
    # T x T x k x k
    p_i_j /= p_i_j.sum(dim=[2, 3], keepdim=True)  # norm

    # symmetrise, transpose the k x k part
    if symmetric:
        p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0
    p_i_j /= p_i_j.sum()  # norm

    # exclude batchsize
    # p_joint_posbs = []
    # for idx in range(int(n/3)):
    #     p_joint_pos = torch.zeros_like(p_i_j[0])
    #     for pos_idx in range(idx, n, int(n/3)):
    #         p_joint_pos = p_joint_pos + p_i_j[pos_idx,:,:,:]
    #     p_joint_pos = p_joint_pos / 3 # 3 patients in one batch
    #     p_joint_posbs.append(p_joint_pos)

    return p_i_j.contiguous()


def single_head_loss(clusters: Tensor, clustert: Tensor, *, displacement_maps: t.Sequence[t.Tuple[int, int]]):
    assert simplex(clustert) and simplex(clusters)

    align_loss_list = []
    for dis_map in displacement_maps:
        p_joint_S = compute_joint_distribution(
            x_out=clusters.detach(),
            displacement_map=(dis_map[0],
                              dis_map[1]))
        p_joint_T = compute_joint_distribution(
            x_out=clustert,
            displacement_map=(dis_map[0],
                              dis_map[1]))
        # align
        align_1disp_loss = torch.mean(torch.abs((p_joint_S.detach() - p_joint_T)))
        # align_1disp_loss = torch.abs(torch.cat(p_joint_S)  - torch.cat(p_joint_T)).mean()

        align_loss_list.append(align_1disp_loss)
    align_loss = average_list(align_loss_list)
    # todo: visualization.

    return align_loss, p_joint_S, p_joint_T

def compute_cross_correlation(x_out, displacement_map: (int, int)):
    n, d, h, w = x_out.shape
    padding_max = max(np.abs(displacement_map))
    padding_amount = (
        padding_max, padding_max, padding_max, padding_max)  # pad last dim by (1, 1) and 2nd to last by (2, 2)
    out = F.pad(x_out, padding_amount, "constant", 0)
    after_displacement = out[:, :, padding_max - displacement_map[0]:padding_max - displacement_map[0] + h,
                         padding_max - displacement_map[1]:padding_max - displacement_map[1] + w]

    assert x_out.shape[1] == after_displacement.shape[1]

    x_out_norm = (x_out - x_out.mean(0)) / (x_out.std(0) + 1e-16) # NXDXhw
    after_displacement_norm = (after_displacement - after_displacement.mean(0)) / (after_displacement.std(0) + 1e-16)

    cc = x_out_norm.transpose(1,0).reshape(d,n*h*w) @ after_displacement_norm.transpose(1,0).reshape(d,n*h*w).T
    return cc

def multi_resilution_cluster(clusters_S: t.List, clusters_T: t.List):
    low_res_clusters_S, low_res_clusters_T = [], []
    for cluster_s, cluster_t in zip(clusters_S, clusters_T):
        assert simplex(cluster_s)
        assert simplex(cluster_t)
        low_res_cluster_s = F.avg_pool2d(cluster_s, kernel_size=(2, 2))
        low_res_cluster_t = F.avg_pool2d(cluster_t, kernel_size=(2, 2))
        assert simplex(low_res_cluster_s)
        assert simplex(low_res_cluster_t)

        low_res_clusters_S.append(low_res_cluster_s)
        low_res_clusters_T.append(low_res_cluster_t)

    return low_res_clusters_S, low_res_clusters_T
