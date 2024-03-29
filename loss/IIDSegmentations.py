import typing as t
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from loss.entropy import KL_div, Entropy
from utils.general import average_list, simplex
from utils.image_save_utils import plot_joint_matrix1

KL_loss =KL_div()
ent_loss = Entropy()

def compute_joint_distribution(x_out, displacement_map: (int, int), symmetric=True, cc_relation=False):
    n, d, h, w = x_out.shape
    after_displacement = x_out.roll(shifts=[displacement_map[0], displacement_map[1]], dims=[2, 3])
    if not cc_relation:
        assert simplex(after_displacement)

    x_out = x_out.reshape(n,d, h*w)
    after_displacement = after_displacement.reshape(n,d, h*w).transpose(2,1)

    p_i_j = (x_out @ after_displacement).mean(0).unsqueeze(0).unsqueeze(0)
    p_i_j = p_i_j.contiguous()
    p_i_j = p_i_j - p_i_j.min().detach() + 1e-8
    p_i_j /= p_i_j.sum(dim=[2, 3], keepdim=True)  # norm

    # symmetrise, transpose the k x k part
    if symmetric:
        p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0
    p_i_j /= p_i_j.sum()  # norm

    return p_i_j.contiguous()

def single_head_loss(clusters: Tensor, clustert: Tensor, *, displacement_maps: t.Sequence[t.Tuple[int, int]], cc_based=False, cur_batch, cur_epoch, vis):
    if not cc_based:
        assert simplex(clustert) and simplex(clusters)

    align_loss_list = []
    for dis_map in displacement_maps:
        p_joint_S = compute_joint_distribution(
            x_out=clusters.detach(),
            displacement_map=(dis_map[0],
                              dis_map[1]),
            cc_relation=cc_based)
        p_joint_T = compute_joint_distribution(
            x_out=clustert,
            displacement_map=(dis_map[0],
                              dis_map[1]),
            cc_relation=cc_based)
        # align
        align_1disp_loss = torch.mean(torch.abs((p_joint_S.detach() - p_joint_T)))
        # align_1disp_loss = KL_loss(p_joint_T, p_joint_S.detach())
        align_loss_list.append(align_1disp_loss)
        if cur_batch == 0:
            p_joint_S_fig = plot_joint_matrix1(p_joint_S, indicator="error")
            vis.add_figure(tag=f"joint_S_{dis_map[0]}_{dis_map[1]}", figure=p_joint_S_fig, global_step=cur_epoch, close=True)

            p_joint_T_fig = plot_joint_matrix1(p_joint_T, indicator="error")
            vis.add_figure(tag=f"joint_T_{dis_map[0]}_{dis_map[1]}", figure=p_joint_T_fig, global_step=cur_epoch, close=True)

    align_loss = average_list(align_loss_list)
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

def multi_resilution_cluster(clusters_S: t.List, clusters_T: t.List, cc_based=False, pool_size=2):
    low_res_clusters_S, low_res_clusters_T = [], []
    for cluster_s, cluster_t in zip(clusters_S, clusters_T):
        if not cc_based:
            assert simplex(cluster_s)
            assert simplex(cluster_t)
        low_res_cluster_s = F.avg_pool2d(cluster_s, kernel_size=(pool_size, pool_size))
        low_res_cluster_t = F.avg_pool2d(cluster_t, kernel_size=(pool_size, pool_size))
        if not cc_based:
            assert simplex(low_res_cluster_s)
            assert simplex(low_res_cluster_t)

        low_res_clusters_S.append(low_res_cluster_s)
        low_res_clusters_T.append(low_res_cluster_t)

    return low_res_clusters_S, low_res_clusters_T
