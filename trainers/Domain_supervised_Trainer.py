from typing import Union

import rising.random as rr
import rising.transforms as rt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from arch.projectors import DenseClusterHead
from arch.utils import FeatureExtractor
from loss.IIDSegmentations import compute_joint_distribution, IIDSegmentationLoss
from scheduler.customized_scheduler import RampScheduler
from trainers.SourceTrainer import SourcebaselineTrainer
from utils.general import class2one_hot
from utils.image_save_utils import plot_joint_matrix, FeatureMapSaver
from utils.rising import RisingWrapper


class DomainsupervisedTrainer(SourcebaselineTrainer):

    def __init__(self, TrainS_loader: Union[DataLoader, _BaseDataLoaderIter],
                 TrainT_loader: Union[DataLoader, _BaseDataLoaderIter],
                 valS_loader: Union[DataLoader, _BaseDataLoaderIter],
                 valT_loader: Union[DataLoader, _BaseDataLoaderIter], weight_scheduler: RampScheduler,
                 weight_cluster: RampScheduler,
                 model: nn.Module,
                 optimizer, scheduler, *args, **kwargs) -> None:

        super().__init__(model, optimizer, scheduler, TrainS_loader, TrainT_loader, valS_loader, valT_loader,
                         weight_scheduler, weight_cluster, *args, **kwargs)
        self._trainS_loader = TrainS_loader
        self._trainT_loader = TrainT_loader
        self._valS_loader = valS_loader
        self._valT_loader = valT_loader
        self._weight_scheduler = weight_scheduler
        self._weight_cluster = weight_cluster
        self.projector = DenseClusterHead(
            input_dim=self.model.get_channel_dim(self._config['DA']['align_layer']['name']),
            num_clusters=self._config['DA']['align_layer']['clusters'])

        self.extractor = FeatureExtractor(self.model, feature_names=self._config['DA']['align_layer']['name'])
        self.extractor.bind()
        self.saver = FeatureMapSaver(save_dir=self._save_dir)
        self.IICLoss = IIDSegmentationLoss()

        geometric_transform = rt.Compose(
            rt.BaseAffine(
                scale=rr.UniformParameter(0.5, 1.5),
                rotation=rr.UniformParameter(-30, 30), degree=True,
                translation=rr.UniformParameter(-0.2, 0.2), grad=True,
                interpolation_mode="nearest"
            ),
            rt.Mirror(dims=[0, 1], p_sample=0.5, grad=True)
        )
        intensity_transform = rt.Compose(
            rt.GammaCorrection(gamma=rr.UniformParameter(0.8, 1.2), grad=True),
            rt.GaussianNoise(mean=0, std=0.01),
        )

        self._rising_augmentation = RisingWrapper(
            geometry_transform=geometric_transform, intensity_transform=intensity_transform
        )

    def run_step(self, s_data, t_data, cur_batch: int):
        extracted_layer = self.extractor.feature_names[0]
        C = int(self._config['Data_input']['num_class'])
        S_img, S_target, S_filename = (
            s_data[0][0].to(self.device),
            s_data[0][1].to(self.device),
            s_data[1],
        )
        S_img = self._rising_augmentation(S_img, mode="image", seed=cur_batch)
        S_target = self._rising_augmentation(S_target.float(), mode="feature", seed=cur_batch)

        T_img, T_target, T_filename = (
            t_data[0][0].to(self.device),
            t_data[0][1].to(self.device),
            t_data[1],
        )
        T_img = self._rising_augmentation(T_img, mode="image", seed=cur_batch)
        T_target = self._rising_augmentation(T_target.float(), mode="feature", seed=cur_batch)


        with self.switch_bn(self.model, 0), self.extractor.enable_register(True):
            self.extractor.clear()
            pred_S = self.model(S_img).softmax(1)

        onehot_targetS = class2one_hot(S_target.squeeze(1), C)
        s_loss = self.crossentropy(pred_S, onehot_targetS)


        with self.switch_bn(self.model, 1), self.extractor.enable_register(True):
            self.extractor.clear()
            pred_T = self.model(T_img).softmax(1)

        onehot_targetT = class2one_hot(T_target.squeeze(1), C)
        t_loss = self.crossentropy(pred_T, onehot_targetT)

        clusters_S = pred_S
        clusters_T = pred_T


        assert len(clusters_S) == len(clusters_T)

        def single_head_loss(clusters, clustert):
            # cluster_loss = 0.5 * self.IICLoss(clusters, clusters) + 0.5 * self.IICLoss(clustert, clustert)
            cluster_loss = self.IICLoss(clustert, clustert)
            p_joint_S = compute_joint_distribution(
                x_out=clusters,
                displacement_map=(self._config['DA']['displacement']['map_x'],
                                  self._config['DA']['displacement']['map_y']))
            p_joint_T = compute_joint_distribution(
                x_out=clustert,
                displacement_map=(self._config['DA']['displacement']['map_x'],
                                  self._config['DA']['displacement']['map_y']))
            # cluster loss
            # align
            align_loss = torch.mean(torch.abs((p_joint_S - p_joint_T)))
            return align_loss, cluster_loss, p_joint_S, p_joint_T

        align_losses, cluster_losses, p_joint_Ss, p_joint_Ts = \
            zip(*[single_head_loss(clusters, clustert) for clusters, clustert in zip([clusters_S], [clusters_T])])

        # for visualization
        p_joint_S = p_joint_Ss[-1]
        p_joint_T = p_joint_Ts[-1]

        self.meters[f"train_dice"].add(
            pred_S.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )

        self.meters[f"trainT_dice"].add(
            pred_T.max(1)[1],
            T_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in T_filename],
        )

        if cur_batch == 0:
            source_joint_fig = plot_joint_matrix(p_joint_S)
            target_joint_fig = plot_joint_matrix(p_joint_T)
            self.writer.add_figure(tag=f"source_joint", figure=source_joint_fig, global_step=self.cur_epoch,
                                   close=True, )
            self.writer.add_figure(tag=f"target_joint", figure=target_joint_fig, global_step=self.cur_epoch,
                                   close=True, )
            self.saver.save_map(imageS=S_img, imageT=T_img, feature_mapS=pred_S, feature_mapT=pred_T,
                                cur_epoch=self.cur_epoch, cur_batch_num=cur_batch, save_name="cluster"
                                )

        align_loss = torch.tensor(0, dtype=torch.float, device=pred_S.device)
        cluster_loss = torch.tensor(0, dtype=torch.float, device=pred_S.device)
        s_loss = t_loss + s_loss
        return s_loss, cluster_loss, align_loss
