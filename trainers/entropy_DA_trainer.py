from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from arch.utils import FeatureExtractor
from loss.IIDSegmentations import single_head_loss
from loss.entropy import Entropy
from scheduler.customized_scheduler import RampScheduler
from trainers.SourceTrainer import SourcebaselineTrainer
from utils.general import class2one_hot
from utils.image_save_utils import plot_joint_matrix1


class EntropyDA(SourcebaselineTrainer):

    def __init__(self, TrainS_loader: Union[DataLoader, _BaseDataLoaderIter],
                 TrainT_loader: Union[DataLoader, _BaseDataLoaderIter],
                 valT_loader: Union[DataLoader, _BaseDataLoaderIter],
                 test_loader: Union[DataLoader, _BaseDataLoaderIter],
                 weight_scheduler: RampScheduler,
                 weight_cluster: RampScheduler,
                 model: nn.Module,
                 optimizer, scheduler, *args, **kwargs) -> None:
        super().__init__(model, optimizer, scheduler, TrainS_loader, TrainT_loader, valT_loader, test_loader,
                         weight_scheduler, weight_cluster, *args, **kwargs)
        self.ent_loss = Entropy()
        self.extractor = FeatureExtractor(self.model, feature_names='Up_conv2')
        self.extractor.bind()

    def run_step(self, s_data, t_data, cur_batch: int):
        C = int(self._config['Data_input']['num_class'])
        S_img, S_target, S_filename = (
            s_data[0][0].to(self.device),
            s_data[0][1].to(self.device),
            s_data[1],
        )
        T_img, T_target, T_filename = (
            t_data[0][0].to(self.device),
            t_data[0][1].to(self.device),
            t_data[1],
        )
        S_img = self._rising_augmentation(S_img, mode="image", seed=cur_batch)
        S_target = self._rising_augmentation(S_target.float(), mode="feature", seed=cur_batch)
        T_img = self._rising_augmentation(T_img, mode="image", seed=cur_batch)
        # T_target = self._rising_augmentation(T_target.float(), mode="feature", seed=cur_batch)

        # data augmentation
        with self.switch_bn(self.model, 0), self.extractor.enable_register(True):
            self.extractor.clear()
            pred_S = self.model(S_img).softmax(1)
            feature_S = next(self.extractor.features())


        onehot_targetS = class2one_hot(S_target.squeeze(1), C)
        s_loss = self.crossentropy(pred_S, onehot_targetS)

        with self.switch_bn(self.model, 1), self.extractor.enable_register(True):
            self.extractor.clear()
            pred_T = self.model(T_img).softmax(1)
            feature_T = next(self.extractor.features())

        clusters_S = [feature_S]
        clusters_T = [feature_T]

        align_losses, p_joint_Ss, p_joint_Ts = \
            zip(*[single_head_loss(clusters, clustert, displacement_maps=self.displacement_map_list, cc_based=True, cur_batch=cur_batch, cur_epoch=self.cur_epoch,
                                   vis=self.writer) for clusters, clustert in zip(clusters_S, clusters_T)])
        p_joint_S = p_joint_Ss[-1]
        p_joint_T = p_joint_Ts[-1]
        joint_error = torch.abs(p_joint_S - p_joint_T)
        joint_error_percent = joint_error / p_joint_S
        if cur_batch == 0:
            joint_error_fig = plot_joint_matrix1(joint_error, indicator="error")
            joint_error_percent_fig = plot_joint_matrix1(joint_error_percent, indicator="percenterror")
            self.writer.add_figure(tag=f"error_joint", figure=joint_error_fig, global_step=self.cur_epoch,
                                   close=True, )
            self.writer.add_figure(tag=f"error_percent", figure=joint_error_percent_fig, global_step=self.cur_epoch,
                                   close=True, )

        align_loss = self.ent_loss(pred_T)












        self.meters[f"train_dice"].add(
            pred_S.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )
        cluster_loss = torch.tensor(0, dtype=torch.float, device=pred_S.device)

        return s_loss, cluster_loss, align_loss
