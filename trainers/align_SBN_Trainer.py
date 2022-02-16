from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from arch.projectors import DenseClusterHead
from loss.IIDSegmentations import compute_joint_distribution
from scheduler.customized_scheduler import RampScheduler
from trainers.SourceTrainer import SourcebaselineTrainer
from utils.general import class2one_hot, average_list


class align_SBN_Trainer(SourcebaselineTrainer):

    def __init__(self, TrainS_loader: Union[DataLoader, _BaseDataLoaderIter],
                 TrainT_loader: Union[DataLoader, _BaseDataLoaderIter],
                 valS_loader: Union[DataLoader, _BaseDataLoaderIter],
                 valT_loader: Union[DataLoader, _BaseDataLoaderIter], weight_scheduler: RampScheduler, model: nn.Module,
                 optimizer, scheduler, *args, **kwargs) -> None:

        super().__init__(model, optimizer, scheduler, TrainS_loader, TrainT_loader, valS_loader, valT_loader,
                         weight_scheduler, *args, **kwargs)
        self._trainS_loader = TrainS_loader
        self._trainT_loader = TrainT_loader
        self._valS_loader = valS_loader
        self._valT_loader = valT_loader
        self._weight_scheduler = weight_scheduler
        self.projector = DenseClusterHead(
            input_dim=self.model.get_channel_dim(self._config['DA']['align_layer']['name']),
            num_clusters=self._config['DA']['align_layer']['clusters'])

    def run_step(self, s_data, t_data):
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

        pred_S = self.model(S_img).softmax(1)

        onehot_targetS = class2one_hot(
            S_target.squeeze(1), self._config['Data_input']['num_class']
        )
        s_loss = self.crossentropy(pred_S, onehot_targetS)

        if self._config['DA']['align_layer']['name'] in ['Up_conv5', 'Up_conv4', 'Up_conv3', 'Up_conv2']:
            feature_S = self.model(S_img, until=self._config['DA']['align_layer']['name'])
            feature_T = self.model(T_img, until=self._config['DA']['align_layer']['name'])

            # projector cluster --->joint
            clusters_S = average_list(self.projector(feature_S))
            clusters_T = average_list(self.projector(feature_T))

        elif self._config['DA']['align_layer']['name'] == 'Deconv_1x1':
            pred_T = self.model(T_img).softmax(1)
            clusters_S = pred_S
            clusters_T = pred_T
        else:
            raise RuntimeError(self._config['DA']['align_layer']['name'])
        # displacement_join
        p_joint_S = compute_joint_distribution(x_out=clusters_S,
                                               displacement_map=(self._config['DA']['displacement']['map_x'],
                                                                 self._config['DA']['displacement']['map_y']))
        p_joint_T = compute_joint_distribution(x_out=clusters_T,
                                               displacement_map=(self._config['DA']['displacement']['map_x'],
                                                                 self._config['DA']['displacement']['map_y']))

        # align prediction
        align_loss = torch.mean((p_joint_S.mean(dim=0) - p_joint_T.mean(dim=0)) ** 2)

        self.meters[f"train_dice"].add(
            pred_S.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )

        return s_loss, align_loss, p_joint_S, p_joint_T
