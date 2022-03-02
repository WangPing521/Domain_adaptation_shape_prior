from typing import Union

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from loss.entropy import Entropy, KL_div
from meters import AverageValueMeter
from scheduler.customized_scheduler import RampScheduler
from trainers.SourceTrainer import SourcebaselineTrainer
from utils.general import class2one_hot


class AverageValueMeterPrior(AverageValueMeter):

    def _summary(self):
        return self.sum / self.n


class entPlusPriorTrainer(SourcebaselineTrainer):

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
        self.ent_loss = Entropy()
        self.KL_loss = KL_div()
        self.prior = self.compute_prior(TrainS_loader)

    def run_step(self, s_data, t_data, cur_batch: int):
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

        with self.switch_bn(self.model, 0):
            pred_S = self.model(S_img).softmax(1)

        onehot_targetS = class2one_hot(S_target.squeeze(1), self.C)
        s_loss = self.crossentropy(pred_S, onehot_targetS)

        with self.switch_bn(self.model, 1):
            pred_T = self.model(T_img).softmax(1)

        align_loss = self.ent_loss(pred_T)

        # cluster_loss = self.KL_loss(pred_T.mean(dim=[0, 2, 3])[None,...], self.prior[None,...])
        cluster_loss = torch.abs(pred_T.mean(dim=[0, 2, 3])[None, ...] - self.prior[None,...]).mean()
        self.meters[f"train_dice"].add(
            pred_S.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )

        return s_loss, cluster_loss, align_loss

    @property
    def C(self): return int(self._config['Data_input']['num_class'])

    @torch.no_grad()
    def compute_prior(self, tra_loader):
        meter = AverageValueMeterPrior()

        for _, (data, *_) in zip(range(len(tra_loader)), tra_loader):
            target = class2one_hot(data[1], self.C).float()
            meter.add(target.mean(dim=[0, 2, 3, 4]), n=target.shape[0])

        prior = meter.summary().to(self.device)
        logger.info(f"Obtained source prior: {prior}")
        return prior
