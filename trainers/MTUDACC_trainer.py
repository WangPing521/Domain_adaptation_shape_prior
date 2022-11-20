from typing import Union
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.nn import functional as F

from arch.utils import FeatureExtractor
from loss.IIDSegmentations import multi_resilution_cluster, single_head_loss
from scheduler.customized_scheduler import RampScheduler
from trainers.MTUDA_trainer import MTUDA_trainer
from utils.general import class2one_hot, average_list


class MTUDACCtrainer(MTUDA_trainer):

    def __init__(
            self,
            model: nn.Module,
            source_ema_model: nn.Module,
            target_ema_model: nn.Module,
            optimizer,
            scheduler,
            lkdScheduler,
            consScheduler,
            TrainS_loader: Union[DataLoader, _BaseDataLoaderIter],
            TrainT_loader: Union[DataLoader, _BaseDataLoaderIter],
            test_loader: Union[DataLoader, _BaseDataLoaderIter],
            *args,
            **kwargs
    ) -> None:
        super().__init__(model, source_ema_model, target_ema_model, optimizer, scheduler,
                         lkdScheduler, consScheduler, TrainS_loader, TrainT_loader, test_loader,
                         *args, **kwargs)
        self.extractor = FeatureExtractor(self.model, feature_names='Up_conv2')
        self.extractor.bind()

        self.ccalignScheduler = RampScheduler(**self._config['weights']["ccalignScheduler"])

        if self._config['DA']['displacement']:
            s = self._config['DA']['dis_scale']
            self.displacement_map_list = [[(0, 0)]]
            for scale in s:
                self.displacement_map_list.append([(-scale, -scale), (scale, scale), (-scale, 0), (scale, 0),
                                                  (0, scale), (0, -scale), (-scale, scale), (scale, -scale)])
            for s_idx in range(1, len(s)+1):
                for i in range(8):
                    self.displacement_map_list[0].append(self.displacement_map_list[s_idx][i])
            self.displacement_map_list = self.displacement_map_list[0]

        else:
            self.displacement_map_list = [(0, 0)]

    def run_step(self, s_data, t_data, cur_batch: int):
        S_img, S_target, S_filename = (
            s_data[0][0][0].to(self.device),
            s_data[0][0][1].to(self.device),
            s_data[0][1],
        )

        T_img, T_target, T_filename = (
            t_data[0][0][0].to(self.device),
            t_data[0][0][1].to(self.device),
            t_data[0][1],
        )

        T2S_img, T2S_target, T2S_filename = (
            t_data[1][0][0].to(self.device),
            t_data[1][0][1].to(self.device),
            t_data[1][1],
        )
        S2T2S_img, S2T2S_target, S2T2S_filename = (
            s_data[2][0][0].to(self.device),
            s_data[2][0][1].to(self.device),
            s_data[2][1],
        )

        S2T_img, S2T_target, S2T_filename = (
            s_data[1][0][0].to(self.device),
            s_data[1][0][1].to(self.device),
            s_data[1][1],
        )
        T2S2T_img, T2S2T_target, T2S2T_filename = (
            t_data[2][0][0].to(self.device),
            t_data[2][0][1].to(self.device),
            t_data[2][1],
        )
        S_target = self._rising_augmentation(S_target.float(), mode="feature", seed=cur_batch)
        T_target = self._rising_augmentation(T_target.float(), mode="feature", seed=cur_batch)

        S_img = self._rising_augmentation(S_img, mode="image", seed=cur_batch)
        T_img = self._rising_augmentation(T_img, mode="image", seed=cur_batch)
        T2S_img = self._rising_augmentation(T2S_img, mode="image", seed=cur_batch)
        S2T2S_img = self._rising_augmentation(S2T2S_img, mode="image", seed=cur_batch)
        S2T_img = self._rising_augmentation(S2T_img, mode="image", seed=cur_batch)
        T2S2T_img = self._rising_augmentation(T2S2T_img, mode="image", seed=cur_batch)

        with self.extractor.enable_register(True):
            self.extractor.clear()
            pred_s_0 = self.model(S_img).softmax(1)
            feature_S = next(self.extractor.features())
        with self.extractor.enable_register(True):
            self.extractor.clear()
            pred_t2s_0 = self.model(T2S_img).softmax(1)
            feature_t2s = next(self.extractor.features())
        with self.extractor.enable_register(True):
            self.extractor.clear()
            pred_s2t2s_1 = self.model(S2T2S_img).softmax(1)
            feature_s2t2s = next(self.extractor.features())

        # model loss
        onehot_targetS = class2one_hot(S_target.squeeze(1), self._config['Data_input']['num_class'])
        sup_loss = 0.5 * (self.crossentropy(pred_s_0, onehot_targetS) + self.dice_loss(pred_s_0, onehot_targetS))

        noiseS = torch.clamp(torch.randn_like(S_img) * self.noise, -0.2, 0.2).to(self.device)
        noiseT = torch.clamp(torch.randn_like(T_img) * self.noise, -0.2, 0.2).to(self.device)

        S_img_noise = S_img + noiseS
        T2S_img_noise = T2S_img + noiseT
        with torch.no_grad():
            pred_s_ema = self.source_ema_model(S_img_noise).softmax(1)
            pred_t2s_ema = self.source_ema_model(T2S_img_noise).softmax(1)
        # semantic
        lkd_loss = self.kl(pred_s_0, pred_s_ema) + self.kl(pred_t2s_0, pred_t2s_ema)

        T_img_noise = T_img + noiseT
        S2T_img_noise = S2T_img + noiseS
        T2S2T_img_noise = T2S2T_img + noiseT

        with torch.no_grad():
            pred_t_ema = self.target_ema_model(T_img_noise).softmax(1)
            pred_s2t_ema = self.target_ema_model(S2T_img_noise).softmax(1)
            pred_t2s2t_ema = self.target_ema_model(T2S2T_img_noise).softmax(1)
        #structural
        consistency = F.mse_loss(self.entropy(pred_s2t_ema), self.entropy(pred_s_0)) + \
                         F.mse_loss(self.entropy(pred_s2t_ema), self.entropy(pred_s2t2s_1)) + \
                         F.mse_loss(self.entropy(pred_t_ema), self.entropy(pred_t2s_0)) + \
                         F.mse_loss(self.entropy(pred_t2s2t_ema), self.entropy(pred_t2s_0))

        CC_S = [feature_S]
        CC_s2t2s = [feature_s2t2s]
        CC_t2s = [feature_t2s]
        assert len(CC_S) == len(CC_s2t2s)

        align_loss_multires1,  align_loss_multires2= []
        for rs in range(self._config['DA']['multi_scale']):
            if rs:
                clusters_S, clusters_s2t2s = multi_resilution_cluster(CC_S, CC_s2t2s, cc_based=True, pool_size=2)
            align_losses, p_joint_Ss, p_joint_s2t2ss = \
                zip(*[single_head_loss(clusters, clustert, displacement_maps=self.displacement_map_list,
                                       cc_based=True, cur_batch=cur_batch, cur_epoch=self.cur_epoch,
                                       vis=self.writer) for
                      clusters, clustert in zip(clusters_S, clusters_s2t2s)])

            align_loss = sum(align_losses) / len(align_losses)
            align_loss_multires1.append(align_loss)
            align_loss1 = average_list(align_loss_multires1)

        for rs in range(self._config['DA']['multi_scale']):
            if rs:
                clusters_S, clusters_T2S = multi_resilution_cluster(CC_S, CC_t2s, cc_based=True, pool_size=2)
            align_losses, p_joint_Ss, p_joint_T2Ss = \
                zip(*[single_head_loss(clusters, clustert, displacement_maps=self.displacement_map_list,
                                       cc_based=True, cur_batch=cur_batch, cur_epoch=self.cur_epoch,
                                       vis=self.writer) for
                      clusters, clustert in zip(clusters_S, clusters_T2S)])

            align_loss = sum(align_losses) / len(align_losses)
            align_loss_multires2.append(align_loss)
        align_loss2 = average_list(align_loss_multires2)

        ccalign = 0.5 * (align_loss1 + align_loss2)

        self.meters[f"trainT_dice"].add(
            pred_s_0.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )

        return sup_loss, lkd_loss, consistency, ccalign