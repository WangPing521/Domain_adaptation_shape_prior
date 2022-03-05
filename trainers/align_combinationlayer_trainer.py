from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from arch.projectors import DenseClusterHead
from arch.utils import FeatureExtractor
from loss.IIDSegmentations import single_head_loss, multi_resilution_cluster
from scheduler.customized_scheduler import RampScheduler
from trainers.SourceTrainer import SourcebaselineTrainer
from utils.general import class2one_hot, average_list
from utils.image_save_utils import plot_joint_matrix, FeatureMapSaver, plot_seg


class mutli_aligntrainer(SourcebaselineTrainer):

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

        self.optimizer.add_param_group({'params': self.projector.parameters(),
                                        })

        self.extractor = FeatureExtractor(self.model, feature_names=self._config['DA']['align_layer']['name'])
        self.extractor.bind()
        self.saver = FeatureMapSaver(save_dir=self._save_dir)

    def run_step(self, s_data, t_data, cur_batch: int):
        extracted_layer = self.extractor.feature_names[0]
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
        T_target = self._rising_augmentation(T_target.float(), mode="feature", seed=cur_batch)


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

        # projector cluster --->joint
        clusters_S1 = [pred_S]
        clusters_T1 = [pred_T]

        clusters_S2 = self.projector(feature_S)
        clusters_T2 = self.projector(feature_T)

        assert len(clusters_S1) == len(clusters_T1)
        assert len(clusters_S2) == len(clusters_T2)

        align_loss_multires1, cluster_losses_multires1, align_loss_multires2, cluster_losses_multires2 = [], [], [], []
        p_jointS_list2, p_jointT_list2 = [], []

        for rs in range(self._config['DA']['multi_scale']):
            if rs:
                clusters_S1, clusters_T1 = multi_resilution_cluster(clusters_S1, clusters_T1)
                clusters_S2, clusters_T2 = multi_resilution_cluster(clusters_S2, clusters_T2)
            align_losses1, cluster_losses1, p_joint_Ss1, p_joint_Ts1 = \
                zip(*[single_head_loss(clusters1, clustert1, displacement_maps=self.displacement_map_list) for
                      clusters1, clustert1 in zip(clusters_S1, clusters_T1)])
            align_losses2, cluster_losses2, p_joint_Ss2, p_joint_Ts2 = \
                zip(*[single_head_loss(clusters2, clustert2, displacement_maps=self.displacement_map_list) for
                      clusters2, clustert2 in zip(clusters_S2, clusters_T2)])

            align_loss1 = sum(align_losses1) / len(align_losses1)
            cluster_loss1 = sum(cluster_losses1) / len(cluster_losses1)
            align_loss_multires1.append(align_loss1)
            cluster_losses_multires1.append(cluster_loss1)

            align_loss2 = sum(align_losses2) / len(align_losses2)
            cluster_loss2 = sum(cluster_losses2) / len(cluster_losses2)
            align_loss_multires2.append(align_loss2)
            cluster_losses_multires2.append(cluster_loss2)

            p_jointS_list2.append(p_joint_Ss1[-1])
            p_jointT_list2.append(p_joint_Ts1[-1])

        align_loss1 = average_list(align_loss_multires1)
        cluster_loss1 = average_list(cluster_losses_multires1)

        align_loss2 = average_list(align_loss_multires2)
        cluster_loss2 = average_list(cluster_losses_multires2)

        cluster_loss = cluster_loss1 + self._config['DA']['weight1'] * cluster_loss2
        align_loss = align_loss1 + self._config['DA']['weight2'] * align_loss2
        entT_loss = self.ent_loss(pred_T) # entropy on target

        # for visualization
        p_joint_S = sum(p_jointS_list2) / len(p_jointS_list2)
        p_joint_T = sum(p_jointT_list2) / len(p_jointT_list2)
        clusters = clusters_S1[-1]
        clustert = clusters_T1[-1]

        self.meters[f"train_dice"].add(
            pred_S.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )

        if cur_batch == 0:
            source_joint_fig = plot_joint_matrix(p_joint_S)
            target_joint_fig = plot_joint_matrix(p_joint_T)
            self.writer.add_figure(tag=f"source_joint", figure=source_joint_fig, global_step=self.cur_epoch,
                                   close=True, )
            self.writer.add_figure(tag=f"target_joint", figure=target_joint_fig, global_step=self.cur_epoch,
                                   close=True, )
            self.saver.save_map(imageS=S_img, imageT=T_img, feature_mapS=clusters, feature_mapT=clustert,
                                cur_epoch=self.cur_epoch, cur_batch_num=cur_batch, save_name="cluster"
                                )
            source_seg = plot_seg(S_img[10], pred_S.max(1)[1][10])
            target_seg = plot_seg(T_img[10], pred_T.max(1)[1][10])
            self.writer.add_figure(tag=f"train_source_seg", figure=source_seg, global_step=self.cur_epoch, close=True)
            self.writer.add_figure(tag=f"train_target_seg", figure=target_seg, global_step=self.cur_epoch, close=True)

        return s_loss, entT_loss, align_loss
