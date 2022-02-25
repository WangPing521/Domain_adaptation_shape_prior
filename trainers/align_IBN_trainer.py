from typing import Union

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from arch.projectors import DenseClusterHead
from arch.utils import FeatureExtractor
from loss.IIDSegmentations import single_head_loss, multi_resilution_cluster
from scheduler.customized_scheduler import RampScheduler
from trainers.SourceTrainer import SourcebaselineTrainer
from utils.general import class2one_hot, average_list, simplex
from utils.image_save_utils import plot_joint_matrix, FeatureMapSaver, plot_seg


class align_IBNtrainer(SourcebaselineTrainer):

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
        # T_target = self._rising_augmentation(T_target.float(), mode="feature", seed=cur_batch)

        with self.switch_bn(self.model, 0), self.extractor.enable_register(True):
            self.extractor.clear()
            pred_S = self.model(S_img).softmax(1)
            feature_S = next(self.extractor.features())

        onehot_targetS = class2one_hot(S_target.squeeze(1), C)
        s_loss = self.crossentropy(pred_S, onehot_targetS)

        if extracted_layer == 'Deconv_1x1':
            with self.switch_bn(self.model, 1), self.extractor.enable_register(True):
                pred_T = self.model(T_img).softmax(1)
            clusters_S = [pred_S]
            clusters_T = [pred_T]

        else:
            with self.switch_bn(self.model, 1), self.extractor.enable_register(True):
                self.extractor.clear()
                _ = self.model(T_img, until=extracted_layer)
                feature_T = next(self.extractor.features())

            # projector cluster --->joint
            clusters_S = self.projector(feature_S)
            clusters_T = self.projector(feature_T)

        assert simplex(clusters_S[0]) and simplex(clusters_T[0])
        assert len(clusters_S) == len(clusters_T)
        align_loss_multires, cluster_losses_multires = [], []
        p_jointS_list, p_jointT_list = [], []

        for rs in range(self._config['DA']['multi_scale']):
            if rs:
                clusters_S, clusters_T = multi_resilution_cluster(clusters_S, clusters_T)

            align_losses, cluster_losses, p_joint_Ss, p_joint_Ts = \
                zip(*[single_head_loss(clusters, clustert, displacement_maps=self.displacement_map_list) for
                      clusters, clustert in zip(clusters_S, clusters_T)])
            align_loss = sum(align_losses) / len(align_losses)
            cluster_loss = sum(cluster_losses) / len(cluster_losses)
            align_loss_multires.append(align_loss)
            cluster_losses_multires.append(cluster_loss)
            p_jointS_list.append(p_joint_Ss[-1])
            p_jointT_list.append(p_joint_Ts[-1])

        align_loss = average_list(align_loss_multires)
        cluster_loss = average_list(cluster_losses_multires)

        # for visualization
        p_joint_S = sum(p_jointS_list) / len(p_jointS_list)
        p_joint_T = sum(p_jointT_list) / len(p_jointT_list)
        clusters = clusters_S[-1]
        clustert = clusters_T[-1]

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

        return s_loss, cluster_loss, align_loss
