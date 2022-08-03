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
from utils.general import class2one_hot, average_list, simplex
from utils.image_save_utils import plot_joint_matrix1, plot_feature, plot_seg
from utils.utils import fix_all_seed_within_context


class align_IBNtrainer(SourcebaselineTrainer):

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

        with fix_all_seed_within_context(self._config['seed']):
            self.projector = DenseClusterHead(
                input_dim=self.model.get_channel_dim(self._config['DA']['align_layer']['name']),
                num_clusters=self._config['DA']['align_layer']['clusters'], T=0.5)
        self.optimizer.add_param_group({'params': self.projector.parameters(),
                                        })

        self.extractor = FeatureExtractor(self.model, feature_names=self._config['DA']['align_layer']['name'])
        self.extractor.bind()
        self.cc_based = self._config['DA']['align_layer']['cc_based']
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

        with self.switch_bn(self.model, 0), self.extractor.enable_register(True):
            self.extractor.clear()
            pred_S = self.model(S_img).softmax(1)
            feature_S = next(self.extractor.features())

        onehot_targetS = class2one_hot(S_target.squeeze(1), C)
        s_loss = self.crossentropy(pred_S, onehot_targetS)

        if extracted_layer == 'Deconv_1x1':
            with self.switch_bn(self.model, 1), self.extractor.enable_register(True):
                pred_T = self.model(T_img).softmax(1)
            if self._config['DA']['statistic']:
                clusters_S = [onehot_targetS.float()]
            else:
                clusters_S = [pred_S]
            clusters_T = [pred_T]
            assert simplex(clusters_S[0]) and simplex(clusters_T[0])

            if cur_batch == 0:
                source_seg = plot_seg(S_filename[-1], pred_S.max(1)[1][-1])
                target_seg = plot_seg(T_filename[-1], pred_T.max(1)[1][-1])
                self.writer.add_figure(tag=f"train_source_seg", figure=source_seg, global_step=self.cur_epoch, close=True)
                self.writer.add_figure(tag=f"train_target_seg", figure=target_seg, global_step=self.cur_epoch, close=True)

        else:
            with self.switch_bn(self.model, 1), self.extractor.enable_register(True):
                self.extractor.clear()
                pred_T = self.model(T_img).softmax(1)
                feature_T = next(self.extractor.features())
            # projector cluster --->joint
            if self.cc_based:
                # cross_correlation
                feature_S = (feature_S - feature_S.mean(1).unsqueeze(1)) / feature_S.std([1,2,3]).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                feature_T = (feature_T - feature_T.mean(1).unsqueeze(1)) / feature_T.std([1,2,3]).unsqueeze(1).unsqueeze(1).unsqueeze(1)

                clusters_S = [feature_S]
                clusters_T = [feature_T]
                if cur_batch == 0 and extracted_layer == 'Up_conv2':
                    source_f1 = plot_feature(feature_S[-1][0])
                    source_f2 = plot_feature(feature_S[-1][1])
                    source_f3 = plot_feature(feature_S[-1][2])
                    source_f4 = plot_feature(feature_S[-1][3])
                    source_f5 = plot_feature(feature_S[-1][4])
                    source_f6 = plot_feature(feature_S[-1][5])
                    source_f7 = plot_feature(feature_S[-1][6])
                    source_f8 = plot_feature(feature_S[-1][7])
                    source_f9 = plot_feature(feature_S[-1][8])
                    source_f10 = plot_feature(feature_S[-1][9])
                    source_f11 = plot_feature(feature_S[-1][10])

                    target_f1 = plot_feature(feature_T[-1][0])
                    target_f2 = plot_feature(feature_T[-1][1])
                    target_f3 = plot_feature(feature_T[-1][2])
                    target_f4 = plot_feature(feature_T[-1][3])
                    target_f5 = plot_feature(feature_T[-1][4])
                    target_f6 = plot_feature(feature_T[-1][5])
                    target_f7 = plot_feature(feature_T[-1][6])
                    target_f8 = plot_feature(feature_T[-1][7])
                    target_f9 = plot_feature(feature_T[-1][8])
                    target_f10 = plot_feature(feature_T[-1][9])
                    target_f11 = plot_feature(feature_T[-1][10])

                    self.writer.add_figure(tag=f"train_source_feature1", figure=source_f1, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_source_feature2", figure=source_f2, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_source_feature3", figure=source_f3, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_source_feature4", figure=source_f4, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_source_feature5", figure=source_f5, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_source_feature6", figure=source_f6, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_source_feature7", figure=source_f7, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_source_feature8", figure=source_f8, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_source_feature9", figure=source_f9, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_source_feature10", figure=source_f10, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_source_feature11", figure=source_f11, global_step=self.cur_epoch, close=True)


                    self.writer.add_figure(tag=f"train_target_feature1", figure=target_f1, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_target_feature2", figure=target_f2, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_target_feature3", figure=target_f3, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_target_feature4", figure=target_f4, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_target_feature5", figure=target_f5, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_target_feature6", figure=target_f6, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_target_feature7", figure=target_f7, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_target_feature8", figure=target_f8, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_target_feature9", figure=target_f9, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_target_feature10", figure=target_f10, global_step=self.cur_epoch, close=True)
                    self.writer.add_figure(tag=f"train_target_feature11", figure=target_f11, global_step=self.cur_epoch, close=True)


            else:
                clusters_S = self.projector(feature_S)
                clusters_T = self.projector(feature_T)
                if cur_batch == 0:
                    # len(clusters_S): 3 projectors
                    _, d, _, _ = clusters_S[0].shape
                    source_cluster = plot_seg(S_filename[-1], clusters_S[0].max(1)[1][-1])
                    self.writer.add_figure(tag=f"source_clusters_{d}", figure=source_cluster, global_step=self.cur_epoch, close=True)
                    target_cluster = plot_seg(T_filename[-1], clusters_T[0].max(1)[1][-1])
                    self.writer.add_figure(tag=f"target_clusters_{d}", figure=target_cluster, global_step=self.cur_epoch, close=True)

        assert len(clusters_S) == len(clusters_T)

        align_loss_multires = []
        p_jointS_list, p_jointT_list = [], []

        for rs in range(self._config['DA']['multi_scale']):
            if rs:
                clusters_S, clusters_T = multi_resilution_cluster(clusters_S, clusters_T, cc_based=self.cc_based)
            align_losses, p_joint_Ss, p_joint_Ts = \
                zip(*[single_head_loss(clusters, clustert, displacement_maps=self.displacement_map_list, cc_based=self.cc_based, cur_batch=cur_batch, cur_epoch=self.cur_epoch, vis=self.writer) for
                      clusters, clustert in zip(clusters_S, clusters_T)])

            align_loss = sum(align_losses) / len(align_losses)
            align_loss_multires.append(align_loss)
            p_jointS_list.append(p_joint_Ss[-1])
            p_jointT_list.append(p_joint_Ts[-1])

        align_loss = average_list(align_loss_multires)
        entT_loss = self.ent_loss(pred_T) # entropy on target

        # for visualization
        p_joint_S = sum(p_jointS_list) / len(p_jointS_list)
        p_joint_T = sum(p_jointT_list) / len(p_jointT_list)
        joint_error = torch.abs(p_joint_S - p_joint_T)
        joint_error_percent = joint_error / p_joint_S

        self.meters[f"train_dice"].add(
            pred_S.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )

        if cur_batch == 0:
            # source_joint_fig = plot_joint_matrix(p_joint_S)
            # target_joint_fig = plot_joint_matrix(p_joint_T)
            # self.writer.add_figure(tag=f"source_joint", figure=source_joint_fig, global_step=self.cur_epoch,
            #                        close=True, )
            # self.writer.add_figure(tag=f"target_joint", figure=target_joint_fig, global_step=self.cur_epoch,
            #                        close=True, )
            joint_error_fig = plot_joint_matrix1(joint_error, indicator="error")
            joint_error_percent_fig = plot_joint_matrix1(joint_error_percent, indicator="percenterror")
            self.writer.add_figure(tag=f"error_joint", figure=joint_error_fig, global_step=self.cur_epoch,
                                   close=True, )
            self.writer.add_figure(tag=f"error_percent", figure=joint_error_percent_fig, global_step=self.cur_epoch,
                                   close=True, )

        return s_loss, entT_loss, align_loss
