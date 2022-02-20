from typing import Union

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from arch.projectors import DenseClusterHead
from arch.utils import FeatureExtractor
from loss.IIDSegmentations import single_head_loss
from scheduler.customized_scheduler import RampScheduler
from trainers.SourceTrainer import SourcebaselineTrainer
from utils.general import class2one_hot
from utils.image_save_utils import plot_joint_matrix, FeatureMapSaver, plot_seg


class Semi_alignTrainer(SourcebaselineTrainer):

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
        unlab_img, unlab_target, unlab_filename = (
            t_data[0][0].to(self.device),
            t_data[0][1].to(self.device),
            t_data[1],
        )
        S_img = self._rising_augmentation(S_img, mode="image", seed=cur_batch)
        S_target = self._rising_augmentation(S_target.float(), mode="feature", seed=cur_batch)

        pred_S = self.model(S_img).softmax(1)
        onehot_targetS = class2one_hot(S_target.squeeze(1), C)
        s_loss = self.crossentropy(pred_S, onehot_targetS)

        if extracted_layer == 'Deconv_1x1':
            pred_unlab = self.model(unlab_img).softmax(1)

        else:
            pred_unlab = self.model(unlab_img).softmax(1)
            with self.extractor.enable_register(True):
                self.extractor.clear()
                _ = self.model(unlab_img, until=extracted_layer)
                feature_unlab = next(self.extractor.features())

            # projector cluster --->joint
            clusters_unlab = self.projector(feature_unlab)

        self.meters[f"train_dice"].add(
            pred_S.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )
        cluster_loss, align_loss=0,0
        return s_loss, cluster_loss, align_loss
