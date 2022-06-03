from typing import Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from scheduler.customized_scheduler import RampScheduler
from trainers.SourceTrainer import SourcebaselineTrainer
from utils.general import class2one_hot



class UpperbaselineTrainer(SourcebaselineTrainer):

    def __init__(self, TrainS_loader: Union[DataLoader, _BaseDataLoaderIter],
                 TrainT_loader: Union[DataLoader, _BaseDataLoaderIter],
                 valT_loader: Union[DataLoader, _BaseDataLoaderIter], test_loader: Union[DataLoader, _BaseDataLoaderIter], weight_scheduler: RampScheduler,
                 weight_cluster: RampScheduler,
                 model: nn.Module,
                 optimizer, scheduler, *args, **kwargs) -> None:
        super().__init__(model, optimizer, scheduler, TrainS_loader, TrainT_loader, valT_loader, test_loader,
                         weight_scheduler, weight_cluster, *args, **kwargs)

    def run_step(self, s_data, t_data, cur_batch: int):
        C = int(self._config['Data_input']['num_class'])
        T_img, T_target, T_filename = (
            t_data[0][0].to(self.device),
            t_data[0][1].to(self.device),
            t_data[1],
        )
        T_img = self._rising_augmentation(T_img, mode="image", seed=cur_batch)
        T_target = self._rising_augmentation(T_target.float(), mode="feature", seed=cur_batch)

        pred_T = self.model(T_img).softmax(1)

        onehot_targetT = class2one_hot(T_target.squeeze(1), C)
        s_loss = self.crossentropy(pred_T, onehot_targetT)

        self.meters[f"train_dice"].add(
            pred_T.max(1)[1],
            T_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in T_filename],
        )

        align_loss = torch.tensor(0, dtype=torch.float, device=pred_T.device)
        cluster_loss = torch.tensor(0, dtype=torch.float, device=pred_T.device)
        return s_loss, cluster_loss, align_loss
