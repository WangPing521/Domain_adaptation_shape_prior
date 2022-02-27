from typing import Union

import ot
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from meters import AverageValueMeter
from scheduler.customized_scheduler import RampScheduler
from trainers.SourceTrainer import SourcebaselineTrainer
from utils.general import class2one_hot
from .align_IBN_trainer import align_IBNtrainer
from .olva_helper import VAEUNet


class OLVATrainer(align_IBNtrainer, SourcebaselineTrainer):
    def __init__(self, TrainS_loader: Union[DataLoader, _BaseDataLoaderIter],
                 TrainT_loader: Union[DataLoader, _BaseDataLoaderIter],
                 valS_loader: Union[DataLoader, _BaseDataLoaderIter],
                 valT_loader: Union[DataLoader, _BaseDataLoaderIter], weight_scheduler: RampScheduler,
                 weight_cluster: RampScheduler, model: VAEUNet, optimizer, scheduler, config, *args,
                 **kwargs) -> None:
        assert config['DA']['align_layer']['name'] == "Deconv_1x1"
        super().__init__(TrainS_loader, TrainT_loader, valS_loader, valT_loader, weight_scheduler, weight_cluster,
                         model, optimizer, scheduler, config=config, *args,**kwargs)
        del self.projector
        self.optimizer.param_groups.pop()
        self.extractor.remove()
        del self.extractor
        with self.meters.focus_on("olva"):
            self.meters.register_meter("kl", AverageValueMeter())
            self.meters.register_meter("ot", AverageValueMeter())

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

        with self.switch_bn(self.model, 0):
            pred_S = self.model(S_img).softmax(1)
        source_latent_mean = self.model.latent_code_mean
        source_latent_logvar = self.model.latent_code_log_var
        source_latent_sampled = self.model.latent_code_sampled

        onehot_targetS = class2one_hot(S_target.squeeze(1), C)
        s_loss = self.crossentropy(pred_S, onehot_targetS)

        with self.switch_bn(self.model, 1):
            _ = self.model(T_img).softmax(1)
        target_latent_mean = self.model.latent_code_mean
        target_latent_logvar = self.model.latent_code_log_var
        target_latent_sampled = self.model.latent_code_sampled

        self.meters[f"train_dice"].add(
            pred_S.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )

        kl_loss = 0.5 * self.vae_kl_divergence(source_latent_mean, source_latent_logvar) \
                  + 0.5 * self.vae_kl_divergence(target_latent_mean, target_latent_logvar)

        ot_loss = self.ot_loss(source_latent_sampled, target_latent_sampled, alpha=10.0)

        with self.meters.focus_on("olva"):
            self.meters["kl"].add(kl_loss.item())
            self.meters["ot"].add(ot_loss.item())
        return s_loss, kl_loss, ot_loss

    def vae_kl_divergence(self, mean, log_var) -> Tensor:
        mean, log_var = [torch.flatten(x, start_dim=1) for x in (mean, log_var)]
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = torch.mean(-0.5 * torch.mean(1 + log_var - mean ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

    def ot_loss(self, source_sampled: Tensor, target_sampled: Tensor, *, alpha: float = 1.0):
        assert source_sampled.shape == target_sampled.shape
        B = source_sampled.shape[0]
        source_sampled, target_sampled = [torch.flatten(x, start_dim=1) for x in (source_sampled, target_sampled)]
        pairwise_distance = torch.cdist(source_sampled, target_sampled) * alpha
        assert pairwise_distance.shape == torch.Size([B, B])

        T = ot.emd(torch.ones(B) / B, torch.ones(B) / B, pairwise_distance)
        return (pairwise_distance * T).mean()
