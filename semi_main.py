import torch

from arch.DomainSpecificBNUnet import convert2TwinBN, switch_bn as _switch_bn
from arch.unet import UNet
from configure import ConfigManager
from dataset.mmwhs import mmWHSMRInterface, mmWHSCTInterface, mmwhsCTSemiInterface, mmwhsMRSemiInterface
from demo.criterions import nullcontext
from scheduler.customized_scheduler import RampScheduler
from scheduler.warmup_scheduler import GradualWarmupScheduler
from trainers.Domain_supervised_Trainer import DomainsupervisedTrainer
from trainers.SourceTrainer import SourcebaselineTrainer
from trainers.align_IBN_trainer import align_IBNtrainer
from trainers.entropy_DA_trainer import EntropyDA
from trainers.semi_alignsource_trainer import Semi_alignTrainer
from trainers.upper_supervised_Trainer import UpperbaselineTrainer
from utils.radam import RAdam
from utils.utils import fix_all_seed_within_context

torch.backends.cudnn.benchmark = True

cmanager = ConfigManager("configs/config.yaml", strict=True)
config = cmanager.config
switch_bn = _switch_bn if config['DA']['double_bn'] else nullcontext

with fix_all_seed_within_context(config['seed']):
    model = UNet(num_classes=config['Data_input']['num_class'], input_dim=1)
    optimizer = RAdam(model.parameters(), lr=config["Optim"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(90, 1), eta_min=1e-7)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=300, total_epoch=10, after_scheduler=scheduler)

if config['Data_input']['dataset'] == 'mmwhs':
    CT_handler = mmwhsCTSemiInterface(**config["Data"])
    MR_handler = mmwhsMRSemiInterface(**config["Data"])
else:
    raise NotImplementedError(config['Data_input']['dataset'])

CT_handler.compile_dataloader_params(**config["DataLoader"])
MR_handler.compile_dataloader_params(**config["DataLoader"])

with fix_all_seed_within_context(config['Data']['seed']):
    if config['DA']['source'] == 'CT':
        label_loader, unlab_loader, val_loader = CT_handler.SemiSupervisedDataLoaders(
            labeled_transform=None,
            unlabeled_transform=None,
            val_transform=None,
            group_val=True,
            use_infinite_sampler=True,
        )
    else:
        label_loader, unlab_loader, val_loader = MR_handler.SemiSupervisedDataLoaders(
            labeled_transform=None,
            unlabeled_transform=None,
            val_transform=None,
            group_val=True,
            use_infinite_sampler=True,
        )


RegScheduler = RampScheduler(**config['Scheduler']["RegScheduler"])
weight_cluster = RampScheduler(**config['Scheduler']["ClusterScheduler"])

trainer = Semi_alignTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    lab_loader=label_loader,
    unlab_loader=unlab_loader,
    val_loader=val_loader,
    weight_scheduler=RegScheduler,
    config=config,
    **config['Trainer']
)
trainer.start_training()
