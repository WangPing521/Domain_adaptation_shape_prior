import torch

from arch.DomainSpecificBNUnet import convert2TwinBN, switch_bn as _switch_bn
from arch.unet import UNet
from dataset.mmwhs import mmWHSMRInterface, mmWHSCTInterface
from demo.criterions import nullcontext
from scheduler.customized_scheduler import RampScheduler
from scheduler.warmup_scheduler import GradualWarmupScheduler
from trainers.Domain_supervised_Trainer import DomainsupervisedTrainer
from trainers.SourceTrainer import SourcebaselineTrainer
from trainers.align_IBN_trainer import align_IBNtrainer
from utils.radam import RAdam
from utils.utils import ConfigManger, fix_all_seed_within_context

torch.backends.cudnn.benchmark = True

config = ConfigManger("configs/config.yaml").config
switch_bn = _switch_bn if config['DA']['double_bn'] else nullcontext

with fix_all_seed_within_context(config['seed']):
    model = UNet(num_classes=config['Data_input']['num_class'], input_dim=1)
with fix_all_seed_within_context(config['seed']):
    if config['DA']['double_bn']:
        model = convert2TwinBN(model)
    # optimizer = torch.optim.Adam(chain(model.parameters(), projector.parameters()), lr=5e-4, weight_decay=1e-5)
    optimizer = RAdam(model.parameters(), lr=config["Optim"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(90, 1), eta_min=1e-7)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=300, total_epoch=10, after_scheduler=scheduler)

# with fix_all_seed_within_context(config['seed']):
#     projector = Projector(input_dim=96, output_dim=20, intermediate_dim=128).to(device)
#     optimizer = torch.optim.Adam(chain(model.parameters(), projector.parameters()), lr=5e-4, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

if config['Data_input']['dataset'] == 'mmwhs':
    CT_handler = mmWHSCTInterface(**config["Data"])
    MR_handler = mmWHSMRInterface(**config["Data"])
else:
    raise NotImplementedError(config['Data_input']['dataset'])

CT_handler.compile_dataloader_params(**config["DataLoader"])
MR_handler.compile_dataloader_params(**config["DataLoader"])

with fix_all_seed_within_context(config['Data']['seed']):
    if config['DA']['source'] == 'CT' and config['DA']['target'] == 'MRI':
        trainS_loader, valS_loader = CT_handler.DataLoaders(
            train_transform=None,
            val_transform=None,
            group_val=True,
            use_infinite_sampler=True,
        )
        trainT_loader, valT_loader = MR_handler.DataLoaders(
            train_transform=None,
            val_transform=None,
            group_val=True,
            use_infinite_sampler=True,
        )
    elif config['DA']['source'] == 'MRI' and config['DA']['target'] == 'CT':
        trainT_loader, valT_loader = CT_handler.DataLoaders(
            train_transform=None,
            val_transform=None,
            group_val=True,
            use_infinite_sampler=True,
        )
        trainS_loader, valS_loader = MR_handler.DataLoaders(
            train_transform=None,
            val_transform=None,
            group_val=True,
            use_infinite_sampler=True,
        )

RegScheduler = RampScheduler(**config['Scheduler']["RegScheduler"])
weight_cluster = RampScheduler(**config['Scheduler']["ClusterScheduler"])

Trainer_container = {
    "baseline": SourcebaselineTrainer,
    "supervised": DomainsupervisedTrainer,
    "align_IndividualBN": align_IBNtrainer,
}
trainer_name = Trainer_container.get(config['Trainer'].get('name'))

trainer = trainer_name(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    TrainS_loader=trainS_loader,
    TrainT_loader=trainT_loader,
    valS_loader=valS_loader,
    valT_loader=valT_loader,
    weight_scheduler=RegScheduler,
    weight_cluster=weight_cluster,
    switch_bn=switch_bn,
    config=config,
    **config['Trainer']
)
trainer.start_training()
