import torch
from torch.optim import Adam
from arch.DomainSpecificBNUnet import convert2TwinBN, switch_bn as _switch_bn

from arch.disc import OfficialDiscriminator
from arch.pointNet import PointNetCls
from arch.unet import UNet
from configure import ConfigManager
from dataset.prostate import ProstateInterface, PromiseInterface
from dataset.mmwhs import mmWHSMRInterface, mmWHSCTInterface
from scheduler.warmup_scheduler import GradualWarmupScheduler
from trainers.pointcloudUDA_trainer import pointCloudUDA_trainer
from utils.radam import RAdam
from utils.utils import fix_all_seed_within_context, fix_all_seed
from demo.criterions import nullcontext

cmanager = ConfigManager("configs/pointcloud_config.yaml", strict=True)
config = cmanager.config
fix_all_seed(config['seed'])

switch_bn = _switch_bn if config['DA']['double_bn'] else nullcontext

with fix_all_seed_within_context(config['seed']):
    model = UNet(num_classes=config['Data_input']['num_class'], input_dim=1)
with fix_all_seed_within_context(config['seed']):
    if config['DA']['double_bn']:
        model = convert2TwinBN(model)
    optimizer = RAdam(model.parameters(), lr=config["Optim"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(90, 1), eta_min=1e-7)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=300, total_epoch=10, after_scheduler=scheduler)

with fix_all_seed_within_context(config['seed']):
    discriminator_1 = OfficialDiscriminator(nc=config['Data_input']['num_class'], ndf=64)
    optimizer_1 = Adam(discriminator_1.parameters(), lr=config["Optim"]["disc_lr"], betas=(0.5, 0.999))
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=max(90, 1), eta_min=1e-7)
    scheduler_1 = GradualWarmupScheduler(optimizer_1, multiplier=300, total_epoch=10, after_scheduler=scheduler_1)

    discriminator_2 = OfficialDiscriminator(nc=1, ndf=64)
    optimizer_2 = Adam(discriminator_2.parameters(), lr=config["Optim"]["disc_lr"], betas=(0.5, 0.999))
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=max(90, 1), eta_min=1e-7)
    scheduler_2 = GradualWarmupScheduler(optimizer_2, multiplier=300, total_epoch=10, after_scheduler=scheduler_2)

    discriminator_3 = PointNetCls()
    optimizer_3 = Adam(discriminator_3.parameters(), lr=config["Optim"]["disc_lr"], betas=(0.5, 0.999))
    scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_3, T_max=max(90, 1), eta_min=1e-7)
    scheduler_3 = GradualWarmupScheduler(optimizer_3, multiplier=300, total_epoch=10, after_scheduler=scheduler_3)

if config['Data_input']['dataset'] == 'mmwhs':
    handler1 = mmWHSMRInterface(seed = config["Data"]["seed"])
    handler2 = mmWHSCTInterface(seed = config["Data"]["seed"], kfold=config["Data"]["kfold"])
elif config['Data_input']['dataset'] == 'prostate':
    handler1 = ProstateInterface(seed = config["Data"]["seed"])
    handler2 = PromiseInterface(seed = config["Data"]["seed"])
else:
    raise NotImplementedError(config['Data_input']['dataset'])

handler1.compile_dataloader_params(**config["DataLoader"])
handler2.compile_dataloader_params(**config["DataLoader"])

with fix_all_seed_within_context(config['Data']['seed']):
    trainS_loader = handler1.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=config['DA']['batchsize_indicator'],
        constrastve = config['DA']['constrastve_sampler']

    )
    trainT_loader, valT_loader, test_loader = handler2.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=config['DA']['batchsize_indicator'],
        constrastve = config['DA']['constrastve_sampler']
    )

trainer = pointCloudUDA_trainer(
    model=model,
    discriminator_1=discriminator_1,
    discriminator_2=discriminator_2,
    discriminator_3=discriminator_3,
    optimizer=optimizer,
    optimizer_1=optimizer_1,
    optimizer_2=optimizer_2,
    optimizer_3=optimizer_3,
    scheduler=scheduler,
    scheduler_1=scheduler_1,
    scheduler_2=scheduler_2,
    scheduler_3=scheduler_3,
    TrainS_loader=trainS_loader,
    TrainT_loader=trainT_loader,
    valT_loader=valT_loader,
    test_loader=test_loader,
    switch_bn=switch_bn,
    config=config,
    **config['Trainer']
)
trainer.start_training()
