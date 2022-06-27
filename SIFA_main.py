import torch
from torch.optim import Adam

from arch.disc import OfficialDiscriminator
from arch.unet import UNet, decoderU
from configure import ConfigManager
from dataset.prostate import ProstateInterface, PromiseInterface
from dataset.mmwhs import mmWHSMRInterface, mmWHSCTInterface
from scheduler.warmup_scheduler import GradualWarmupScheduler
from trainers.SIFA_trainer import SIFA_trainer
from utils.radam import RAdam
from utils.utils import fix_all_seed_within_context, fix_all_seed

cmanager = ConfigManager("configs/sifa_config.yaml", strict=True)
config = cmanager.config
fix_all_seed(config['seed'])

with fix_all_seed_within_context(config['seed']):
    Generator = UNet(input_dim=1, num_classes=1)
    discriminator_t = OfficialDiscriminator(nc=1, ndf=64)
    optimizer_G = RAdam(Generator.parameters(), lr=config["Optim"]["lr"])
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=max(90, 1), eta_min=1e-7)
    scheduler_G = GradualWarmupScheduler(optimizer_G, multiplier=300, total_epoch=10, after_scheduler=scheduler_G)
    optimizer_t = Adam(discriminator_t.parameters(), lr=config["Optim"]["disc_lr"], betas=(0.5, 0.999))
    scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_t, T_max=max(90, 1), eta_min=1e-7)
    scheduler_t = GradualWarmupScheduler(optimizer_t, multiplier=300, total_epoch=10, after_scheduler=scheduler_t)

    model = UNet(num_classes=config['Data_input']['num_class'], input_dim=1)
    optimizer = RAdam(model.parameters(), lr=config["Optim"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(90, 1), eta_min=1e-7)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=300, total_epoch=10, after_scheduler=scheduler)

    decoder = decoderU(input_dim=256, out_dim=1)
    optimizer_U = RAdam(decoder.parameters(), lr=config["Optim"]["lr"])
    scheduler_U = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_U, T_max=max(90, 1), eta_min=1e-7)
    scheduler_U = GradualWarmupScheduler(optimizer_U, multiplier=300, total_epoch=10, after_scheduler=scheduler_U)

    discriminator_s = OfficialDiscriminator(nc=1, ndf=64)
    optimizer_s = Adam(discriminator_s.parameters(), lr=config["Optim"]["disc_lr"], betas=(0.5, 0.999))
    scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=max(90, 1), eta_min=1e-7)
    scheduler_s = GradualWarmupScheduler(optimizer_s, multiplier=300, total_epoch=10, after_scheduler=scheduler_s)

    discriminator_p1 = OfficialDiscriminator(nc=config['Data_input']['num_class'], ndf=64)
    optimizer_p1 = Adam(discriminator_p1.parameters(), lr=config["Optim"]["disc_lr"], betas=(0.5, 0.999))
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=max(90, 1), eta_min=1e-7)
    scheduler_p1 = GradualWarmupScheduler(optimizer_p1, multiplier=300, total_epoch=10, after_scheduler=scheduler_p1)

    discriminator_p2 = OfficialDiscriminator(nc=config['Data_input']['num_class'], ndf=64)
    optimizer_p2 = Adam(discriminator_p2.parameters(), lr=config["Optim"]["disc_lr"], betas=(0.5, 0.999))
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=max(90, 1), eta_min=1e-7)
    scheduler_p2 = GradualWarmupScheduler(optimizer_p2, multiplier=300, total_epoch=10, after_scheduler=scheduler_p2)

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

trainer = SIFA_trainer(
    Generator=Generator,
    discriminator_t=discriminator_t,
    model=model,
    decoder=decoder,
    discriminator_s=discriminator_s,
    discriminator_p1=discriminator_p1,
    discriminator_p2=discriminator_p2,
    optimizer_G=optimizer_G,
    optimizer_t=optimizer_t,
    optimizer=optimizer,
    optimizer_U=optimizer_U,
    optimizer_s=optimizer_s,
    optimizer_p1=optimizer_p1,
    optimizer_p2=optimizer_p2,
    scheduler_G=scheduler_G,
    scheduler_t=scheduler_t,
    scheduler=scheduler,
    scheduler_U=scheduler_U,
    scheduler_s=scheduler_s,
    scheduler_p1=scheduler_p1,
    scheduler_p2=scheduler_p2,
    TrainS_loader=trainS_loader,
    TrainT_loader=trainT_loader,
    valT_loader=valT_loader,
    test_loader=test_loader,
    config=config,
    **config['Trainer']
)
trainer.start_training()
