import torch
from arch.unet import UNet
from configure import ConfigManager
from dataset.prostate import ProstateInterface, PromiseInterface
from dataset.mmwhs import mmWHSMRInterface, mmWHSCTInterface
from scheduler.warmup_scheduler import GradualWarmupScheduler
from trainers.MTUDA_trainer import MTUDA_trainer
from utils.radam import RAdam
from utils.utils import fix_all_seed_within_context, fix_all_seed

cmanager = ConfigManager("configs/MTUDA_config.yaml", strict=True)
config = cmanager.config
fix_all_seed(config['seed'])

model = UNet(input_dim=1, num_classes=config['Data_input']['num_class'])
optimizer = RAdam(model.parameters(), lr=config["Optim"]["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(90, 1), eta_min=1e-7)
scheduler = GradualWarmupScheduler(optimizer, multiplier=300, total_epoch=10, after_scheduler=scheduler)

source_ema_model = UNet(input_dim=1, num_classes=config['Data_input']['num_class'])
for param_s in source_ema_model.parameters():
    param_s.detach_()
target_ema_model = UNet(input_dim=1, num_classes=config['Data_input']['num_class'])
for param_t in target_ema_model.parameters():
    param_t.detach_()

if config['Data_input']['dataset'] == 'mmwhs':
    handler1 = mmWHSCTInterface(**config["Data"])
    handler2 = mmWHSMRInterface(**config["Data"])
elif config['Data_input']['dataset'] == 'prostate':
    handler1 = PromiseInterface(**config["Data"])
    handler2 = ProstateInterface(**config["Data"])
else:
    raise NotImplementedError(config['Data_input']['dataset'])

handler1.compile_dataloader_params(**config["DataLoader"])
handler2.compile_dataloader_params(**config["DataLoader"])

with fix_all_seed_within_context(config['Data']['seed']):
    if config['DA']['source'] in ['CT', 'promise'] and config['DA']['target'] in ['MRI','prostate']:
        trainS_loader, valS_loader = handler1.DataLoaders(
            train_transform=None,
            val_transform=None,
            group_val=False,
            use_infinite_sampler=True,
            batchsize_indicator=config['DA']['batchsize_indicator']
        )
        trainT_loader, valT_loader = handler2.DataLoaders(
            train_transform=None,
            val_transform=None,
            group_val=False,
            use_infinite_sampler=True,
            batchsize_indicator=config['DA']['batchsize_indicator']
        )
    elif config['DA']['source'] in ['MRI','prostate'] and config['DA']['target'] in ['CT', 'promise']:
        trainT_loader, valT_loader, test_loader = handler1.DataLoaders(
            train_transform=None,
            val_transform=None,
            group_val=False,
            use_infinite_sampler=True,
            batchsize_indicator=config['DA']['batchsize_indicator']
        )
        trainS_loader, valS_loader, testS_loader = handler2.DataLoaders(
            train_transform=None,
            val_transform=None,
            group_val=False,
            use_infinite_sampler=True,
            batchsize_indicator=config['DA']['batchsize_indicator']
        )

trainer = MTUDA_trainer(
    model=model,
    source_ema_model=source_ema_model,
    target_ema_model=target_ema_model,
    optimizer=optimizer,
    scheduler=scheduler,
    TrainS_loader=trainS_loader,
    TrainT_loader=trainT_loader,
    valT_loader=valT_loader,
    test_loader=test_loader,
    config=config,
    **config['Trainer']
)
trainer.start_training()
