import torch
from arch.unet import UNet
from configure import ConfigManager
from dataset.mmwhs_fake import mmWHS_T2S2T_Interface, mmWHS_T2S_Interface, mmWHS_S2T2S_Interface, mmWHS_S2T_Interface, \
    mmWHS_T2S_test_Interface, Source_like, Target_like
from dataset.prostate import ProstateInterface
from dataset.mmwhs import mmWHSMRInterface
from dataset.prostate_fake import prostate_T2S2T_Interface, prostate_T2S_Interface, prostate_S2T2S_Interface, \
    prostate_S2T_Interface, mmWHS_T2S_val_Interface
from scheduler.warmup_scheduler import GradualWarmupScheduler
from trainers.MTUDA_trainer import MTUDA_trainer, MTUDA_prostate_trainer
from utils.radam import RAdam
from utils.utils import fix_all_seed_within_context, fix_all_seed
from torch.utils.data import DataLoader

cmanager = ConfigManager("configs/MTUDA_config.yaml", strict=True)
config = cmanager.config
fix_all_seed(config['seed'])

model = UNet(input_dim=1, num_classes=config['Data_input']['num_class'])
optimizer = RAdam(model.parameters(), lr=config["Optim"]["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(90, 1), eta_min=1e-7)
scheduler = GradualWarmupScheduler(optimizer, multiplier=300, total_epoch=10, after_scheduler=scheduler)

source_ema_model = UNet(input_dim=1, num_classes=config['Data_input']['num_class'])
optimizer_emaS = RAdam(source_ema_model.parameters(), lr=config["Optim"]["lr"])
scheduler_emaS = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_emaS, T_max=max(90, 1), eta_min=1e-7)
scheduler_emaS = GradualWarmupScheduler(optimizer_emaS, multiplier=300, total_epoch=10, after_scheduler=scheduler_emaS)
for param_s in source_ema_model.parameters():
    param_s.detach_()
target_ema_model = UNet(input_dim=1, num_classes=config['Data_input']['num_class'])
optimizer_emaT = RAdam(target_ema_model.parameters(), lr=config["Optim"]["lr"])
scheduler_emaT = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_emaT, T_max=max(90, 1), eta_min=1e-7)
scheduler_emaT = GradualWarmupScheduler(optimizer_emaT, multiplier=300, total_epoch=10, after_scheduler=scheduler_emaT)
for param_t in target_ema_model.parameters():
    param_t.detach_()

if config['Data_input']['dataset'] == 'mmwhs':
    handler1 = mmWHSMRInterface(seed = config["Data"]["seed"])  # S

    handlerS2T = mmWHS_S2T_Interface(seed = config["Data"]["seed"])
    handlerT2S2T = mmWHS_T2S2T_Interface(seed = config["Data"]["seed"])

    handlerT2S = mmWHS_T2S_Interface(seed = config["Data"]["seed"])
    handlerS2T2S = mmWHS_S2T2S_Interface(seed = config["Data"]["seed"])

    handler_test = mmWHS_T2S_test_Interface(seed = config["Data"]["seed"]) # T2S_test

elif config['Data_input']['dataset'] == 'prostate':
    handler1 = ProstateInterface(seed = config["Data"]["seed"]) # S

    handlerS2T = prostate_S2T_Interface(seed = config["Data"]["seed"])
    handlerT2S2T = prostate_T2S2T_Interface(seed = config["Data"]["seed"])

    handlerT2S = prostate_T2S_Interface(seed = config["Data"]["seed"])
    handlerS2T2S = prostate_S2T2S_Interface(seed = config["Data"]["seed"])

    handler_val = mmWHS_T2S_val_Interface(seed = config["Data"]["seed"]) # T2S_val

    handler_test = mmWHS_T2S_test_Interface(seed = config["Data"]["seed"]) # T2S_test

else:
    raise NotImplementedError(config['Data_input']['dataset'])


S_dataset = handler1._create_datasets(train_transform=None, val_transform=None)
S2T_dataset = handlerS2T._create_datasets(train_transform=None, val_transform=None)
S2T2S_dataset = handlerS2T2S._create_datasets(train_transform=None, val_transform=None)

T2S_dataset = handlerT2S._create_datasets(train_transform=None, val_transform=None)
T2S2T_dataset = handlerT2S2T._create_datasets(train_transform=None, val_transform=None)

dataset_S = Source_like(S_dataset, S2T_dataset, S2T2S_dataset)
dataset_T = Target_like(T2S_dataset, T2S2T_dataset)

source_like_loader = DataLoader(dataset_S, batch_size=10, shuffle=True)
target_like_loader = DataLoader(dataset_T, batch_size=10, shuffle=True)

handler_test.compile_dataloader_params(**config["DataLoader"])
with fix_all_seed_within_context(config['Data']['seed']):
    trainT2S_test_loader = handler_test.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=config['DA']['batchsize_indicator'],
        constrastve=config['DA']['constrastve_sampler']
    )

if config['Data_input']['dataset'] == 'prostate':
    handler_val.compile_dataloader_params(**config["DataLoader"])
    with fix_all_seed_within_context(config['Data']['seed']):
        trainT2S_val_loader = handler_val.DataLoaders(
            train_transform=None,
            val_transform=None,
            group_val=False,
            use_infinite_sampler=True,
            batchsize_indicator=config['DA']['batchsize_indicator'],
            constrastve=config['DA']['constrastve_sampler']
        )

if config['Data_input']['dataset'] == 'prostate':
    trainer = MTUDA_prostate_trainer(
        model=model,
        source_ema_model=source_ema_model,
        target_ema_model=target_ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_emaS=scheduler_emaS,
        scheduler_emaT=scheduler_emaT,
        TrainS_loader=source_like_loader,
        TrainT_loader=target_like_loader,
        val_loader=trainT2S_val_loader,
        test_loader=trainT2S_test_loader,
        config=config,
        **config['Trainer']
    )
else:
    trainer = MTUDA_trainer(
        model=model,
        source_ema_model=source_ema_model,
        target_ema_model=target_ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_emaS=scheduler_emaS,
        scheduler_emaT=scheduler_emaT,
        TrainS_loader=source_like_loader,
        TrainT_loader= target_like_loader,
        test_loader=trainT2S_test_loader,
        config=config,
        **config['Trainer']
    )
trainer.start_training()

# import os
#
# file_fold = '.data/prostate_CYC/recover_promise_train/img'
# def modifyfilename(fileroot):
#     for file_png in os.listdir(fileroot):
#         os.rename(f'{fileroot}/{file_png}', f'{fileroot}/{file_png[:9]}{file_png[11:]}')
#
# modifyfilename(file_fold)