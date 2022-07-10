import torch
from arch.unet import UNet
from configure import ConfigManager
from dataset.mmwhs_fake import mmWHS_T2S2T_Interface, mmWHS_T2S_Interface, mmWHS_S2T2S_Interface, mmWHS_S2T_Interface, \
    mmWHS_T2S_test_Interface
from dataset.prostate import ProstateInterface
from dataset.mmwhs import mmWHSMRInterface
from dataset.prostate_fake import prostate_T2S2T_Interface, prostate_T2S_Interface, prostate_S2T2S_Interface, \
    prostate_S2T_Interface
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
    handler1 = mmWHSMRInterface(seed = config["Data"]["seed"])  # S

    handlerS2T = mmWHS_S2T_Interface(seed = config["Data"]["seed"])
    handlerT2S2T = mmWHS_T2S2T_Interface(seed = config["Data"]["seed"])

    handlerT2S = mmWHS_T2S_Interface(seed = config["Data"]["seed"])
    handlerS2T2S = mmWHS_S2T2S_Interface(seed = config["Data"]["seed"])

    handler_test = mmWHS_T2S_test_Interface(seed = config["Data"]["seed"]) # T2S_test

elif config['Data_input']['dataset'] == 'prostate':
    handler1 = ProstateInterface(seed = config["Data"]["seed"]) # S

    handlerS2T = prostate_S2T_Interface(**config["Data"])
    handlerT2S2T = prostate_T2S2T_Interface(**config["Data"])

    handlerT2S = prostate_T2S_Interface(**config["Data"])
    handlerS2T2S = prostate_S2T2S_Interface(**config["Data"])
    handler_test = mmWHS_T2S_test_Interface(**config["Data"]) # T2S_test

else:
    raise NotImplementedError(config['Data_input']['dataset'])

handler1.compile_dataloader_params(**config["DataLoader"])

handlerS2T.compile_dataloader_params(**config["DataLoader"])
handlerT2S2T.compile_dataloader_params(**config["DataLoader"])

handlerT2S.compile_dataloader_params(**config["DataLoader"])
handlerS2T2S.compile_dataloader_params(**config["DataLoader"])
handler_test.compile_dataloader_params(**config["DataLoader"])

with fix_all_seed_within_context(config['Data']['seed']):
    trainS_loader = handler1.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=config['DA']['batchsize_indicator']
    )
    trainS2T_loader  = handlerS2T.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=config['DA']['batchsize_indicator']
    )

    trainT2S2T_loader = handlerT2S2T.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=config['DA']['batchsize_indicator']
    )

    trainS2T2S_loader = handlerS2T2S.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=config['DA']['batchsize_indicator']
    )
    trainT2S_loader= handlerT2S.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=config['DA']['batchsize_indicator']
    )

    trainT2S_test_loader = handler_test.DataLoaders(
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
    TrainS2T_loader=trainS2T_loader,
    TrainS2T2S_loader=trainS2T2S_loader,
    TrainT2S_loader=trainT2S_loader,
    TrainT2S2T_loader=trainT2S2T_loader,
    test_loader=trainT2S_test_loader,
    config=config,
    **config['Trainer']
)
trainer.start_training()
