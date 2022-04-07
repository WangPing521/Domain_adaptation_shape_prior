from collections import OrderedDict

import torch

from arch.DomainSpecificBNUnet import convert2TwinBN
from arch.unet import UNet
from configure import ConfigManager
from dataset.mmwhs import mmWHSMRInterface, mmWHSCTInterface
from scheduler.warmup_scheduler import GradualWarmupScheduler
from trainers.psuedo_lableingDA import Pseudo_labelingDATrainer
from utils.radam import RAdam
from utils.utils import fix_all_seed_within_context, fix_all_seed

cmanager = ConfigManager("configs/config.yaml", strict=True)
config = cmanager.config
fix_all_seed(config['seed'])

Smodel = UNet(num_classes=config['Data_input']['num_class'], input_dim=1)
Smodel = convert2TwinBN(Smodel)
Smodel = Smodel.eval()
weight = f'runs/psuedoDA/0405/last.pth'
new_state_dict = OrderedDict()
state_dict = torch.load(weight)

Smodel.load_state_dict(state_dict.get('model'))

with fix_all_seed_within_context(config['seed']):
    model = UNet(num_classes=config['Data_input']['num_class'], input_dim=1)
    optimizer = RAdam(model.parameters(), lr=config["Optim"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(90, 1), eta_min=1e-7)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=300, total_epoch=10, after_scheduler=scheduler)

if config['Data_input']['dataset'] == 'mmwhs':
    CT_handler = mmWHSCTInterface(**config["Data"])
    MR_handler = mmWHSMRInterface(**config["Data"])
else:
    raise NotImplementedError(config['Data_input']['dataset'])

CT_handler.compile_dataloader_params(**config["DataLoader"])
MR_handler.compile_dataloader_params(**config["DataLoader"])

with fix_all_seed_within_context(config['Data']['seed']):
    if config['DA']['source'] == 'CT' and config['DA']['target'] == 'MRI':
        trainT_loader, valT_loader = MR_handler.DataLoaders(
            train_transform=None,
            val_transform=None,
            group_val=False,
            use_infinite_sampler=True,
            batchsize_indicator=config['DA']['batchsize_indicator']
        )
    elif config['DA']['source'] == 'MRI' and config['DA']['target'] == 'CT':
        trainT_loader, valT_loader = CT_handler.DataLoaders(
            train_transform=None,
            val_transform=None,
            group_val=False,
            use_infinite_sampler=True,
            batchsize_indicator=config['DA']['batchsize_indicator']
        )

trainer = Pseudo_labelingDATrainer(
    Smodel=Smodel,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    TrainT_loader=trainT_loader,
    valT_loader=valT_loader,
    config=config,
    **config['Trainer']
)
trainer.start_training()
