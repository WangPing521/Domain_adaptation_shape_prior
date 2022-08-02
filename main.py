import torch

from arch.DomainSpecificBNUnet import convert2TwinBN, switch_bn as _switch_bn
from configure import ConfigManager
from dataset.prostate import ProstateInterface, PromiseInterface
from dataset.mmwhs import mmWHSMRInterface, mmWHSCTInterface
from demo.criterions import nullcontext
from scheduler.customized_scheduler import RampScheduler
from scheduler.warmup_scheduler import GradualWarmupScheduler
from trainers.SourceTrainer import SourcebaselineTrainer
from trainers.align_IBN_trainer import align_IBNtrainer
from trainers.align_combinationlayer_trainer import mutli_aligntrainer
from trainers.ent_prior_trainer import entPlusPriorTrainer
from trainers.entropy_DA_trainer import EntropyDA
from trainers.upper_supervised_Trainer import UpperbaselineTrainer
from utils.radam import RAdam
from utils.utils import fix_all_seed_within_context, fix_all_seed

cmanager = ConfigManager("configs/config.yaml", strict=True)
config = cmanager.config
fix_all_seed(config['seed'])
not_norm = config['not_norm']
switch_bn = _switch_bn if config['DA']['double_bn'] else nullcontext

with fix_all_seed_within_context(config['seed']):
    if not_norm:
        from arch.unet import UNet
    else:
        from arch.myunet import UNet
    model = UNet(num_classes=config['Data_input']['num_class'], input_dim=1)
with fix_all_seed_within_context(config['seed']):
    if config['DA']['double_bn']:
        model = convert2TwinBN(model)
    optimizer = RAdam(model.parameters(), lr=config["Optim"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(90, 1), eta_min=1e-7)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=300, total_epoch=10, after_scheduler=scheduler)

if config['Data_input']['dataset'] == 'mmwhs':
    handler1 = mmWHSMRInterface(seed = config["Data"]["seed"])
    handler2 = mmWHSCTInterface(seed = config["Data"]["seed"], kfold=config["Data"]["kfold"])
elif config['Data_input']['dataset'] == 'prostate':
    handler1 = ProstateInterface(seed = config["Data"]["seed"])
    handler2 = PromiseInterface(seed = config["Data"]["seed"], kfold=config["Data"]["kfold"])
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
        batchsize_indicator=config['DA']['batchsize_indicator']
    )
    trainT_loader, valT_loader, test_loader = handler2.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=config['DA']['batchsize_indicator']
    )

RegScheduler = RampScheduler(**config['Scheduler']["RegScheduler"])
weight_cluster = RampScheduler(**config['Scheduler']["ClusterScheduler"])

Trainer_container = {
    "baseline": SourcebaselineTrainer,
    "upperbaseline": UpperbaselineTrainer,
    "entda": EntropyDA,
    "align_IndividualBN": align_IBNtrainer,
    "combinationlayer": mutli_aligntrainer,
    "priorbased": entPlusPriorTrainer
}
trainer_name = Trainer_container.get(config['Trainer'].get('name'))

trainer = trainer_name(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    TrainS_loader=trainS_loader,
    TrainT_loader=trainT_loader,
    valT_loader=valT_loader,
    test_loader=test_loader,
    weight_scheduler=RegScheduler,
    weight_cluster=weight_cluster,
    switch_bn=switch_bn,
    config=config,
    **config['Trainer']
)

# trainer.inference(identifier='last.pth')
trainer.start_training()
