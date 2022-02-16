import argparse
from functools import partial
from itertools import chain

import torch
from loguru import logger
from tqdm import tqdm


from arch.DomainSpecificBNUnet import convert2TwinBN, switch_bn as _switch_bn
from demo.criterions import nullcontext, cluster_alignment_criterion, imsat_loss
from demo.data import get_data
from demo.model import SimpleNet, Projector, switch_grad

from meters import MeterInterface, AverageValueMeter
from loss.KL_divergence import KL_div
from utils.general import class2one_hot
from utils.printable import item2str
from utils.utils import fix_all_seed_within_context

torch.use_deterministic_algorithms(True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cluster-weight", default=0.0, type=float, )
    parser.add_argument("--target-cluster-weight", default=0.0, type=float, )
    parser.add_argument("--alignment-weight", default=0.0, type=float, )
    parser.add_argument("--seed", default=10, type=int, )
    parser.add_argument("--double-bn", default=False, action="store_true")
    parser.add_argument("--log-save-file", required=True, type=str)
    return parser.parse_args()


args = get_args()
logger.info(vars(args))
num_classes = 10
device = "cpu"
seed = args.seed
max_epoch = 100
num_batches = 500
tqdm = partial(tqdm, leave=False)
switch_bn = _switch_bn if args.double_bn else nullcontext
logger.add(f"{args.log_save_file}_seed_{seed}.log", level="TRACE")

with fix_all_seed_within_context(seed):
    model = SimpleNet(num_classes=num_classes, input_dim=1)
with fix_all_seed_within_context(seed):
    if args.double_bn:
        model = convert2TwinBN(model)
    model = model.to(device)
with fix_all_seed_within_context(seed):
    projector = Projector(input_dim=96, output_dim=20, intermediate_dim=128).to(device)
    optimizer = torch.optim.Adam(chain(model.parameters(), projector.parameters()), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
with fix_all_seed_within_context(seed):
    stra_loader, stest_loader = get_data(mode="source")
    ttra_loader, ttest_loader = get_data(mode="target")


def val_epoch(*, model, loader, sup_criterion, ):
    meters = MeterInterface()
    meters.register_meter("sup_loss", AverageValueMeter())
    meters.register_meter("acc", AverageValueMeter())
    model.eval()
    with tqdm(range(len(loader))) as indicator, torch.no_grad():
        for cur_batch, (data, label) in zip(indicator, loader):
            data, label = data.to(device), label.to(device)
            logits, _ = model(data)
            loss = sup_criterion(logits.softmax(1), class2one_hot(label, C=num_classes))
            meters["sup_loss"].add(loss.item())
            meters["acc"].add(torch.eq(logits.argmax(1), label).float().mean(), n=len(label))
            display_dict = meters.statistics()
            indicator.set_postfix_str(item2str(display_dict))
    return dict(meters.statistics())


def train_epoch(*, model, projector, optimizer: torch.optim.Optimizer, sup_criterion, cluster_criterion,
                alignment_criterion, source_loader, target_loader, num_batches: int):
    model.train()
    meters = MeterInterface()
    meters.register_meter("s_sup_loss", AverageValueMeter())
    meters.register_meter("s_c_loss", AverageValueMeter())
    meters.register_meter("s_acc", AverageValueMeter())
    meters.register_meter("t_acc", AverageValueMeter())
    meters.register_meter("t_c_loss", AverageValueMeter())
    meters.register_meter("align_loss", AverageValueMeter())
    meters.register_meter("total_loss", AverageValueMeter())
    with tqdm(range(num_batches)) as indicator:
        for cur_batch, (source_data, source_label), (target_data, _target_label) in zip(indicator, source_loader,
                                                                                        target_loader):
            source_data, source_label = source_data.to(device), source_label.to(device)
            target_data, _target_label = target_data.to(device), _target_label.to(device)
            optimizer.zero_grad()
            # source part
            with switch_bn(model, 0):
                source_logits, source_features = model(source_data)
            source_loss = sup_criterion(source_logits.softmax(1), class2one_hot(source_label, C=num_classes))
            meters["s_sup_loss"].add(source_loss.item())

            source_clusters = projector(source_features)
            source_cluster_loss = cluster_criterion(source_clusters)
            meters["s_c_loss"].add(source_cluster_loss.item())
            meters["s_acc"].add(torch.eq(source_logits.argmax(1), source_label).float().mean(), n=len(source_label))

            # target part
            with switch_bn(model, 1):
                target_logits, target_features = model(target_data)
            meters["t_acc"].add(torch.eq(target_logits.argmax(1), _target_label).float().mean(), n=len(_target_label))
            with switch_grad(projector, enable=False):
                target_clusters = projector(target_features)
            target_cluster_loss = cluster_criterion(target_clusters)
            meters["t_c_loss"].add(target_cluster_loss.item())

            # alignment loss
            source_target_alignment_loss = alignment_criterion(source_clusters.detach(), target_clusters)
            meters["align_loss"].add(source_target_alignment_loss.item())

            loss = source_loss \
                   + args.source_cluster_weight * source_cluster_loss \
                   + args.target_cluster_weight * target_cluster_loss \
                   + args.alignment_weight * source_target_alignment_loss

            meters["total_loss"].add(loss.item())
            loss.backward()
            optimizer.step()
            if cur_batch % 40 == 0:
                display_dict = meters.statistics()
                indicator.set_postfix_str(item2str(display_dict))
    display_dict = meters.statistics()
    indicator.set_postfix_str(item2str(display_dict))
    return dict(meters.statistics())


with fix_all_seed_within_context(seed):
    for cur_epoch in range(max_epoch):
        tra_stats = train_epoch(model=model, projector=projector, optimizer=optimizer, sup_criterion=KL_div(),
                                cluster_criterion=lambda x: imsat_loss(x),
                                alignment_criterion=lambda x, y: cluster_alignment_criterion(x, y),
                                source_loader=stra_loader,
                                target_loader=ttra_loader, num_batches=300)
        logger.info(f"training epoch {cur_epoch}: {item2str(tra_stats)}")
        # with switch_bn(model, 0):
        #     val_stats = val_epoch(model=model, loader=stest_loader, sup_criterion=KL_div())
        # logger.info(f"val epoch {cur_epoch}  on source: {item2str(val_stats)}")
        with switch_bn(model, 1):
            val_stats = val_epoch(model=model, loader=ttest_loader, sup_criterion=KL_div())
        logger.info(f"val epoch {cur_epoch}  on target: {item2str(val_stats)}")
        scheduler.step(cur_epoch)









