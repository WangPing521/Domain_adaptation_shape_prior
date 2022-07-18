from typing import Union, Tuple, Any, Dict
from pathlib import Path

import torch
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from arch.pointNet import PointNet, getGreedyPerm
from arch.utils import FeatureExtractor
from loss.entropy import SimplexCrossEntropyLoss, Entropy, jaccard_loss, batch_NN_loss
from meters.SummaryWriter import SummaryWriter
from utils import tqdm
from utils.rising import RisingWrapper
from utils.utils import set_environment, write_yaml, fix_all_seed_within_context
from utils.general import class2one_hot, path2Path
from torch import nn
from meters import Storage, MeterInterface, AverageValueMeter, UniversalDice
import rising.random as rr
import rising.transforms as rt

def meters_registerpointcloud(c):
    meters = MeterInterface()
    report_axis = list(range(1, c))

    with meters.focus_on("train"):
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter(
            f"trainT_dice", UniversalDice(C=c, report_axis=report_axis))

        # loss
        meters.register_meter(
            "loss", AverageValueMeter()
        )
        meters.register_meter(
            "loss_adv", AverageValueMeter()
        )

        meters.register_meter(
            "loss_Dis1", AverageValueMeter()
        )
        meters.register_meter(
            "loss_Dis2", AverageValueMeter()
        )
        meters.register_meter(
            "loss_Dis3", AverageValueMeter()
        )

    with meters.focus_on("val"):
        meters.register_meter(
            f"valT_dice", UniversalDice(C=c, report_axis=report_axis)
        )
        meters.register_meter(
            f"test_dice", UniversalDice(C=c, report_axis=report_axis)
        )

    return meters


class pointCloudUDA_trainer:
    PROJECT_PATH = str(Path(__file__).parents[1])

    RUN_PATH = str(Path(PROJECT_PATH) / "runs")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")
    wholemeter_filename = "wholeMeter.csv"
    checkpoint_identifier = "last.pth"

    def __init__(self,
                 model,
                 discriminator_1,
                 discriminator_2,
                 discriminator_3,
                 optimizer,
                 optimizer_1,
                 optimizer_2,
                 optimizer_3,
                 scheduler,
                 scheduler_1,
                 scheduler_2,
                 scheduler_3,
                 TrainS_loader: Union[DataLoader, _BaseDataLoaderIter],
                 TrainT_loader: Union[DataLoader, _BaseDataLoaderIter],
                 valT_loader: Union[DataLoader, _BaseDataLoaderIter],
                 test_loader: Union[DataLoader, _BaseDataLoaderIter],
                 switch_bn,
                 max_epoch: int = 100,
                 save_dir: str = "base",
                 checkpoint_path: str = None,
                 device='cpu',
                 config: dict = None,
                 num_batches=200,
                *args, **kwargs) -> None:
        self._save_dir: Path = Path(self.RUN_PATH) / str(save_dir)
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._start_epoch = 0
        if config:
            self._config = config.copy()
            self._config.pop("Config", None)
            write_yaml(self._config, save_dir=self._save_dir, save_name="config.yaml")
            set_environment(config.get("Environment"))

        self.model = model
        self.discriminator_1 = discriminator_1  # output
        self.discriminator_2 = discriminator_2  # entropy
        self.discriminator_3 = discriminator_3  # point

        self.optimizer = optimizer
        self.optimizer_1 = optimizer_1
        self.optimizer_2 = optimizer_2
        self.optimizer_3 = optimizer_3

        with fix_all_seed_within_context(self._config['seed']):
            self.point_net = PointNet(num_points=300, ext=False)
        self.optimizer.add_param_group({'params': self.point_net.parameters()})

        self.scheduler = scheduler
        self.scheduler_1 = scheduler_1
        self.scheduler_2 = scheduler_2
        self.scheduler_3 = scheduler_3
        self.TrainS_loader = TrainS_loader
        self.TrainT_loader = TrainT_loader
        self.valT_loader = valT_loader
        self.test_loader = test_loader

        self._max_epoch = max_epoch
        self._num_batches = num_batches
        self.device = device
        self.switch_bn = switch_bn
        self.checkpoint_path = checkpoint_path
        self._storage = Storage(self._save_dir)
        self.writer = SummaryWriter(str(self._save_dir))

        c = self._config['Data_input']['num_class']
        self.meters = meters_registerpointcloud(c)

        self.disc_weight = self._config['weights']
        self.extractor = FeatureExtractor(self.model, feature_names='Conv5')
        self.extractor.bind()

        self.crossentropy = SimplexCrossEntropyLoss()
        self._bce_criterion = nn.BCELoss()
        self.entropy = Entropy(reduction='none')
        self.z = self.compute_z(TrainS_loader)

        geometric_transform = rt.Compose(
            rt.BaseAffine(
                scale=rr.UniformParameter(0.5, 1.5),
                rotation=rr.UniformParameter(-30, 30), degree=True,
                translation=rr.UniformParameter(-0.2, 0.2), grad=True,
                interpolation_mode="nearest"
            ),
            rt.Mirror(dims=[0, 1], p_sample=0.5, grad=True)
        )
        intensity_transform = rt.Compose(
            rt.GammaCorrection(gamma=rr.UniformParameter(0.8, 1.2), grad=True),
            rt.GaussianNoise(mean=0, std=0.01),
        )

        self._rising_augmentation = RisingWrapper(
            geometry_transform=geometric_transform, intensity_transform=intensity_transform
        )


    def to(self, device):
        self.model.to(device=device)
        self.discriminator_1.to(device=device)
        self.discriminator_2.to(device=device)
        self.discriminator_3.to(device=device)

    def run_step(self, s_data, t_data, cur_batch: int):
        C = int(self._config['Data_input']['num_class'])
        S_img, S_target, S_filename = (
            s_data[0][0].to(self.device),
            s_data[0][1].to(self.device),
            s_data[1],
        )
        T_img, T_target, T_filename = (
            t_data[0][0].to(self.device),
            t_data[0][1].to(self.device),
            t_data[1],
        )

        S_img = self._rising_augmentation(S_img, mode="image", seed=cur_batch)
        S_target = self._rising_augmentation(S_target.float(), mode="feature", seed=cur_batch)
        T_img = self._rising_augmentation(T_img, mode="image", seed=cur_batch)

        # estimate vertexS
        S_target = torch.where(S_target == 4, torch.Tensor([0]).to(self.device), S_target)
        S_target = torch.where(S_target > 0, torch.Tensor([1]).to(self.device), S_target)
        vertexA = torch.where(S_target > 0)
        vertexS_index = []
        for i in range(len(vertexA[0])):
            point = torch.cat([vertexA[0][i].unsqueeze(0), vertexA[1][i].unsqueeze(0), vertexA[2][i].unsqueeze(0), vertexA[3][i].unsqueeze(0)],0)
            vertexS_index.append(point)
        vertexS = torch.stack(vertexS_index)
        vertexStran = vertexS.transpose(1,0)
        vertexS_BSindex = []
        for img_bs in range(S_img.shape[0]):
            index_img = torch.where(vertexStran[0] == img_bs)[0]
            ToSamplePoints = vertexS[index_img,:]
            D = pairwise_distances(ToSamplePoints.squeeze(0).squeeze(0), metric='euclidean')
            (perm, lambdas) = getGreedyPerm(D)
            vertexS_BSindex.append(ToSamplePoints[perm,:].unsqueeze(0))
        vertexS = torch.stack(vertexS_BSindex, dim=0)
        vertexS = vertexS.squeeze(1)

        # add z
        # norm to [0-1]
        self.z



        onehot_targetS = class2one_hot(S_target.squeeze(1), C)
        source_domain_label = 1
        target_domain_label = 0

        # 1. train the generator (do not update the params in the discriminators)
        self.optimizer.zero_grad()
        with self.switch_bn(self.model, 0), self.extractor.enable_register(True):
            self.extractor.clear()
            pred_S = self.model(S_img).softmax(1)
            feature_S = next(self.extractor.features())
            point_S = self.point_net(feature_S)

        s_loss1 = self.crossentropy(pred_S, onehot_targetS)
        s_loss2 = jaccard_loss(logits=pred_S, label=S_target.float(), activation=False)
        s_loss3 = batch_NN_loss(x=point_S, y=vertexS.float())
        ent_mapS = self.entropy(pred_S) # entropy on source
        ent_lossS = ent_mapS.mean()
        s_loss = s_loss1 + s_loss2 + s_loss3 + ent_lossS
        s_loss.backward()

        # 2. train the segmentation model to fool the discriminators
        with self.switch_bn(self.model, 1), self.extractor.enable_register(True):
            self.extractor.clear()
            pred_T = self.model(T_img).softmax(1)
            feature_T = next(self.extractor.features())
            point_T = self.point_net(feature_T)

        ent_mapT = self.entropy(pred_T) # entropy on target

        out_disPred = self.discriminator_1(pred_T)
        out_disEnt = self.discriminator_2(ent_mapT)
        out_disPoint = self.discriminator_3(point_T)
        loss_adv1 = self._bce_criterion(out_disPred, torch.zeros(out_disPred.shape[0], device=self.device).fill_(source_domain_label))
        loss_adv2 = self._bce_criterion(out_disEnt, torch.zeros(out_disEnt.shape[0], device=self.device).fill_(source_domain_label))
        loss_adv3 = self._bce_criterion(out_disPoint, torch.zeros(out_disPoint.shape[0], device=self.device).fill_(source_domain_label))

        loss_adv =  0.2 * loss_adv1 + 0.2 * loss_adv2 + 0.2 * loss_adv3
        loss_adv.backward()
        self.optimizer.step()

        # 3. train the discriminators with images from source domain
        self.optimizer_2.zero_grad()
        out_disEntS = self.discriminator_2(ent_mapS.detach())
        out_disEntT = self.discriminator_2(ent_mapT.detach())

        out_disEntSadv = self._bce_criterion(out_disEntS,  torch.zeros(out_disEntS.shape[0], device=self.device).fill_(source_domain_label))
        out_disEntTadv = self._bce_criterion(out_disEntT,  torch.zeros(out_disEntT.shape[0], device=self.device).fill_(target_domain_label))
        loss_disadv2 = out_disEntSadv + out_disEntTadv
        loss_disadv2.backward()
        self.optimizer_2.step()

        self.optimizer_1.zero_grad()
        out_disPredS = self.discriminator_1(pred_S.detach())
        out_disPredT = self.discriminator_1(pred_T.detach())

        out_disPredSadv = self._bce_criterion(out_disPredS, torch.zeros(out_disPredS.shape[0], device=self.device).fill_(
            source_domain_label))
        out_disPredTadv = self._bce_criterion(out_disPredT,
                                              torch.zeros(out_disPredT.shape[0], device=self.device).fill_(
                                                  target_domain_label))
        loss_disadv1 = out_disPredSadv + out_disPredTadv
        loss_disadv1.backward()
        self.optimizer_1.step()

        self.optimizer_3.zero_grad()
        out_disPointS = self.discriminator_3(point_S.detach())
        out_disPointT = self.discriminator_3(point_T.detach())
        out_disPointSadv = self._bce_criterion(out_disPointS,
                                              torch.zeros(out_disPointS.shape[0], device=self.device).fill_(
                                                  source_domain_label))
        out_disPointTadv = self._bce_criterion(out_disPointT,
                                               torch.zeros(out_disPointT.shape[0], device=self.device).fill_(
                                                   target_domain_label))
        loss_disadv3 = out_disPointSadv + out_disPointTadv

        loss_disadv3.backward()
        self.optimizer_3.step()

        self.meters[f"train_dice"].add(
            pred_S.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )
        return s_loss, loss_adv, loss_disadv1, loss_disadv2, loss_disadv3


    def train_loop(
            self,
            trainS_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            trainT_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            *args,
            **kwargs,
    ):
        self.model.train()
        self.discriminator_1.train()
        self.discriminator_2.train()
        self.discriminator_3.train()

        batch_indicator = tqdm(range(self._num_batches))
        batch_indicator.set_description(f"Training Epoch {epoch:03d}")

        for cur_batch, s_data, t_data in zip(batch_indicator, trainS_loader, trainT_loader):

            s_loss, loss_adv, loss_disadv1, loss_disadv2, loss_disadv3 = self.run_step(s_data=s_data, t_data=t_data, cur_batch=cur_batch)

            self.meters['loss'].add(s_loss.item())
            self.meters['loss_adv'].add(loss_adv.item())
            self.meters['loss_Dis1'].add(loss_disadv1.item())
            self.meters['loss_Dis2'].add(loss_disadv2.item())
            self.meters['loss_Dis3'].add(loss_disadv3.item())

            report_dict = self.meters.statistics()
            batch_indicator.set_postfix_statics(report_dict, cache_time=20)
        batch_indicator.close()
        report_dict = self.meters.statistics()
        assert report_dict is not None
        return dict(report_dict)

    def eval_loop(
            self,
            valT_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            test_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            *args,
            **kwargs,
    ) -> Tuple[Any, Any]:
        self.model.eval()
        valT_indicator = tqdm(valT_loader)
        valT_indicator.set_description(f"ValT_Epoch {epoch:03d}")
        test_indicator = tqdm(test_loader)
        test_indicator.set_description(f"test_Epoch {epoch:03d}")

        for batch_idT, data_T in enumerate(valT_indicator):
            imageT, targetT, filenameT = (
                data_T[0][0].to(self.device),
                data_T[0][1].to(self.device),
                data_T[1]
            )
            with self.switch_bn(self.model, 1):
                preds_T = self.model(imageT).softmax(1)
            self.meters[f"valT_dice"].add(
                preds_T.max(1)[1],
                targetT.squeeze(1),
                group_name=["_".join(x.split("_")[:-1]) for x in filenameT])

            report_dict = self.meters.statistics()
            valT_indicator.set_postfix_statics(report_dict, cache_time=20)

        valT_indicator.close()

        for batch_id_test, data_test in enumerate(test_indicator):
            image_test, target_test, filename_test = (
                data_test[0][0].to(self.device),
                data_test[0][1].to(self.device),
                data_test[1]
            )
            with self.switch_bn(self.model, 1):
                preds_test = self.model(image_test).softmax(1)
            self.meters[f"test_dice"].add(
                preds_test.max(1)[1],
                target_test.squeeze(1),
                group_name=["_".join(x.split("_")[:-1]) for x in filename_test])

        test_indicator.close()
        report_dict = self.meters.statistics()
        assert report_dict is not None
        return dict(report_dict), self.meters["valT_dice"].summary()["DSC_mean"]

    def schedulerStep(self):
        self.scheduler.step()
        self.scheduler_1.step()
        self.scheduler_2.step()
        self.scheduler_3.step()

    def start_training(self):
        self.to(self.device)
        self.cur_epoch = 0

        for self.cur_epoch in range(self._start_epoch, self._max_epoch):
            self.meters.reset()
            with self.meters.focus_on("train"):
                self.meters['lr'].add(self.optimizer.param_groups.__getitem__(0).get('lr'))
                train_metrics = self.train_loop(
                    trainS_loader=self.TrainS_loader,
                    trainT_loader=self.TrainT_loader,
                    epoch=self.cur_epoch
                )

            with self.meters.focus_on("val"), torch.no_grad():
                val_metric, _ = self.eval_loop(self.valT_loader, self.test_loader, self.cur_epoch)

            with self._storage:
                self._storage.add_from_meter_interface(tra=train_metrics, val=val_metric, epoch=self.cur_epoch)
                self.writer.add_scalars_from_meter_interface(tra=train_metrics, val=val_metric, epoch=self.cur_epoch)

            self.schedulerStep()
            self.save_checkpoint(self.state_dict(), self.cur_epoch)

    @torch.no_grad()
    def compute_z(self, TrainS_loader):
        patient_Dic = dict()
        for _, data in zip(range(len(TrainS_loader)), TrainS_loader):
            file_name = data[1]
            key_0 = file_name[7:13]
            if patient_Dic == None or patient_Dic[key_0] == None:
                patient_Dic[key_0:""]
        return patient_Dic

    def state_dict(self) -> Dict[str, Any]:
        """
        return trainer's state dict. The dict is built by considering all the submodules having `state_dict` method.
        """
        state_dictionary = {}
        for module_name, module in self.__dict__.items():
            if hasattr(module, "state_dict"):
                state_dictionary[module_name] = module.state_dict()
        return state_dictionary

    def save_checkpoint(
            self, state_dict, current_epoch, save_dir=None, save_name=None
    ):
        """
        save checkpoint with adding 'epoch' and 'best_score' attributes
        :param state_dict:
        :param current_epoch:
        :return:
        """
        state_dict["epoch"] = current_epoch
        save_dir = self._save_dir if save_dir is None else path2Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_name is None:
            # regular saving
            torch.save(state_dict, str(save_dir / "last.pth"))
        else:
            # periodic saving
            torch.save(state_dict, str(save_dir / save_name))


