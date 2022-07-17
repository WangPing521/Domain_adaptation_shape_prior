from pathlib import Path
from typing import Union, Dict, Any, Tuple
import rising.random as rr
import rising.transforms as rt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.nn import functional as F

from loss.diceloss import DiceLoss
from loss.entropy import SimplexCrossEntropyLoss, Entropy, KL_div
from meters import Storage, MeterInterface, AverageValueMeter, UniversalDice
from meters.SummaryWriter import SummaryWriter
from utils import tqdm
from utils.general import path2Path, class2one_hot
from utils.rising import RisingWrapper
from utils.utils import set_environment, write_yaml

def meters_registerMTUDA(c):
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
            "sup_loss", AverageValueMeter()
        )
        meters.register_meter(
            "consistency_loss", AverageValueMeter()
        )
        meters.register_meter(
            "lkd_loss", AverageValueMeter()
        )

    with meters.focus_on("val"):
        meters.register_meter(
            f"test_dice", UniversalDice(C=c, report_axis=report_axis)
        )
        meters.register_meter(
            f"val_dice", UniversalDice(C=c, report_axis=report_axis)
        )

    return meters


class MTUDA_trainer:
    PROJECT_PATH = str(Path(__file__).parents[1])

    RUN_PATH = str(Path(PROJECT_PATH) / "runs")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")
    wholemeter_filename = "wholeMeter.csv"
    checkpoint_identifier = "last.pth"

    def __init__(
            self,
            model: nn.Module,
            source_ema_model: nn.Module,
            target_ema_model: nn.Module,
            optimizer,
            scheduler,
            TrainS_loader: Union[DataLoader, _BaseDataLoaderIter],
            TrainT_loader: Union[DataLoader, _BaseDataLoaderIter],
            test_loader: Union[DataLoader, _BaseDataLoaderIter],
            max_epoch: int = 100,
            save_dir: str = "base",
            checkpoint_path: str = None,
            device='cpu',
            config: dict = None,
            num_batches=200,
            *args,
            **kwargs
    ) -> None:
        self._save_dir: Path = Path(self.RUN_PATH) / str(save_dir)
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._start_epoch = 0
        if config:
            self._config = config.copy()
            self._config.pop("Config", None)
            write_yaml(self._config, save_dir=self._save_dir, save_name="config.yaml")
            set_environment(config.get("Environment"))

        self.model = model
        self.source_ema_model = source_ema_model
        self.target_ema_model = target_ema_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._trainS_loader = TrainS_loader
        self._trainT_loader = TrainT_loader
        self._test_loader = test_loader
        self._max_epoch = max_epoch
        self._num_batches = num_batches
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.crossentropy = SimplexCrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.entropy = Entropy(reduction='none')
        self.kl = KL_div()
        self._storage = Storage(self._save_dir)
        self.writer = SummaryWriter(str(self._save_dir))

        c = self._config['Data_input']['num_class']
        self.meters = meters_registerMTUDA(c)

        self.lkd_weight = self._config['weights']['lkd_weight']
        self.consistency = self._config['weights']['consistency']

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
        self.source_ema_model.to(device=device)
        self.target_ema_model.to(device=device)

    def run_step(self, s_data, t_data, cur_batch: int):
        S_img, S_target, S_filename = (
            s_data[0][0][0].to(self.device),
            s_data[0][0][1].to(self.device),
            s_data[0][1],
        )

        T_img, T_target, T_filename = (
            t_data[0][0][0].to(self.device),
            t_data[0][0][1].to(self.device),
            t_data[0][1],
        )

        T2S_img, T2S_target, T2S_filename = (
            t_data[1][0][0].to(self.device),
            t_data[1][0][1].to(self.device),
            t_data[1][1],
        )
        S2T2S_img, S2T2S_target, S2T2S_filename = (
            s_data[2][0][0].to(self.device),
            s_data[2][0][1].to(self.device),
            s_data[2][1],
        )

        S2T_img, S2T_target, S2T_filename = (
            s_data[1][0][0].to(self.device),
            s_data[1][0][1].to(self.device),
            s_data[1][1],
        )
        T2S2T_img, T2S2T_target, T2S2T_filename = (
            t_data[2][0][0].to(self.device),
            t_data[2][0][1].to(self.device),
            t_data[2][1],
        )
        S_target = self._rising_augmentation(S_target.float(), mode="feature", seed=cur_batch)
        T_target = self._rising_augmentation(T_target.float(), mode="feature", seed=cur_batch)

        S_img = self._rising_augmentation(S_img, mode="image", seed=cur_batch)
        T_img = self._rising_augmentation(T_img, mode="image", seed=cur_batch)
        T2S_img = self._rising_augmentation(T2S_img, mode="image", seed=cur_batch)
        S2T2S_img = self._rising_augmentation(S2T2S_img, mode="image", seed=cur_batch)
        S2T_img = self._rising_augmentation(S2T_img, mode="image", seed=cur_batch)
        T2S2T_img = self._rising_augmentation(T2S2T_img, mode="image", seed=cur_batch)

        pred_s_0 = self.model(S_img).softmax(1)
        pred_t2s_0 = self.model(T2S_img).softmax(1)
        pred_s2t2s_1 = self.model(S2T2S_img).softmax(1)

        # model loss
        onehot_targetS = class2one_hot(S_target.squeeze(1), self._config['Data_input']['num_class'])
        sup_loss = 0.5 * (self.crossentropy(pred_s_0, onehot_targetS) + self.dice_loss(pred_s_0, onehot_targetS))

        with torch.no_grad():
            pred_s_ema = self.source_ema_model(S_img).softmax(1)
            pred_t2s_ema = self.source_ema_model(T2S_img).softmax(1)
        # semantic
        # lkd_loss = self.kl(pred_s_0, pred_s_ema) + self.kl(pred_t2s_0, pred_t2s_ema)
        lkd_loss = F.mse_loss(pred_s_0, pred_s_ema) + F.mse_loss(pred_t2s_0, pred_t2s_ema)
        with torch.no_grad():
            pred_t_ema = self.target_ema_model(T_img).softmax(1)
            pred_s2t_ema = self.target_ema_model(S2T_img).softmax(1)
            pred_t2s2t_ema = self.target_ema_model(T2S2T_img).softmax(1)
        #structural
        consistency = F.mse_loss(self.entropy(pred_s2t_ema), self.entropy(pred_s_0)) + \
                         F.mse_loss(self.entropy(pred_s2t_ema), self.entropy(pred_s2t2s_1)) + \
                         F.mse_loss(self.entropy(pred_t_ema), self.entropy(pred_t2s_0)) + \
                         F.mse_loss(self.entropy(pred_t2s2t_ema), self.entropy(pred_t2s_0))

        self.meters[f"trainT_dice"].add(
            pred_s_0.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )

        return sup_loss, lkd_loss, consistency

    def train_loop(
            self,
            trainS_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            trainT_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            *args,
            **kwargs,
    ):
        self.model.train()
        self.source_ema_model.train()
        self.target_ema_model.train()

        s_co = min(1 - 1 / (epoch + 1), 0.999)

        batch_indicator = tqdm(range(self._num_batches))
        batch_indicator.set_description(f"Training Epoch {epoch:03d}")

        for cur_batch, (batch_id, s_data, t_data) in enumerate(zip(batch_indicator, trainS_loader, trainT_loader)):

            self.optimizer.zero_grad()
            sup_loss, lkd_loss, consistency_loss = self.run_step(s_data=s_data, t_data=t_data, cur_batch=cur_batch)

            loss = sup_loss + self.lkd_weight * lkd_loss + self.consistency * consistency_loss
            loss.backward()
            self.optimizer.step()

            for ema_param_s, param in zip(self.source_ema_model.parameters(), self.model.parameters()):
                ema_param_s.data.mul_(s_co).add_(1 - s_co, param.data)

            for ema_param_t, param in zip(self.target_ema_model.parameters(), self.model.parameters()):
                ema_param_t.data.mul_(s_co).add_(1 - s_co, param.data)

            self.meters['loss'].add(loss.item())
            self.meters['sup_loss'].add(sup_loss.item())
            self.meters['consistency_loss'].add(consistency_loss.item())
            self.meters['lkd_loss'].add(lkd_loss.item())

            report_dict = self.meters.statistics()
            batch_indicator.set_postfix_statics(report_dict)
        batch_indicator.close()

        report_dict = self.meters.statistics()
        assert report_dict is not None
        return dict(report_dict)

    def eval_loop(
            self,
            test_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            *args,
            **kwargs,
    ) -> Tuple[Any, Any]:
        self.model.eval()
        test_indicator = tqdm(test_loader)
        test_indicator.set_description(f"test_Epoch {epoch:03d}")

        for batch_id_test, data_test in enumerate(test_indicator):
            image_test, target_test, filename_test = (
                data_test[0][0].to(self.device),
                data_test[0][1].to(self.device),
                data_test[1]
            )
            preds_test = self.model(image_test).softmax(1)
            self.meters[f"test_dice"].add(
                preds_test.max(1)[1],
                target_test.squeeze(1),
                group_name=["_".join(x.split("_")[:-1]) for x in filename_test])

            report_dict = self.meters.statistics()
            test_indicator.set_postfix_statics(report_dict, cache_time=20)
        test_indicator.close()

        report_dict = self.meters.statistics()
        assert report_dict is not None
        return dict(report_dict), self.meters["test_dice"].summary()["DSC_mean"]

    def schedulerStep(self):
        self.scheduler.step()

    def start_training(self):
        self.to(self.device)
        self.cur_epoch = 0

        for self.cur_epoch in range(self._start_epoch, self._max_epoch):
            self.meters.reset()
            with self.meters.focus_on("train"):
                self.meters['lr'].add(self.optimizer.param_groups.__getitem__(0).get('lr'))
                train_metrics = self.train_loop(
                    trainS_loader=self._trainS_loader,
                    trainT_loader=self._trainT_loader,
                    epoch=self.cur_epoch
                )

            with self.meters.focus_on("val"), torch.no_grad():
                val_metric, _ = self.eval_loop(self._test_loader, self.cur_epoch)

            with self._storage:
                self._storage.add_from_meter_interface(tra=train_metrics, val=val_metric, epoch=self.cur_epoch)
                self.writer.add_scalars_from_meter_interface(tra=train_metrics, val=val_metric, epoch=self.cur_epoch)

            self.schedulerStep()
            self.save_checkpoint(self.state_dict(), self.cur_epoch)

    def inference(self, identifier="best.pth", *args, **kwargs):
        """
        Inference using the checkpoint, to be override by subclasses.
        :param args:
        :param kwargs:
        :return:
        """
        if self.checkpoint_path is None:
            self.checkpoint_path = self._save_dir
        assert Path(self.checkpoint_path).exists(), Path(self.checkpoint_path)
        assert (Path(self.checkpoint_path).is_dir() and identifier is not None) or (
                Path(self.checkpoint_path).is_file() and identifier is None
        )

        state_dict = torch.load(
            str(Path(self.checkpoint_path) / identifier)
            if identifier is not None
            else self.checkpoint_path,
            map_location=torch.device("cpu"),
        )
        self.load_checkpoint(state_dict)
        self.model.to(self.device)
        # to be added
        # probably call self._eval() method.

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
        # save_best: bool = True if float(cur_score) > float(self._best_score) else False
        # if save_best:
        #     self._best_score = float(cur_score)
        state_dict["epoch"] = current_epoch
        # state_dict["best_score"] = float(self._best_score)
        save_dir = self._save_dir if save_dir is None else path2Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_name is None:
            # regular saving
            torch.save(state_dict, str(save_dir / "last.pth"))
            # if save_best:
            #     torch.save(state_dict, str(save_dir / "best.pth"))
        else:
            # periodic saving
            torch.save(state_dict, str(save_dir / save_name))

    def _load_state_dict(self, state_dict) -> None:
        """
        Load state_dict for submodules having "load_state_dict" method.
        :param state_dict:
        :return:
        """
        for module_name, module in self.__dict__.items():
            if hasattr(module, "load_state_dict"):
                try:
                    module.load_state_dict(state_dict[module_name])
                except KeyError as e:
                    print(f"Loading checkpoint error for {module_name}, {e}.")
                except RuntimeError as e:
                    print(f"Interface changed error for {module_name}, {e}")

    def load_checkpoint(self, state_dict) -> None:
        """
        load checkpoint to models, meters, best score and _start_epoch
        Can be extended by add more state_dict
        :param state_dict:
        :return:
        """
        self._load_state_dict(state_dict)
        # self._best_score = state_dict["best_score"]
        self._start_epoch = state_dict["epoch"] + 1

    def load_checkpoint_from_path(self, checkpoint_path):
        checkpoint_path = path2Path(checkpoint_path)
        assert checkpoint_path.exists(), checkpoint_path
        if checkpoint_path.is_dir():
            state_dict = torch.load(
                str(Path(checkpoint_path) / self.checkpoint_identifier),
                map_location=torch.device("cpu"),
            )
        else:
            assert checkpoint_path.suffix == ".pth", checkpoint_path
            state_dict = torch.load(
                str(checkpoint_path), map_location=torch.device("cpu"),
            )
        self.load_checkpoint(state_dict)

    def clean_up(self, wait_time=3):
        """
        Do not touch
        :return:
        """
        import shutil
        import time

        time.sleep(wait_time)  # to prevent that the call_draw function is not ended.
        Path(self.ARCHIVE_PATH).mkdir(exist_ok=True, parents=True)
        sub_dir = self._save_dir.relative_to(Path(self.RUN_PATH))
        save_dir = Path(self.ARCHIVE_PATH) / str(sub_dir)
        if Path(save_dir).exists():
            shutil.rmtree(save_dir, ignore_errors=True)
        shutil.move(str(self._save_dir), str(save_dir))
        shutil.rmtree(str(self._save_dir), ignore_errors=True)

class MTUDA_prostate_trainer(MTUDA_trainer):
    def __init__(self, model: nn.Module,
                 source_ema_model: nn.Module, target_ema_model: nn.Module, optimizer, scheduler,
                 TrainS_loader: Union[DataLoader, _BaseDataLoaderIter],
                 TrainT_loader: Union[DataLoader, _BaseDataLoaderIter],
                 val_loader: Union[DataLoader, _BaseDataLoaderIter],
                 test_loader: Union[DataLoader, _BaseDataLoaderIter], *args, **kwargs) -> None:
        super().__init__(model, source_ema_model, target_ema_model, optimizer, scheduler, TrainS_loader, TrainT_loader,
                         test_loader, *args, **kwargs)
        self._val_loader = val_loader

    def eval_loop(
            self,
            val_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            test_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            *args,
            **kwargs,
    ) -> Tuple[Any, Any]:
        self.model.eval()
        val_indicator = tqdm(val_loader)
        val_indicator.set_description(f"test_Epoch {epoch:03d}")

        test_indicator = tqdm(test_loader)
        test_indicator.set_description(f"test_Epoch {epoch:03d}")
        report_dict = {}

        for batch_id_val, data_val in enumerate(val_indicator):
            image_val, target_val, filename_val = (
                data_val[0][0].to(self.device),
                data_val[0][1].to(self.device),
                data_val[1]
            )
            preds_val = self.model(image_val).softmax(1)
            self.meters[f"val_dice"].add(
                preds_val.max(1)[1],
                target_val.squeeze(1),
                group_name=["_".join(x.split("_")[:-1]) for x in filename_val])

            report_dict = self.meters.statistics()
            val_indicator.set_postfix_statics(report_dict, cache_time=20)
        val_indicator.close()

        for batch_id_test, data_test in enumerate(test_indicator):
            image_test, target_test, filename_test = (
                data_test[0][0].to(self.device),
                data_test[0][1].to(self.device),
                data_test[1]
            )
            preds_test = self.model(image_test).softmax(1)
            self.meters[f"test_dice"].add(
                preds_test.max(1)[1],
                target_test.squeeze(1),
                group_name=["_".join(x.split("_")[:-1]) for x in filename_test])

            report_dict = self.meters.statistics()
            test_indicator.set_postfix_statics(report_dict, cache_time=20)
        test_indicator.close()

        assert report_dict is not None
        return dict(report_dict), self.meters["test_dice"].summary()["DSC_mean"]



    def start_training(self):
        self.to(self.device)
        self.cur_epoch = 0

        for self.cur_epoch in range(self._start_epoch, self._max_epoch):
            self.meters.reset()
            with self.meters.focus_on("train"):
                self.meters['lr'].add(self.optimizer.param_groups.__getitem__(0).get('lr'))
                train_metrics = self.train_loop(
                    trainS_loader=self._trainS_loader,
                    trainT_loader=self._trainT_loader,
                    epoch=self.cur_epoch
                )

            with self.meters.focus_on("val"), torch.no_grad():
                val_metric, _ = self.eval_loop(self._val_loader, self._test_loader, self.cur_epoch)

            with self._storage:
                self._storage.add_from_meter_interface(tra=train_metrics, val=val_metric, epoch=self.cur_epoch)
                self.writer.add_scalars_from_meter_interface(tra=train_metrics, val=val_metric, epoch=self.cur_epoch)

            self.schedulerStep()
            self.save_checkpoint(self.state_dict(), self.cur_epoch)
