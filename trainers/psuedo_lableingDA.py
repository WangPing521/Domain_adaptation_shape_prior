from pathlib import Path
from typing import Union, Dict, Any, Tuple
from arch.DomainSpecificBNUnet import switch_bn
import rising.random as rr
import rising.transforms as rt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
import numpy as np
from loss.entropy import SimplexCrossEntropyLoss
from meters import Storage, MeterInterface, AverageValueMeter, UniversalDice
from meters.SummaryWriter import SummaryWriter
from utils import tqdm
from utils.general import path2Path
from utils.image_save_utils import plot_seg
from utils.rising import RisingWrapper
from utils.utils import set_environment, write_yaml

def meters_registerPDA(c):
    meters = MeterInterface()
    report_axis = list(range(1, c))

    with meters.focus_on("train"):
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter(
            f"trainT_dice", UniversalDice(C=c, report_axis=report_axis))

        # loss
        meters.register_meter(
            "s_loss", AverageValueMeter()
        )

    with meters.focus_on("val"):
        meters.register_meter(
            f"test_dice", UniversalDice(C=c, report_axis=report_axis)
        )
    return meters


class Pseudo_labelingDATrainer:
    PROJECT_PATH = str(Path(__file__).parents[1])

    RUN_PATH = str(Path(PROJECT_PATH) / "runs")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")
    wholemeter_filename = "wholeMeter.csv"
    checkpoint_identifier = "last.pth"

    def __init__(
            self,
            Smodel:nn.Module,
            model: nn.Module,
            optimizer,
            scheduler,
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

        self.Smodel=Smodel
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._trainT_loader = TrainT_loader
        self.test_loader = test_loader
        self._max_epoch = max_epoch
        self._num_batches = num_batches
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.crossentropy = SimplexCrossEntropyLoss(reduction='none')
        self._storage = Storage(self._save_dir)
        self.writer = SummaryWriter(str(self._save_dir))
        c = self._config['Data_input']['num_class']
        self.meters = meters_registerPDA(c)

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
        self.Smodel.to(device=device)

    def run_step(self, t_data, cur_batch: int):
        T_img, T_target, T_filename = (
            t_data[0][0].to(self.device),
            t_data[0][1].to(self.device),
            t_data[1],
        )
        T_img = self._rising_augmentation(T_img, mode="image", seed=cur_batch)
        T_target = self._rising_augmentation(T_target.float(), mode="feature", seed=cur_batch)

        pred_t_list = []
        with torch.no_grad():
            with switch_bn(self.Smodel, 1):
                for i in range(10):
                    pred_tt = self.Smodel(T_img).softmax(1)
                    pred_t_list.append(pred_tt.unsqueeze(0).cpu().numpy())

        preds = np.concatenate(pred_t_list, axis=0)
        output_segs = np.mean(preds, axis=0)
        output_stds = np.std(preds, axis=0)
        idx_pos = output_segs >= 0.5
        idx_neg = output_segs < 0.5
        stds_pos = output_stds[idx_pos]
        stds_neg = output_stds[idx_neg]
        percent = 90
        th_pos = np.percentile(stds_pos, percent)
        th_neg = np.percentile(stds_neg, percent)
        # print('%d, %.6f, %.6f' % (percent, th_pos, th_neg))
        mask = np.logical_or(np.logical_and(idx_pos, output_stds < th_pos), np.logical_and(idx_neg, output_stds < th_neg))
        mask = mask.astype(np.uint8)

        pred_T = self.model(T_img).softmax(1)
        s_loss = self.crossentropy(pred_T, torch.Tensor(output_segs).to(self.device))
        s_loss = torch.mean(s_loss * torch.Tensor(mask).to(self.device))

        self.meters[f"trainT_dice"].add(
            pred_T.max(1)[1],
            T_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in T_filename],
        )

        return s_loss

    def train_loop(
            self,
            trainT_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            *args,
            **kwargs,
    ):
        self.model.train()
        batch_indicator = tqdm(range(self._num_batches))
        batch_indicator.set_description(f"Training Epoch {epoch:03d}")

        for cur_batch, (batch_id, t_data) in enumerate(zip(batch_indicator, trainT_loader)):
            self.optimizer.zero_grad()

            loss = self.run_step(t_data=t_data, cur_batch=cur_batch)
            loss.backward()
            self.optimizer.step()
            self.meters['s_loss'].add(loss.item())
            report_dict = self.meters.statistics()
            batch_indicator.set_postfix_statics(report_dict, cache_time=20)
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
            test_indicator.set_postfix_statics(report_dict)
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
                    trainT_loader=self._trainT_loader,
                    epoch=self.cur_epoch
                )

            with self.meters.focus_on("val"), torch.no_grad():
                val_metric, _ = self.eval_loop(self.test_loader,self.cur_epoch)

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


