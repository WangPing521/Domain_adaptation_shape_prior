from pathlib import Path
from typing import Union, Dict, Any, Tuple

import rising.random as rr
import rising.transforms as rt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from arch.utils import FeatureExtractor
from loss.diceloss import DiceLoss
from loss.entropy import SimplexCrossEntropyLoss
from meters import Storage, MeterInterface, AverageValueMeter, UniversalDice
from meters.SummaryWriter import SummaryWriter
from utils import tqdm
from utils.general import path2Path, class2one_hot
from utils.image_save_utils import save_images
from utils.rising import RisingWrapper
from utils.utils import set_environment, write_yaml


def meters_registerSIFA(c):
    meters = MeterInterface()
    report_axis = list(range(1, c))

    with meters.focus_on("train"):
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter(
            f"trainT_dice", UniversalDice(C=c, report_axis=report_axis))

        # loss
        meters.register_meter(
            "loss_Dt_real_t", AverageValueMeter()
        )
        meters.register_meter(
            "loss_G", AverageValueMeter()
        )
        meters.register_meter(
            "loss_Dt_adv", AverageValueMeter()
        )
        meters.register_meter(
            "loss_E", AverageValueMeter()
        )
        meters.register_meter(
            "loss_U", AverageValueMeter()
        )
        meters.register_meter(
            "loss_Ds", AverageValueMeter()
        )
        meters.register_meter(
            "loss_Dp_advp1", AverageValueMeter()
        )

    with meters.focus_on("val"):
        meters.register_meter(
            f"valT_dice", UniversalDice(C=c, report_axis=report_axis)
        )
        meters.register_meter(
            f"test_dice", UniversalDice(C=c, report_axis=report_axis)
        )

    return meters


class SIFA_trainer:
    PROJECT_PATH = str(Path(__file__).parents[1])

    RUN_PATH = str(Path(PROJECT_PATH) / "runs")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")
    wholemeter_filename = "wholeMeter.csv"
    checkpoint_identifier = "last.pth"

    def __init__(
            self,
            Generator: nn.Module,
            discriminator_t: nn.Module,
            model: nn.Module,
            decoder: nn.Module,
            discriminator_s: nn.Module,
            discriminator_p1: nn.Module,
            discriminator_p2: nn.Module,
            optimizer_G,
            optimizer_t,
            optimizer,
            optimizer_U,
            optimizer_s,
            optimizer_p1,
            optimizer_p2,
            scheduler_G,
            scheduler_t,
            scheduler,
            scheduler_U,
            scheduler_s,
            scheduler_p1,
            scheduler_p2,
            TrainS_loader: Union[DataLoader, _BaseDataLoaderIter],
            TrainT_loader: Union[DataLoader, _BaseDataLoaderIter],
            valT_loader: Union[DataLoader, _BaseDataLoaderIter],
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

        self.Generator = Generator
        self.discriminator_t = discriminator_t
        self.model = model
        self.decoder = decoder
        self.discriminator_s = discriminator_s
        self.discriminator_p1 = discriminator_p1
        self.discriminator_p2 = discriminator_p2
        self.optimizer_G = optimizer_G
        self.optimizer_t = optimizer_t
        self.optimizer = optimizer
        self.optimizer_U = optimizer_U
        self.optimizer_s = optimizer_s
        self.optimizer_p1 = optimizer_p1
        self.optimizer_p2 = optimizer_p2
        self.scheduler_G = scheduler_G
        self.scheduler_t = scheduler_t
        self.scheduler = scheduler
        self.scheduler_U = scheduler_U
        self.scheduler_s = scheduler_s
        self.scheduler_p1 = scheduler_p1
        self.scheduler_p2 = scheduler_p2
        self._trainS_loader = TrainS_loader
        self._trainT_loader = TrainT_loader
        self._valT_loader = valT_loader
        self._test_loader = test_loader
        self._max_epoch = max_epoch
        self._num_batches = num_batches
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.crossentropy = SimplexCrossEntropyLoss()
        self._bce_criterion = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self._storage = Storage(self._save_dir)
        self.writer = SummaryWriter(str(self._save_dir))
        self.extractor = FeatureExtractor(self.model,
                                          feature_names=[f"Conv{str(f)}" for f in range(5, 0, -1)]
                                          )
        self.extractor.bind()

        self.cycWeight = self._config['weights']['cyc_weight']
        self.segWeight = self._config['weights']['seg_weight']
        self.discWeight = self._config['weights']['disc_weight']

        c = self._config['Data_input']['num_class']
        self.meters = meters_registerSIFA(c)

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
        self.Generator.to(device=device)
        self.discriminator_t.to(device=device)
        self.model.to(device=device)
        self.decoder.to(device=device)
        self.discriminator_s.to(device=device)
        self.discriminator_p1.to(device=device)
        self.discriminator_p2.to(device=device)

    def run_step(self, s_data, t_data, cur_batch: int):
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
        T_target = self._rising_augmentation(T_target.float(), mode="feature", seed=cur_batch)

        fake = torch.zeros(S_img.shape[0], device=self.device).fill_(0)
        real = torch.zeros(S_img.shape[0], device=self.device).fill_(1)

        # update G_t
        self.optimizer_G.zero_grad()
        # G(s)->fake_t  EU(fake_t)->recov_s
        fakeS2T_img = torch.tanh(self.Generator(S_img))

        # visualiztion
        if cur_batch == 0:
            save_images(fakeS2T_img[1].detach(), names=[S_filename[1]], root=self._config['Trainer']['save_dir'], mode='S2T', iter=self.cur_epoch)

        with self.extractor.enable_register(True):
            self.extractor.clear()
            _ = self.model(fakeS2T_img).softmax(1)
            e_list_f = list(self.extractor.features())
            # todo: check the order
        fakeS2T2S_img = torch.tanh(self.decoder(e_list_f))

        if cur_batch == 0:
            save_images(fakeS2T2S_img[1].detach(), names=[S_filename[1]], root=self._config['Trainer']['save_dir'], mode='S2T2S', iter=self.cur_epoch)

        # EU(t)->fake_s G(fake_s)->recov_t
        with self.extractor.enable_register(True):
            self.extractor.clear()
            pred_T = self.model(T_img).softmax(1)
            e_list_T = list(self.extractor.features())
            # todo: check the order
        fakeT2S_img = torch.tanh(self.decoder(e_list_T))
        if cur_batch == 0:
            save_images(fakeT2S_img[1].detach(), names=[T_filename[1]], root=self._config['Trainer']['save_dir'], mode='T2S', iter=self.cur_epoch)

        fakeT2S2T_img = torch.tanh(self.Generator(fakeT2S_img.detach()))
        if cur_batch == 0:
            save_images(fakeT2S2T_img[1].detach(), names=[T_filename[1]], root=self._config['Trainer']['save_dir'], mode='T2S2S', iter=self.cur_epoch)

        # cycle consistency
        cycloss1 = torch.abs(S_img - fakeS2T2S_img).mean()  # # L1-norm loss
        cycloss2 = torch.abs(T_img - fakeT2S2T_img).mean()  # L1-norm loss
        loss_cyc = cycloss1 + 0.5 * cycloss2  # loss_cyc

        fakeS2T_img_0 = self.discriminator_t(fakeS2T_img).squeeze()
        loss_G_adv = self._bce_criterion(fakeS2T_img_0, real)  # loss_gan
        loss_G = loss_cyc + self.discWeight * loss_G_adv
        loss_G.backward()
        self.optimizer_G.step()

        # update D_t
        self.optimizer_t.zero_grad()
        fakeS2T_img_0 = self.discriminator_t(fakeS2T_img.detach()).squeeze()
        T_img_1 = self.discriminator_t(T_img).squeeze()
        loss_Dt_real_t = self._bce_criterion(T_img_1, real)
        loss_Dt_adv = self._bce_criterion(fakeS2T_img_0, fake)
        loss_Dt = loss_Dt_real_t + loss_Dt_adv
        loss_Dt.backward()
        self.optimizer_t.step()

        # update EC
        self.optimizer.zero_grad()
        predS2T_T = self.model(fakeS2T_img.detach()).softmax(1)
        onehot_targetS = class2one_hot(S_target.squeeze(1), predS2T_T.shape[1])
        loss_seg1 = self.crossentropy(predS2T_T, onehot_targetS) + self.dice_loss(predS2T_T, onehot_targetS)
        predS2T_T_0 = self.discriminator_p1(predS2T_T).squeeze()
        loss_E_advp = self._bce_criterion(predS2T_T_0, real)
        fakeT2S_img_0 = self.discriminator_s(fakeT2S_img).squeeze()
        loss_E_advs = self._bce_criterion(fakeT2S_img_0, real)

        with self.extractor.enable_register(True):
            self.extractor.clear()
            _ = self.model(fakeS2T_img.detach())
            e_list_f = list(self.extractor.features())
        fakeS2T2S_img = torch.tanh(self.decoder(e_list_f))

        # fakeS2T2S_img_1 = self.discriminator_s(fakeS2T2S_img).squeeze()
        # loss_E_advs1 = self._bce_criterion(fakeS2T2S_img_1, fake) # real
        fakeT2S2T_img = torch.tanh(self.Generator(fakeT2S_img))
        loss_cyc = torch.abs(S_img - fakeS2T2S_img).mean() + 0.5 * torch.abs(T_img - fakeT2S2T_img).mean()

        loss_E = self.cycWeight * loss_cyc + self.discWeight * loss_E_advs + self.segWeight * loss_seg1 + self.discWeight * loss_E_advp
        # + self.RegScheduler_advss.value * loss_E_advs1
        loss_E.backward()
        self.optimizer.step()

        # update U(Decoder)
        self.optimizer_U.zero_grad()
        e_list_T_detach = []
        for feature in e_list_T:
            e_list_T_detach.append(feature.detach())
        fakeT2S_img = torch.tanh(self.decoder(e_list_T_detach))
        fakeT2S_img_0 = self.discriminator_s(fakeT2S_img).squeeze()
        loss_E_advs = self._bce_criterion(fakeT2S_img_0, real)
        e_list_f_detach = []
        for fake_t_f in e_list_f:
            e_list_f_detach.append(fake_t_f.detach())
        fakeS2T2S_img = torch.tanh(self.decoder(e_list_f_detach))
        fakeT2S2T_img = torch.tanh(self.Generator(fakeT2S_img))
        loss_cyc = 0.5 * torch.abs(T_img - fakeT2S2T_img).mean() + torch.abs(S_img - fakeS2T2S_img).mean()
        loss_U = self.discWeight * loss_E_advs + self.cycWeight * loss_cyc
        loss_U.backward()
        self.optimizer_U.step()

        # update D_s
        self.optimizer_s.zero_grad()
        fakeT2S_img_0 = self.discriminator_s(fakeT2S_img.detach()).squeeze()
        S_img_1 = self.discriminator_s(S_img).squeeze()
        loss_Ds_advs = self._bce_criterion(fakeT2S_img_0, fake) + self._bce_criterion(S_img_1, real)
        # fakeS2T2S_img_1 = self.discriminator_s(fakeS2T2S_img.detach()).squeeze()
        # loss_Ds_advss = self._bce_criterion(fakeS2T2S_img_1, real)
        loss_Ds = loss_Ds_advs
        # + self.RegScheduler_advss.value * loss_Ds_advss
        loss_Ds.backward()
        self.optimizer_s.step()

        # update D_p
        self.optimizer_p1.zero_grad()
        predS2T_T_0 = self.discriminator_p1(predS2T_T.detach()).squeeze()
        pred_T_1 = self.discriminator_p1(pred_T.detach()).squeeze()
        loss_Dp_advp1 = self._bce_criterion(predS2T_T_0, fake) + self._bce_criterion(pred_T_1, real)
        loss_Dp_advp1.backward()
        self.optimizer_p1.step()

        self.meters[f"trainT_dice"].add(
            pred_T.max(1)[1],
            T_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in T_filename],
        )

        return loss_G, loss_Dt_real_t, loss_Dt, loss_E, loss_U, loss_Ds, loss_Dp_advp1

    def train_loop(
            self,
            trainS_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            trainT_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            *args,
            **kwargs,
    ):
        self.Generator.train()
        self.discriminator_t.train()
        self.model.train()
        self.decoder.train()
        self.discriminator_s.train()
        self.discriminator_p1.train()
        self.discriminator_p2.train()

        batch_indicator = tqdm(range(self._num_batches))
        batch_indicator.set_description(f"Training Epoch {epoch:03d}")
        report_dict = None, None

        for cur_batch, (batch_id, s_data, t_data) in enumerate(zip(batch_indicator, trainS_loader, trainT_loader)):
            loss_G, loss_Dt_real_t, loss_Dt, loss_E, loss_U, loss_Ds, loss_Dp_advp1 = self.run_step(s_data=s_data,
                                                                                                    t_data=t_data,
                                                                                                    cur_batch=cur_batch)
            self.meters['loss_Dt_real_t'].add(loss_Dt_real_t.item())
            self.meters['loss_G'].add(loss_G.item())
            self.meters['loss_Dt_adv'].add(loss_Dt.item())
            self.meters['loss_E'].add(loss_E.item())
            self.meters['loss_U'].add(loss_U.item())
            self.meters['loss_Ds'].add(loss_Ds.item())
            self.meters['loss_Dp_advp1'].add(loss_Dp_advp1.item())

            report_dict = self.meters.statistics()
            batch_indicator.set_postfix_statics(report_dict, cache_time=20)
        batch_indicator.close()

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
        report_dict = {}

        for batch_idT, data_T in enumerate(valT_indicator):
            imageT, targetT, filenameT = (
                data_T[0][0].to(self.device),
                data_T[0][1].to(self.device),
                data_T[1]
            )

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
            preds_test = self.model(image_test).softmax(1)
            self.meters[f"test_dice"].add(
                preds_test.max(1)[1],
                target_test.squeeze(1),
                group_name=["_".join(x.split("_")[:-1]) for x in filename_test])

            report_dict = self.meters.statistics()
            test_indicator.set_postfix_statics(report_dict, cache_time=20)
        test_indicator.close()

        assert report_dict is not None
        return dict(report_dict), self.meters["valT_dice"].summary()["DSC_mean"]

    def schedulerStep(self):
        self.scheduler_G.step()
        self.scheduler_t.step()
        self.scheduler.step()
        self.scheduler_U.step()
        self.scheduler_s.step()
        self.scheduler_p1.step()
        # self.scheduler_p2.step()

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
                val_metric, _ = self.eval_loop(self._valT_loader, self._test_loader, self.cur_epoch)

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
