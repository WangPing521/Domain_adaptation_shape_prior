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
from utils.image_save_utils import plot_seg
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
            "total_loss", AverageValueMeter()
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
            RegScheduler_advs,
            RegScheduler_cyc,
            RegScheduler_seg2,
            RegScheduler_advp1,
            RegScheduler_advp2,
            RegScheduler_advss,
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
        self.RegScheduler_advs = RegScheduler_advs
        self.RegScheduler_cyc  = RegScheduler_cyc
        self.RegScheduler_seg2 = RegScheduler_seg2
        self.RegScheduler_advp1= RegScheduler_advp1
        self.RegScheduler_advp2= RegScheduler_advp2
        self.RegScheduler_advss= RegScheduler_advss
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
        self.extractor = FeatureExtractor(self.model, feature_names="Up_conv2")
        self.extractor.bind()

        self.extractor_e1 = FeatureExtractor(self.model, feature_names="Conv1")
        self.extractor_e1.bind()
        self.extractor_e2 = FeatureExtractor(self.model, feature_names="Conv2")
        self.extractor_e2.bind()
        self.extractor_e3 = FeatureExtractor(self.model, feature_names="Conv3")
        self.extractor_e3.bind()
        self.extractor_e4 = FeatureExtractor(self.model, feature_names="Conv4")
        self.extractor_e4.bind()
        self.extractor_e5 = FeatureExtractor(self.model, feature_names="Conv5")
        self.extractor_e5.bind()

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

        # generator + E + U     In:s->t Out: s
        fakeS2T_img = torch.tanh(self.Generator(S_img))
        e_list_f = []
        with self.extractor_e1.enable_register(True), self.extractor_e2.enable_register(True),\
             self.extractor_e3.enable_register(True), self.extractor_e4.enable_register(True), self.extractor_e5.enable_register(True):
            self.extractor_e5.clear()
            self.extractor_e4.clear()
            self.extractor_e3.clear()
            self.extractor_e2.clear()
            self.extractor_e1.clear()
            predS2T_T = self.model(fakeS2T_img).softmax(1)
            # featureS2T_T = next(self.extractor.features())
            # classifier2
            e5 = next(self.extractor_e5.features())
            e_list_f.append(e5)
            e4 = next(self.extractor_e4.features())
            e_list_f.append(e4)
            e3 = next(self.extractor_e3.features())
            e_list_f.append(e3)
            e2 = next(self.extractor_e2.features())
            e_list_f.append(e2)
            e1 = next(self.extractor_e1.features())
            e_list_f.append(e1)
        fakeS2T2S_img = torch.tanh(self.decoder(e_list_f))

        # E + U + generator      In: t  Out: s
        e_list_T = []
        with self.extractor_e1.enable_register(True), self.extractor_e2.enable_register(True), self.extractor_e3.enable_register(True),\
             self.extractor_e4.enable_register(True), self.extractor_e5.enable_register(True):
            self.extractor_e5.clear()
            self.extractor_e4.clear()
            self.extractor_e3.clear()
            self.extractor_e2.clear()
            self.extractor_e1.clear()
            pred_T = self.model(T_img).softmax(1)
            e5 = next(self.extractor_e5.features())
            e_list_T.append(e5)
            e4 = next(self.extractor_e4.features())
            e_list_T.append(e4)
            e3 = next(self.extractor_e3.features())
            e_list_T.append(e3)
            e2 = next(self.extractor_e2.features())
            e_list_T.append(e2)
            e1 = next(self.extractor_e1.features())
            e_list_T.append(e1)
        fakeT2S_img = torch.tanh(self.decoder(e_list_T))
        fakeT2S2T_img = torch.tanh(self.Generator(fakeT2S_img))

        # Dt (s->t, t)
        fakeS2T_img_0 = self.discriminator_t(fakeS2T_img).squeeze()  # include Sigmoid
        T_img_1 = self.discriminator_t(T_img).squeeze()
        fakeS2T_img_0_gt = torch.zeros(fakeS2T_img_0.shape[0], device=self.device).fill_(0)
        T_img_1_gt = torch.zeros(T_img_1.shape[0], device=self.device).fill_(1)

        # Ds (t->s, s)
        S_img_1 = self.discriminator_s(S_img).squeeze()
        fakeT2S_img_0 = self.discriminator_s(fakeT2S_img).squeeze()
        fakeS2T2S_img_1 = self.discriminator_s(fakeS2T2S_img).squeeze()
        fakeT2S_img_0_gt = torch.zeros(fakeT2S_img_0.shape[0], device=self.device).fill_(0)
        S_img_1_gt = torch.zeros(S_img_1.shape[0], device=self.device).fill_(1)

        # discriminator_p1
        pred_T_1 = self.discriminator_p1(pred_T).squeeze()
        predS2T_T_0 = self.discriminator_p1(predS2T_T).squeeze()
        pred_T_1_gt = torch.zeros(predS2T_T_0.shape[0], device=self.device).fill_(1)
        predS2T_T_0_gt = torch.zeros(predS2T_T_0.shape[0], device=self.device).fill_(0)

        # discriminator_p2
        # pred_f_1 = self.discriminator_p2(pred_f, 1).squeeze()
        # pred_S2T_f_0 = self.discriminator_p2(pred_S2T_f, 0).squeeze()
        # pred_f_1_gt = torch.zeros(pred_f_1.shape[0], device=self.device).fill_(1)
        # pred_S2T_f_0_gt = torch.zeros(pred_S2T_f_0.shape[0], device=self.device).fill_(0)
        # loss_advp2 = self._bce_criterion(pred_f_1, pred_f_1_gt) + self._bce_criterion(pred_S2T_f_0, pred_S2T_f_0_gt)

        self.optimizer_G.zero_grad()
        # loss: CycleGan ----> Genetator loss
        cycloss1 = torch.abs(S_img - fakeS2T2S_img).mean() # # L1-norm loss
        cycloss2 = torch.abs(T_img - fakeT2S2T_img).mean() # L1-norm loss
        loss_cyc = 0.5*(cycloss1 + cycloss2) # loss_cyc
        loss_G_adv = self._bce_criterion(fakeS2T_img_0, T_img_1_gt) # loss_gan
        loss_G = self.RegScheduler_cyc.value * loss_cyc + self.RegScheduler_advs.value * loss_G_adv
        loss_G.backward(retain_graph=True)
        self.optimizer_G.step()

        self.optimizer_t.zero_grad()
        # loss: CycleGan ----> discriminoator_t loss
        loss_Dt_real_t = self._bce_criterion(T_img_1, T_img_1_gt)
        loss_Dt_adv = self._bce_criterion(fakeS2T_img_0, fakeS2T_img_0_gt)
        loss_Dt = loss_Dt_real_t + loss_Dt_adv
        loss_Dt.backward(retain_graph=True)
        self.optimizer_t.step()

        self.optimizer.zero_grad()
        # loss: Unet for segmentation: E+C
        loss_E_advs = self._bce_criterion(fakeT2S_img_0, S_img_1_gt)
        onehot_targetS = class2one_hot(S_target.squeeze(1), predS2T_T.shape[1])
        loss_seg1 = self.crossentropy(predS2T_T, onehot_targetS) + self.dice_loss(predS2T_T, onehot_targetS)
        # loss_seg2 = self.crossentropy(pred_S2T_f, onehot_targetS) + self.dice_loss(pred_S2T_f, onehot_targetS)
        loss_E_advp = self._bce_criterion(predS2T_T_0, pred_T_1_gt)
        loss_E_advs1 = self._bce_criterion(fakeS2T2S_img_1, S_img_1_gt)
        loss_E = self.RegScheduler_advs.value * loss_E_advs + \
                 self.RegScheduler_cyc.value * loss_cyc + \
                 loss_seg1 + \
                 self.RegScheduler_advp1.value * loss_E_advp + \
                 self.RegScheduler_advss.value * loss_E_advs1
        loss_E.backward(retain_graph=True)
        self.optimizer.step()

        self.optimizer_U.zero_grad()
        # loss: U(Decoder) for reconstruction
        loss_U = self.RegScheduler_advs.value * loss_E_advs + self.RegScheduler_cyc.value * loss_cyc
        loss_U.backward(retain_graph=True)
        self.optimizer_U.step()

        self.optimizer_s.zero_grad()
        # loss: CycleGan ----> discriminoator_s loss
        loss_Ds_advs = self._bce_criterion(fakeT2S_img_0, fakeT2S_img_0_gt) + self._bce_criterion(S_img_1, S_img_1_gt)
        loss_Ds_advss = self._bce_criterion(fakeS2T2S_img_1, S_img_1_gt)
        loss_Ds = self.RegScheduler_advs.value * loss_Ds_advs + self.RegScheduler_advss.value * loss_Ds_advss
        loss_Ds.backward(retain_graph=True)
        self.optimizer_s.step()

        self.optimizer_p1.zero_grad()
        loss_Dp_advp1 = self.RegScheduler_advp1.value * (self._bce_criterion(predS2T_T_0, predS2T_T_0_gt) + self._bce_criterion(pred_T_1, pred_T_1_gt))
        loss_Dp_advp1.backward()
        self.optimizer_p1.step()
        # self.optimizer_p2.zero_grad()
        # self.optimizer_p2.step()

        self.meters[f"trainT_dice"].add(
            pred_T.max(1)[1],
            T_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in T_filename],
        )

        return loss_G, loss_Dt, loss_E, loss_U, loss_Ds, loss_Dp_advp1

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
            loss = 0
            loss_G, loss_Dt_adv, loss_E, loss_U, loss_Ds, loss_Dp_advp1 = self.run_step(s_data=s_data, t_data=t_data, cur_batch=cur_batch)
            loss = loss_G + loss_Dt_adv + loss_E + loss_U + loss_Ds + loss_Dp_advp1
            self.meters['total_loss'].add(loss.item())
            self.meters['loss_G'].add(loss_G.item())
            self.meters['loss_Dt_adv'].add(loss_Dt_adv.item())
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
        self.RegScheduler_advs.step()
        self.RegScheduler_cyc.step()
        self.RegScheduler_seg2.step()
        self.RegScheduler_advp1.step()
        self.RegScheduler_advp2.step()
        self.RegScheduler_advss.step()

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


