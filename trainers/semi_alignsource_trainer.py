from pathlib import Path
from typing import Union, Dict, Any, Tuple

import torch
import torch.nn as nn
import rising.random as rr
import rising.transforms as rt
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from arch.projectors import DenseClusterHead
from arch.utils import FeatureExtractor
from loss.IIDSegmentations import IIDSegmentationLoss
from utils import tqdm
from utils.rising import RisingWrapper
from loss.entropy import SimplexCrossEntropyLoss
from meters import Storage
from meters.SummaryWriter import SummaryWriter
from scheduler.customized_scheduler import RampScheduler
from utils.general import class2one_hot, path2Path
from utils.image_save_utils import FeatureMapSaver
from utils.utils import set_environment, write_yaml, meters_register, fix_all_seed_within_context


class Semi_alignTrainer:

    def __init__(self,
            model: nn.Module,
            optimizer,
            scheduler,
            lab_loader,
            unlab_loader,
            val_loader,
            weight_scheduler,
            weight_cluster: RampScheduler,
            max_epoch: int = 100,
            save_dir: str = "base",
            checkpoint_path: str = None,
            device='cpu',
            config: dict = None,
            num_batches=200,
            *args,
            **kwargs) -> None:
        self._save_dir: Path = Path(self.RUN_PATH) / str(save_dir)
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._start_epoch = 0
        if config:
            self._config = config.copy()
            self._config.pop("Config", None)
            write_yaml(self._config, save_dir=self._save_dir, save_name="config.yaml")
            set_environment(config.get("Environment"))

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._lab_loader = lab_loader
        self._unlab_loader = unlab_loader
        self._val_loader = val_loader
        self._max_epoch = max_epoch
        self._num_batches = num_batches
        self._weight_scheduler = weight_scheduler
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.crossentropy = SimplexCrossEntropyLoss()
        self._storage = Storage(self._save_dir)
        self.writer = SummaryWriter(str(self._save_dir))
        c = self._config['Data_input']['num_class']
        self.meters = meters_register(c)
        self.displacement = self._config['DA']['displacement']
        if self.displacement:
            with fix_all_seed_within_context(self._config['Data']['seed']):
                # self.displacement_map_list = [(torch.randint(0, 9, (1,)), torch.randint(0, 9, (1,))) for i in
                # range(5)]
                self.displacement_map_list = [(2, 2), (4, 4), (8, 8)]
        else:
            self.displacement_map_list = [(0, 0)]

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

        self.projector = DenseClusterHead(
            input_dim=self.model.get_channel_dim(self._config['DA']['align_layer']['name']),
            num_clusters=self._config['DA']['align_layer']['clusters'])

        self.optimizer.add_param_group({'params': self.projector.parameters(),
                                        })

        self.extractor = FeatureExtractor(self.model, feature_names=self._config['DA']['align_layer']['name'])
        self.extractor.bind()
        self.saver = FeatureMapSaver(save_dir=self._save_dir)
        self.IICLoss = IIDSegmentationLoss()
    def to(self, device):
        self.model.to(device=device)


    def run_step(self, lab_data, unlab_data, cur_batch: int):
        extracted_layer = self.extractor.feature_names[0]
        C = int(self._config['Data_input']['num_class'])
        lab_img, lab_target, lab_filename = (
            lab_data[0][0].to(self.device),
            lab_data[0][1].to(self.device),
            lab_data[1],
        )
        unlab_img, unlab_target, unlab_filename = (
            unlab_data[0][0].to(self.device),
            unlab_data[0][1].to(self.device),
            unlab_data[1],
        )
        lab_img = self._rising_augmentation(lab_img, mode="image", seed=cur_batch)
        lab_target = self._rising_augmentation(lab_target.float(), mode="feature", seed=cur_batch)

        pred_lab = self.model(lab_img).softmax(1)
        onehot_lab_target = class2one_hot(lab_target.squeeze(1), C)
        s_loss = self.crossentropy(pred_lab, onehot_lab_target)

        if extracted_layer == 'Deconv_1x1':
            pred_unlab = self.model(unlab_img).softmax(1)

        else:
            pred_unlab = self.model(unlab_img).softmax(1)
            with self.extractor.enable_register(True):
                self.extractor.clear()
                _ = self.model(unlab_img, until=extracted_layer)
                feature_unlab = next(self.extractor.features())

            # projector cluster --->joint
            clusters_unlab = self.projector(feature_unlab)

        semi_loss = self.IICLoss(clusters_unlab,clusters_unlab)

        self.meters[f"train_dice"].add(
            pred_lab.max(1)[1],
            lab_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in lab_filename],
        )

        return s_loss, semi_loss

    def train_loop(
            self,
            trainS_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            trainT_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            *args,
            **kwargs,
    ):
        self.model.train()
        batch_indicator = tqdm(range(self._num_batches))
        batch_indicator.set_description(f"Training Epoch {epoch:03d}")
        report_dict, p_joint_S, p_joint_T = None, None, None

        for cur_batch, (batch_id, lab_data, unlab_data) in enumerate(zip(batch_indicator, trainS_loader, trainT_loader)):
            self.optimizer.zero_grad()

            s_loss, align_loss = self.run_step(lab_data=lab_data, unlab_data=unlab_data, cur_batch=cur_batch)
            loss = s_loss + self._weight_scheduler.value * align_loss

            loss.backward()
            self.optimizer.step()

            self.meters['total_loss'].add(loss.item())
            self.meters['s_loss'].add(s_loss.item())
            self.meters['align_loss'].add(align_loss.item())

            report_dict = self.meters.statistics()
            batch_indicator.set_postfix_statics(report_dict, cache_time=20)
        batch_indicator.close()

        assert report_dict is not None
        return dict(report_dict)

    def eval_loop(
            self,
            val_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            *args,
            **kwargs,
    ) -> Tuple[Any, Any]:
        self.model.eval()
        val_indicator = tqdm(val_loader)
        val_indicator.set_description(f"Val_Epoch {epoch:03d}")
        report_dict = {}
        for batch_id, data in enumerate(val_indicator):
            image, target, filename = (
                data[0][0].to(self.device),
                data[0][1].to(self.device),
                data[1]
            )
            preds = self.model(image).softmax(1)
            self.meters[f"valS_dice"].add(
                preds.max(1)[1],
                target.squeeze(1),
                group_name=["_".join(x.split("_")[:-1]) for x in filename])

            report_dict = self.meters.statistics()
            val_indicator.set_postfix_statics(report_dict, cache_time=20)
        val_indicator.close()

        assert report_dict is not None
        return dict(report_dict), self.meters["val_dice"].summary()["DSC_mean"]

    def schedulerStep(self):
        self._weight_scheduler.step()
        self._weight_cluster.step()
        self.scheduler.step()

    def start_training(self):
        self.to(self.device)
        self.cur_epoch = 0

        for self.cur_epoch in range(self._start_epoch, self._max_epoch):
            self.meters.reset()
            with self.meters.focus_on("train"):
                self.meters['lr'].add(self.optimizer.param_groups.__getitem__(0).get('lr'))
                self.meters["weight"].add(self._weight_scheduler.value)
                train_metrics = self.train_loop(
                    lab_loader=self._lab_loader,
                    unlab_loader=self._unlab_loader,
                    epoch=self.cur_epoch
                )

            with self.meters.focus_on("val"), torch.no_grad():
                val_metric, _ = self.eval_loop(self._val_loader, self.cur_epoch)

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
        if self._checkpoint is None:
            self._checkpoint = self._save_dir
        assert Path(self._checkpoint).exists(), Path(self._checkpoint)
        assert (Path(self._checkpoint).is_dir() and identifier is not None) or (
                Path(self._checkpoint).is_file() and identifier is None
        )

        state_dict = torch.load(
            str(Path(self._checkpoint) / identifier)
            if identifier is not None
            else self._checkpoint,
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
        self._best_score = state_dict["best_score"]
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

