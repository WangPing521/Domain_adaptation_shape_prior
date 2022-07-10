import random
import re
from copy import deepcopy as dcp
from itertools import repeat
from pathlib import Path
from typing import Tuple, Type, Union, Dict, List, Callable, Pattern, Match

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Sampler

from augment.synchronize import SequentialWrapper
from dataset import MedicalDatasetInterface, SubMedicalDatasetBasedOnIndex
from utils import DATA_PATH
from utils.general import map_, id_
from utils.rearr import ContrastBatchSampler
from utils.sampler import InfiniteRandomSampler
from utils.utils import fix_all_seed_within_context
from ._ioutils import downloading
from .base import MedicalImageSegmentationDataset


class mmWHSS2TDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "MMWHS_CYC.zip"
    folder_name = "MMWHS_CYC"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None,
                 patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="fake_ct_" + mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class mmWHS_S2T_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            mmWHSS2TDataset,
            root_dir,
            seed,
            verbose,
        )

    def _create_datasets(
            self,
            train_transform: SequentialWrapper = None,
            val_transform: SequentialWrapper = None,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"mr_train_\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set

class mmWHST2S2TDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "MMWHS_CYC.zip"
    folder_name = "MMWHS_CYC"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None,
                 patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="recover_ct_" + mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class mmWHS_T2S2T_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            mmWHSS2TDataset,
            root_dir,
            seed,
            verbose,
        )

    def _create_datasets(
            self,
            train_transform: SequentialWrapper = None,
            val_transform: SequentialWrapper = None,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"ct_train_\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set

class mmWHST2SDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "MMWHS_CYC.zip"
    folder_name = "MMWHS_CYC"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None,
                 patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="fake_mr_" + mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class mmWHS_T2S_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            mmWHST2SDataset,
            root_dir,
            seed,
            verbose,
        )

    def _create_datasets(
            self,
            train_transform: SequentialWrapper = None,
            val_transform: SequentialWrapper = None,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"ct_train_\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set

class mmWHSS2T2SDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "MMWHS_CYC.zip"
    folder_name = "MMWHS_CYC"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None,
                 patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="recover_mr_" + mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class mmWHS_S2T2S_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            mmWHSS2T2SDataset,
            root_dir,
            seed,
            verbose,
        )

    def _create_datasets(
            self,
            train_transform: SequentialWrapper = None,
            val_transform: SequentialWrapper = None,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"mr_train_\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set

class mmWHST2S_test_Dataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "MMWHS_CYC.zip"
    folder_name = "MMWHS_CYC"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None,
                 patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="fake_mr_" + mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class mmWHS_T2S_test_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            mmWHST2S_test_Dataset,
            root_dir,
            seed,
            verbose,
        )

    def _create_datasets(
            self,
            train_transform: SequentialWrapper = None,
            val_transform: SequentialWrapper = None,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="test",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"ct_train_\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set
