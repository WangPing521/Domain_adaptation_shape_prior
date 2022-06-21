import random
from pathlib import Path

from typing import List, Tuple

from sklearn.model_selection import train_test_split

from augment.synchronize import SequentialWrapper
from dataset.mmwhs import MedicalDatasetInterface, SubMedicalDatasetBasedOnIndex
from utils import DATA_PATH
from utils.utils import fix_all_seed_within_context
from ._ioutils import downloading
from .base import MedicalImageSegmentationDataset


class PromiseDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1hZISuvq2OGk6MZDhZ-p5ebV0q0IXAlaf"
    zip_name = "Promise2012.zip"
    folder_name = "Promise2012"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None, patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode=mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class prostate_S2T_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            PromiseDataset,
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
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"Case\d+"
        )
        val_set = self.DataClass(
            root_dir=self.root_dir,
            mode="val",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"Case\d+"
        )
        test_set = self.DataClass(
            root_dir=self.root_dir,
            mode="test",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"Case\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)
        if val_transform:
            val_set.set_transform(val_transform)
            test_set.set_transform(val_transform)
        return train_set, val_set, test_set


class ProstateDataset(MedicalImageSegmentationDataset):
    folder_name = "ProstateDK"
    zip_name = "ProstateDK.zip"
    download_link = "https://drive.google.com/uc?id=1MngFjFmbO8lBHC0G6sbW7_kjjijQqSsu"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None, patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode=mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class prostate_S2T2S_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            ProstateDataset,
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
            patient_pattern=r"prostate_\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set

class prostate_T2S_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            ProstateDataset,
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
            patient_pattern=r"prostate_\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set

class prostate_T2S2T_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            ProstateDataset,
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
            patient_pattern=r"prostate_\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set