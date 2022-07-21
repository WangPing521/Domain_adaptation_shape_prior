from pathlib import Path
from typing import List, Tuple

from augment.synchronize import SequentialWrapper
from dataset.mmwhs import MedicalDatasetInterface
from utils import DATA_PATH
from ._ioutils import downloading
from .base import MedicalImageSegmentationDataset

class promise_T_dataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1hZISuvq2OGk6MZDhZ-p5ebV0q0IXAlaf"
    zip_name = "Promise2012.zip"
    folder_name = "Promise2012"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None, patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path),mode= mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class promise_T_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            promise_T_dataset,
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
            patient_pattern=r"Case\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set

class promise_Ttest_dataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1hZISuvq2OGk6MZDhZ-p5ebV0q0IXAlaf"
    zip_name = "Promise2012.zip"
    folder_name = "Promise2012"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None, patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path),mode= mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class promise_Ttest_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            promise_Ttest_dataset,
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
            patient_pattern=r"Case\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set

class Prostate_T2S2T_Dataset(MedicalImageSegmentationDataset):
    folder_name = "prostate_CYC"
    zip_name = "prostate_CYC.zip"
    download_link = "https://drive.google.com/uc?id=1MngFjFmbO8lBHC0G6sbW7_kjjijQqSsu"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None, patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="recover_promise_" + mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class prostate_T2S2T_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            Prostate_T2S2T_Dataset,
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
            patient_pattern=r"Case\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set

class Prostate_T2S_Dataset(MedicalImageSegmentationDataset):
    folder_name = "prostate_CYC"
    zip_name = "prostate_CYC.zip"
    download_link = "https://drive.google.com/uc?id=1MngFjFmbO8lBHC0G6sbW7_kjjijQqSsu"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None, patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="fake_prostate_" + mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class prostate_T2S_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            Prostate_T2S_Dataset,
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
            patient_pattern=r"Case\d+"
        )

        if train_transform:
            train_set.set_transform(train_transform)

        return train_set

class Prostate_S2T_Dataset(MedicalImageSegmentationDataset):
    folder_name = "prostate_CYC"
    zip_name = "prostate_CYC.zip"
    download_link = "https://drive.google.com/uc?id=1MngFjFmbO8lBHC0G6sbW7_kjjijQqSsu"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None, patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="fake_promise_" + mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class prostate_S2T_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            Prostate_S2T_Dataset,
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



class prostate_S2T2S_dataset(MedicalImageSegmentationDataset):
    folder_name = "prostate_CYC"
    zip_name = "prostate_CYC.zip"
    download_link = "https://drive.google.com/uc?id=1MngFjFmbO8lBHC0G6sbW7_kjjijQqSsu"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None, patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="recover_prostate_" + mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)

class prostate_S2T2S_Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            prostate_S2T2S_dataset,
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

