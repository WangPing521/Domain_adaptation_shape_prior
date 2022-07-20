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

class PromiseInterface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
            kfold: int = 1
    ) -> None:
        super().__init__(
            PromiseDataset,
            root_dir,
            seed,
            verbose,
        )
        self.kfold=kfold

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
        test_set = self.DataClass(
            root_dir=self.root_dir,
            mode="test",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"Case\d+"
        )
        with fix_all_seed_within_context(self.seed):
            shuffled_patients = train_set.get_group_list()[:]
            rest_train, fold1 = train_test_split(shuffled_patients, test_size=0.25)
            sub_rest_train, fold2 = train_test_split(rest_train, test_size=0.33)
            fold4, fold3 = train_test_split(sub_rest_train, test_size=0.5)
            if self.kfold == 0:
                train_patients = shuffled_patients
                val_patients = shuffled_patients
            if self.kfold == 1:
                train_patients = rest_train
                val_patients = fold1
            elif self.kfold ==2:
                train_patients = sub_rest_train + fold1
                val_patients = fold2
            elif self.kfold ==3:
                train_patients = fold1 + fold2 + fold4
                val_patients = fold3
            elif self.kfold == 4:
                train_patients = fold1 + fold2 + fold3
                val_patients = fold4
        training_set = SubMedicalDatasetBasedOnIndex(train_set, train_patients)
        validation_set = SubMedicalDatasetBasedOnIndex(train_set, val_patients)
        if self.kfold == 0:
            assert len(validation_set) + len(training_set) == 2 * len(
                train_set
            ), "not full training data."
            del train_set
        else:
            assert len(validation_set) + len(training_set) <= len(
                train_set
            ), "wrong on labeled/unlabeled split."
            del train_set

        if train_transform:
            training_set.set_transform(train_transform)
        if val_transform:
            validation_set.set_transform(val_transform)
            test_set.set_transform(val_transform)
        return training_set, validation_set, test_set


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

class ProstateInterface(MedicalDatasetInterface):
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
