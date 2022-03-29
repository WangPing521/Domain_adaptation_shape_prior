import random
import re
from copy import deepcopy as dcp
from itertools import repeat
from pathlib import Path
from typing import Tuple, Type, Union, Dict, List, Callable, Pattern, Match

import numpy as np
from torch.utils.data import DataLoader, Sampler

from augment.synchronize import SequentialWrapper
from utils import DATA_PATH
from utils.general import map_, id_
from utils.rearr import ContrastBatchSampler
from utils.sampler import InfiniteRandomSampler
from utils.utils import fix_all_seed_within_context
from ._ioutils import downloading
from .base import MedicalImageSegmentationDataset


class MedicalDatasetInterface:
    """
    Semi-supervised interface for datasets using `MedicalImageSegmentationDataset`
    """

    def __init__(
            self,
            DataClass: Type[MedicalImageSegmentationDataset],
            root_dir: str,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__()
        self.DataClass = DataClass
        self.root_dir = root_dir
        self.seed = seed
        self.verbose = verbose

    def compile_dataloader_params(
            self,
            batch_size: int = 4,
            val_batch_size: int = None,
            shuffle: bool = False,
            num_workers: int = 1,
            pin_memory: bool = True,
            drop_last=False,
    ):
        self._if_use_indiv_bz: bool = self._use_individual_batch_size(
            batch_size,
            val_batch_size,
            self.verbose,
        )
        if self._if_use_indiv_bz:
            self.batch_params = {
                "batch_size": batch_size,
                "val_batch_size": val_batch_size,
            }
        self.dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
        }

    def DataLoaders(
            self,
            train_transform: SequentialWrapper = None,
            val_transform: SequentialWrapper = None,
            group_train=False,
            group_val=True,
            use_infinite_sampler: bool = False,
            batchsize_indicator: int = 6
    ) -> Tuple[DataLoader, DataLoader]:

        _dataloader_params = dcp(self.dataloader_params)
        train_set, val_set = self._create_datasets(
            train_transform=train_transform,
            val_transform=val_transform,
        )
        # labeled_dataloader
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("batch_size")}
            )
        if use_infinite_sampler:
            contrastive_sampler = ContrastBatchSampler(train_set, scan_sample_num=batchsize_indicator,
                                                       partition_sample_num=1,
                                                       shuffle=False)

            train_loader = (
                DataLoader(
                    train_set,
                    batch_sampler=contrastive_sampler,
                    **{k: v for k, v in _dataloader_params.items() if k != "shuffle" and k != "batch_size"},
                )
                if not group_train
                else self._grouped_dataloader(
                    train_set, use_infinite_sampler=True, **_dataloader_params
                )
            )
        else:
            # raise RuntimeError()
            train_loader = (
                DataLoader(train_set, **_dataloader_params)
                if not group_train
                else self._grouped_dataloader(
                    train_set, use_infinite_sampler=False, **_dataloader_params
                )
            )

        # val_dataloader
        _dataloader_params.update({"shuffle": False, "drop_last": False})
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("val_batch_size")}
            )
        val_loader = (
            DataLoader(val_set, **_dataloader_params)
            if not group_val
            else self._grouped_dataloader(val_set, **_dataloader_params)
        )
        del _dataloader_params
        return train_loader, val_loader

    @staticmethod
    def _use_individual_batch_size(
            batch_size, val_batch_size, verbose
    ) -> bool:
        if (
                isinstance(batch_size, int)
                and isinstance(val_batch_size, int)
        ):
            assert (
                    batch_size >= 1 and val_batch_size >= 1
            ), "batch_size should be greater than 1."
            if verbose:
                print(
                    f"Using train_batch_size={batch_size}, val_batch_size={val_batch_size}"
                )
            return True
        elif isinstance(batch_size, int) and batch_size >= 1:
            if verbose:
                print(f"Using all same batch size of {batch_size}")
            return False
        else:
            raise ValueError(
                f"batch_size setting error, given batch_size={batch_size}, "
                f"val_batch_size={val_batch_size}."
            )

    def _create_datasets(
            self,
            train_transform: SequentialWrapper = None,
            val_transform: SequentialWrapper = None,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
    ]:
        raise NotImplementedError

    def _grouped_dataloader(
            self,
            dataset: MedicalImageSegmentationDataset,
            use_infinite_sampler: bool = False,
            **dataloader_params: Dict[str, Union[int, float, bool]],
    ) -> DataLoader:
        """
        return a dataloader that requires to be grouped based on the reg of patient's pattern.
        :param dataset:
        :param shuffle:
        :return:
        """
        dataloader_params = dcp(dataloader_params)
        batch_sampler = PatientSampler(
            dataset=dataset,
            grp_regex=dataset._re_pattern,
            shuffle=dataloader_params.get("shuffle", False),
            verbose=self.verbose,
            infinite_sampler=True if use_infinite_sampler else False,
        )
        # having a batch_sampler cannot accept batch_size > 1
        dataloader_params["batch_size"] = 1
        dataloader_params["shuffle"] = False
        dataloader_params["drop_last"] = False
        return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_params)

    @staticmethod
    def override_transforms(
            dataset: MedicalImageSegmentationDataset, transform: SequentialWrapper
    ):
        assert isinstance(dataset, MedicalImageSegmentationDataset), dataset
        assert isinstance(transform, SequentialWrapper), transform
        dataset.set_transform(transform)
        return dataset


class mmWHSCTDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "MMWHS.zip"
    folder_name = "MMWHS"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None,
                 patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="ct_" + mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)


class mmWHSCTInterface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            mmWHSCTDataset,
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
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"ct_train_\d+"
        )
        val_set = self.DataClass(
            root_dir=self.root_dir,
            mode="val",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"ct_train_\d+"
        )
        with fix_all_seed_within_context(self.seed):
            shuffled_patients = train_set.get_group_list()[:]
            random.shuffle(shuffled_patients)

        if train_transform:
            train_set.set_transform(train_transform)
        if val_transform:
            val_set.set_transform(val_transform)
        return train_set, val_set


class mmWHSMRDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "MMWHS.zip"
    folder_name = "MMWHS"

    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], transforms: SequentialWrapper = None,
                 patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=str(path), mode="mr_" + mode, sub_folders=sub_folders,
                         transforms=transforms, patient_pattern=patient_pattern)


class mmWHSMRInterface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            mmWHSMRDataset,
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
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"mr_train_\d+"
        )
        val_set = self.DataClass(
            root_dir=self.root_dir,
            mode="val",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"mr_train_\d+"
        )
        with fix_all_seed_within_context(self.seed):
            shuffled_patients = train_set.get_group_list()[:]
            random.shuffle(shuffled_patients)

        if train_transform:
            train_set.set_transform(train_transform)
        if val_transform:
            val_set.set_transform(val_transform)
        return train_set, val_set


class PatientSampler(Sampler):
    def __init__(self, dataset: MedicalImageSegmentationDataset, grp_regex: str,
                 shuffle=False, verbose=True, infinite_sampler: bool = False) -> None:
        filenames: List[str] = dataset.get_filenames()
        self.grp_regex = grp_regex
        self.shuffle: bool = shuffle
        self.shuffle_fn: Callable = (
            lambda x: random.sample(x, len(x))
        ) if self.shuffle else id_
        self._infinite_sampler = infinite_sampler
        if verbose:
            print(f"Grouping using {self.grp_regex} regex")
        grouping_regex: Pattern = re.compile(self.grp_regex)

        stems: List[str] = [
            Path(filename).stem for filename in filenames
        ]  # avoid matching the extension
        matches: List[Match] = map_(grouping_regex.match, stems)
        patients: List[str] = [match.group(0) for match in matches]

        unique_patients: List[str] = sorted(list(set(patients)))
        assert len(unique_patients) < len(filenames)
        if verbose:
            print(
                f"Found {len(unique_patients)} unique patients out of {len(filenames)} images"
            )
        self.idx_map: Dict[str, List[int]] = dict(zip(unique_patients, repeat(None)))
        for i, patient in enumerate(patients):
            if not self.idx_map[patient]:
                self.idx_map[patient] = []
            self.idx_map[patient] += [i]
        assert sum(len(self.idx_map[k]) for k in unique_patients) == len(filenames)
        if verbose:
            print("Patient to slices mapping done")

    def __len__(self):
        return len(self.idx_map.keys())

    def __iter__(self):
        if not self._infinite_sampler:
            return self._one_iter()
        return self._infinite_iter()

    def _one_iter(self):
        values = list(self.idx_map.values())
        shuffled = self.shuffle_fn(values)
        return iter(shuffled)

    def _infinite_iter(self):

        while True:
            yield from self._one_iter()


# ---------semi-----
class MedicalDatasetSemiInterface:
    """
    Semi-supervised interface for datasets using `MedicalImageSegmentationDataset`
    """

    def __init__(
            self,
            DataClass: Type[MedicalImageSegmentationDataset],
            root_dir: str,
            labeled_data_ratio: float,
            unlabeled_data_ratio: float,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__()
        self.DataClass = DataClass
        self.root_dir = root_dir
        assert (
                       labeled_data_ratio + unlabeled_data_ratio
               ) <= 1, f"`labeled_data_ratio` + `unlabeled_data_ratio` should be less than 1.0, given {labeled_data_ratio + unlabeled_data_ratio}"
        self.labeled_ratio = labeled_data_ratio
        self.unlabeled_ratio = unlabeled_data_ratio
        self.val_ratio = 1 - (labeled_data_ratio + unlabeled_data_ratio)
        self.seed = seed
        self.verbose = verbose

    def compile_dataloader_params(
            self,
            batch_size: int = 1,
            labeled_batch_size: int = None,
            unlabeled_batch_size: int = None,
            val_batch_size: int = None,
            shuffle: bool = False,
            num_workers: int = 1,
            pin_memory: bool = True,
            drop_last=False,
    ):
        self._if_use_indiv_bz: bool = self._use_individual_batch_size(
            batch_size,
            labeled_batch_size,
            unlabeled_batch_size,
            val_batch_size,
            self.verbose,
        )
        if self._if_use_indiv_bz:
            self.batch_params = {
                "labeled_batch_size": labeled_batch_size,
                "unlabeled_batch_size": unlabeled_batch_size,
                "val_batch_size": val_batch_size,
            }
        self.dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
        }

    def SemiSupervisedDataLoaders(
            self,
            labeled_transform: SequentialWrapper = None,
            unlabeled_transform: SequentialWrapper = None,
            val_transform: SequentialWrapper = None,
            group_labeled=False,
            group_unlabeled=False,
            group_val=True,
            use_infinite_sampler: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

        _dataloader_params = dcp(self.dataloader_params)
        labeled_set, unlabeled_set, val_set = self._create_semi_supervised_datasets(
            labeled_transform=labeled_transform,
            unlabeled_transform=unlabeled_transform,
            val_transform=val_transform,
        )
        # labeled_dataloader
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("labeled_batch_size")}
            )
        if use_infinite_sampler:
            labeled_loader = (
                DataLoader(
                    labeled_set,
                    sampler=InfiniteRandomSampler(
                        labeled_set, shuffle=_dataloader_params.get("shuffle", False)
                    ),
                    **{k: v for k, v in _dataloader_params.items() if k != "shuffle"},
                )
                if not group_labeled
                else self._grouped_dataloader(
                    labeled_set, use_infinite_sampler=True, **_dataloader_params
                )
            )
        else:
            labeled_loader = (
                DataLoader(labeled_set, **_dataloader_params)
                if not group_labeled
                else self._grouped_dataloader(
                    labeled_set, use_infinite_sampler=False, **_dataloader_params
                )
            )

        # unlabeled_dataloader
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("unlabeled_batch_size")}
            )
        if use_infinite_sampler:
            unlabeled_loader = (
                DataLoader(
                    unlabeled_set,
                    sampler=InfiniteRandomSampler(
                        unlabeled_set, shuffle=_dataloader_params.get("shuffle", False)
                    ),
                    **{k: v for k, v in _dataloader_params.items() if k != "shuffle"},
                )
                if not group_unlabeled
                else self._grouped_dataloader(
                    unlabeled_set, use_infinite_sampler=True, **_dataloader_params
                )
            )
        else:
            unlabeled_loader = (
                DataLoader(unlabeled_set, **_dataloader_params)
                if not group_unlabeled
                else self._grouped_dataloader(
                    unlabeled_set, use_infinite_sampler=True, **_dataloader_params
                )
            )

        # val_dataloader
        _dataloader_params.update({"shuffle": False, "drop_last": False})
        if self._if_use_indiv_bz:
            _dataloader_params.update(
                {"batch_size": self.batch_params.get("val_batch_size")}
            )
        val_loader = (
            DataLoader(val_set, **_dataloader_params)
            if not group_val
            else self._grouped_dataloader(val_set, **_dataloader_params)
        )
        del _dataloader_params
        return labeled_loader, unlabeled_loader, val_loader

    @staticmethod
    def _use_individual_batch_size(
            batch_size, l_batch_size, un_batch_size, val_batch_size, verbose
    ) -> bool:
        if (
                isinstance(l_batch_size, int)
                and isinstance(un_batch_size, int)
                and isinstance(val_batch_size, int)
        ):
            assert (
                    l_batch_size >= 1 and un_batch_size >= 1 and val_batch_size >= 1
            ), "batch_size should be greater than 1."
            if verbose:
                print(
                    f"Using labeled_batch_size={l_batch_size}, unlabeled_batch_size={un_batch_size}, val_batch_size={val_batch_size}"
                )
            return True
        elif isinstance(batch_size, int) and batch_size >= 1:
            if verbose:
                print(f"Using all same batch size of {batch_size}")
            return False
        else:
            raise ValueError(
                f"batch_size setting error, given batch_size={batch_size}, labeled_batch_size={l_batch_size}, "
                f"unlabeled_batch_size={un_batch_size}, val_batch_size={val_batch_size}."
            )

    def _create_semi_supervised_datasets(
            self,
            labeled_transform: SequentialWrapper = None,
            unlabeled_transform: SequentialWrapper = None,
            val_transform: SequentialWrapper = None,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
    ]:
        raise NotImplementedError

    def _grouped_dataloader(
            self,
            dataset: MedicalImageSegmentationDataset,
            use_infinite_sampler: bool = False,
            **dataloader_params: Dict[str, Union[int, float, bool]],
    ) -> DataLoader:
        """
        return a dataloader that requires to be grouped based on the reg of patient's pattern.
        :param dataset:
        :param shuffle:
        :return:
        """
        dataloader_params = dcp(dataloader_params)
        batch_sampler = PatientSampler(
            dataset=dataset,
            grp_regex=dataset._re_pattern,
            shuffle=dataloader_params.get("shuffle", False),
            verbose=self.verbose,
            infinite_sampler=True if use_infinite_sampler else False,
        )
        # having a batch_sampler cannot accept batch_size > 1
        dataloader_params["batch_size"] = 1
        dataloader_params["shuffle"] = False
        dataloader_params["drop_last"] = False
        return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_params)

    @staticmethod
    def override_transforms(
            dataset: MedicalImageSegmentationDataset, transform: SequentialWrapper
    ):
        assert isinstance(dataset, MedicalImageSegmentationDataset), dataset
        assert isinstance(transform, SequentialWrapper), transform
        dataset.set_transform(transform)
        return dataset


class mmwhsCTSemiInterface(MedicalDatasetSemiInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            labeled_data_ratio: float = 0.3,
            unlabeled_data_ratio: float = 0.7,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            mmWHSCTDataset,
            root_dir,
            labeled_data_ratio,
            unlabeled_data_ratio,
            seed,
            verbose,
        )

    def _create_semi_supervised_datasets(
            self,
            labeled_transform: SequentialWrapper = None,
            unlabeled_transform: SequentialWrapper = None,
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
            patient_pattern=r"ct_train_\d+"
        )
        val_set = self.DataClass(
            root_dir=self.root_dir,
            mode="val",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"ct_train_\d+"
        )
        with fix_all_seed_within_context(self.seed):
            shuffled_patients = train_set.get_group_list()[:]
            random.shuffle(shuffled_patients)
            labeled_patients, unlabeled_patients = (
                shuffled_patients[: int(len(shuffled_patients) * self.labeled_ratio)],
                shuffled_patients[
                -int(len(shuffled_patients) * self.unlabeled_ratio):
                ],
            )

        labeled_set = SubMedicalDatasetBasedOnIndex(train_set, labeled_patients)
        unlabeled_set = SubMedicalDatasetBasedOnIndex(train_set, unlabeled_patients)
        assert len(labeled_set) + len(unlabeled_set) <= len(
            train_set
        ), "wrong on labeled/unlabeled split."
        del train_set
        if labeled_transform:
            labeled_set.set_transform(labeled_transform)
        if unlabeled_transform:
            unlabeled_set.set_transform(unlabeled_transform)
        if val_transform:
            val_set.set_transform(val_transform)
        return labeled_set, unlabeled_set, val_set


class mmwhsMRSemiInterface(MedicalDatasetSemiInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            labeled_data_ratio: float = 0.3,
            unlabeled_data_ratio: float = 0.7,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            mmWHSMRDataset,
            root_dir,
            labeled_data_ratio,
            unlabeled_data_ratio,
            seed,
            verbose,
        )

    def _create_semi_supervised_datasets(
            self,
            labeled_transform: SequentialWrapper = None,
            unlabeled_transform: SequentialWrapper = None,
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
            patient_pattern=r"mr_train_\d+"
        )
        val_set = self.DataClass(
            root_dir=self.root_dir,
            mode="val",
            sub_folders=["img", "gt"],
            transforms=None,
            patient_pattern=r"mr_train_\d+"
        )
        with fix_all_seed_within_context(self.seed):
            shuffled_patients = train_set.get_group_list()[:]
            random.shuffle(shuffled_patients)
            labeled_patients, unlabeled_patients = (
                shuffled_patients[: int(len(shuffled_patients) * self.labeled_ratio)],
                shuffled_patients[
                -int(len(shuffled_patients) * self.unlabeled_ratio):
                ],
            )

        labeled_set = SubMedicalDatasetBasedOnIndex(train_set, labeled_patients)
        unlabeled_set = SubMedicalDatasetBasedOnIndex(train_set, unlabeled_patients)
        assert len(labeled_set) + len(unlabeled_set) <= len(
            train_set
        ), "wrong on labeled/unlabeled split."
        del train_set
        if labeled_transform:
            labeled_set.set_transform(labeled_transform)
        if unlabeled_transform:
            unlabeled_set.set_transform(unlabeled_transform)
        if val_transform:
            val_set.set_transform(val_transform)
        return labeled_set, unlabeled_set, val_set


def SubMedicalDatasetBasedOnIndex(
        dataset: MedicalImageSegmentationDataset, group_list
) -> MedicalImageSegmentationDataset:
    """
    This class divide a list of file path to some different groups in order to split the dataset based on p_pattern string.
    """
    assert (
            isinstance(group_list, (tuple, list)) and group_list.__len__() >= 1
    ), f"group_list to be extracted: {group_list}"
    dataset = dcp(dataset)
    patient_img_list: List[str] = dataset.get_filenames()
    sub_patient_index = [
        dataset._get_group_name(f) in group_list for f in patient_img_list
    ]
    dataset._filenames = {
        k: np.array(v)[np.array(sub_patient_index)].tolist()
        for k, v in dataset._filenames.items()
    }
    return dataset
