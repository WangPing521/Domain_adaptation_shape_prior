import os
import re
import typing as t
from typing import *
from PIL import Image, ImageFile
from collections import OrderedDict
from copy import deepcopy as dcopy
from loguru import logger
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from augment.pil_augment import ToLabel, ToTensor
from augment.synchronize import SequentialWrapper
from utils.general import path2Path, assert_list, map_

ImageFile.LOAD_TRUNCATED_IMAGES = True

typePath = t.TypeVar("typePath", str, Path)

# __all__ = ["DatasetBase", "extract_sub_dataset_based_on_scan_names", "get_stem"]


def read_image(path, mode):
    with Image.open(path) as image:
        return image.convert(mode)


def allow_extension(path: str, extensions: t.List[str]) -> bool:
    try:
        return Path(path).suffixes[0] in extensions
    except:  # noqa
        return False


def default_transform(subfolders) -> SequentialWrapper:
    return SequentialWrapper(
        img_transform=ToTensor(),
        target_transform=ToLabel(),
        if_is_target=[False] * (len(subfolders) - 1) + [True],
    )

def get_stem(path: typePath):
    return path2Path(path).stem


def check_folder_types(type_: str):
    assert type_.lower() in ("image", "img", "gt", "label"), type_


def is_image_folder(type_: str):
    if type_.lower() in ("image", "img"):
        return True
    return False


def make_memory_dictionary(root: str, mode: str, folders: t.List[str], extensions) -> t.OrderedDict[str, t.List[str]]:
    for subfolder in folders:
        assert (Path(root, mode, subfolder).exists() and Path(root, mode, subfolder).is_dir()), \
            os.path.join(root, mode, subfolder)

    items = [os.listdir(Path(os.path.join(root, mode, sub))) for sub in folders]
    cleaned_items = [sorted([x for x in item if allow_extension(x, extensions)]) for item in items]

    memory = OrderedDict()
    for subfolder, item in zip(folders, cleaned_items):
        memory[subfolder] = sorted([os.path.join(root, mode, subfolder, x_path) for x_path in item])

    sub_memory_len_list = [len(x) for x in memory.values()]
    assert len(set(sub_memory_len_list)) == 1, sub_memory_len_list
    return memory


class DatasetBase(Dataset):
    allow_extension = [".jpg", ".png"]

    def __init__(self, *, root_dir: str, mode: str, sub_folders: t.Union[t.List[str], str],
                 sub_folder_types: t.Union[t.List[str], str], transforms: SequentialWrapper = None,
                 group_re: str = None) -> None:
        """
        :param root_dir: dataset root
        :param mode: train or test mode
        :param sub_folders: the folder list inside train or test folder
        :param transforms: SequentialWrapper transformer
        :param group_re: regex to group scans
        """
        self._name: str = f"{self.__class__.__name__}-{mode}"
        self._mode: str = mode
        self._root_dir: str = root_dir
        Path(self._root_dir).mkdir(parents=True, exist_ok=True)

        self._sub_folders: t.List[str] = [sub_folders, ] if isinstance(sub_folders, str) else sub_folders
        sub_folder_types = [sub_folder_types, ] if isinstance(sub_folder_types, str) else sub_folder_types
        assert len(self._sub_folders) == len(sub_folder_types)
        for type_ in sub_folder_types:
            check_folder_types(type_)
        self._sub_folder_types = [is_image_folder(type_) for type_ in sub_folder_types]

        self._transforms = transforms if transforms else default_transform(self._sub_folders)

        logger.opt(depth=1).trace(f"Creating {self.__class__.__name__}")
        self._memory = self.set_memory_dictionary(
            make_memory_dictionary(self._root_dir, self._mode, self._sub_folders, self.allow_extension)
        )
        # pre-load
        self._is_preload = False
        self._preload_storage: OrderedDict = OrderedDict()

        # regex for scan
        self._pattern = group_re
        self._re_pattern = None

        if self._pattern:
            self._re_pattern = re.compile(self._pattern)

    def get_memory_dictionary(self) -> t.Dict[str, t.List[str]]:
        return OrderedDict({k: v for k, v in self._memory.items()})

    def set_memory_dictionary(self, new_dictionary: t.Dict[str, t.Any], deepcopy=True):
        assert isinstance(new_dictionary, dict)
        self._memory = dcopy(new_dictionary) if deepcopy else new_dictionary
        return self._memory

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def pattern(self):
        return self._pattern

    @property
    def mode(self) -> str:
        return self._mode

    def __len__(self) -> int:
        return int(len(self._memory[self._sub_folders[0]]))

    def __getitem__(self, index) -> t.Tuple[t.List[Tensor], str]:
        image_list, filename_list = self._getitem_index(index)
        filename = Path(filename_list[0]).stem

        images = [x for x, _t in zip(image_list, self._sub_folder_types) if _t]
        labels = [x for x, _t in zip(image_list, self._sub_folder_types) if not _t]

        images_, labels_ = self._transforms(images, labels)
        return [*images_, *labels_], filename

    def _getitem_index(self, index):
        image_list = self._preload_storage[index] if self._is_preload else \
            [read_image(self._memory[subfolder][index], "L") for subfolder in self._sub_folders]

        filename_list = [self._memory[subfolder][index] for subfolder in self._sub_folders]

        stem_set = set([get_stem(x) for x in filename_list])
        assert len(stem_set) == 1, stem_set

        return image_list.copy(), filename_list.copy()

    def _preload(self):
        logger.opt(depth=1).trace(f"preloading {len(self.get_scan_list())} {self.__class__.__name__} data ...")

        for index in tqdm(range(len(self)), total=len(self), disable=True):
            self._preload_storage[index] = \
                [read_image(self._memory[subfolder][index], "L") for subfolder in self._sub_folders]

    def preload(self):
        self._is_preload = True
        self._preload()

    def deload(self):
        self._is_preload = False
        del self._preload_storage
        self._preload_storage = OrderedDict()

    def is_preloaded(self) -> bool:
        return self._is_preload

    def _get_scan_name(self, stem: str) -> str:
        if self._re_pattern is None:
            raise RuntimeError("Putting group_re first, instead of None")
        try:
            group_name = self._re_pattern.search(stem).group(0)  # type: ignore
        except AttributeError:
            raise AttributeError(f"Cannot match pattern: {self._pattern} for {str(stem)}")
        return group_name

    def get_stem_list(self):
        return [get_stem(x) for x in self._memory[self._sub_folders[0]]]

    def get_scan_list(self):
        return sorted(set([self._get_scan_name(filename) for filename in self.get_stem_list()]))

    @property
    def transforms(self) -> SequentialWrapper:
        return self._transforms

    @transforms.setter
    def transforms(self, transforms: SequentialWrapper):
        assert isinstance(transforms, SequentialWrapper), type(transforms)
        self._transforms = transforms

class MedicalImageSegmentationDataset(Dataset):
    dataset_modes = ["ct_train", "ct_val","mr_train", "mr_val", 'train', 'val']
    allow_extension = [".jpg", ".png"]

    def __init__(
        self,
        root_dir: str,
        mode: str,
        sub_folders: List[str],
        transforms: SequentialWrapper = None,
        patient_pattern: str = None,
        verbose=True,
    ) -> None:
        """
        :param root_dir: main folder path of the dataset
        :param mode: the subfolder name of this root, usually train, val, test or etc.
        :param sub_folders: subsubfolder name of this root, usually img, gt, etc
        :param transforms: synchronized transformation for all the subfolders
        :param verbose: verbose
        """
        assert (
                len(sub_folders) == set(sub_folders).__len__()
        ), f"subfolders must be unique, given {sub_folders}."
        assert assert_list(
            lambda x: isinstance(x, str), sub_folders
        ), f"`subfolder` elements should be str, given {sub_folders}"
        self._name: str = f"{mode}_dataset"
        self._mode: str = mode
        self._root_dir = root_dir
        self._subfolders: List[str] = sub_folders
        self._transform = default_transform(self._subfolders)
        if transforms:
            self._transform = transforms
        self._verbose = verbose
        if self._verbose:
            print(f"->> Building {self._name}:\t")
        self._filenames = self._make_dataset(
            self._root_dir, self._mode, self._subfolders, verbose=verbose
        )
        self._debug = os.environ.get("PYDEBUG", "0") == "1"
        self._set_patient_pattern(patient_pattern)

    @property
    def subfolders(self) -> List[str]:
        return self._subfolders

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def is_debug(self) -> bool:
        return self._debug

    def get_filenames(self, subfolder_name=None) -> List[str]:
        if subfolder_name:
            return self._filenames[subfolder_name]
        else:
            return self._filenames[self.subfolders[0]]

    @property
    def dataset_pattern(self):
        return self._pattern

    @property
    def mode(self) -> str:
        return self._mode

    def __len__(self) -> int:
        if self._debug:
            return int(len(self._filenames[self.subfolders[0]]) / 10)
        return int(len(self._filenames[self.subfolders[0]]))

    def __getitem__(self, index) -> Tuple[List[Tensor], str]:
        img_list, filename_list = self._getitem_index(index)
        assert img_list.__len__() == self.subfolders.__len__()
        # make sure the filename is the same image
        assert (
            set(map_(lambda x: Path(x).stem, filename_list)).__len__() == 1
        ), f"Check the filename list, given {filename_list}."
        filename = Path(filename_list[0]).stem
        img_list = self._transform(*img_list)
        return img_list, filename

    def _getitem_index(self, index):
        img_list = [
            Image.open(self._filenames[subfolder][index])
            for subfolder in self.subfolders
        ]
        filename_list = [
            self._filenames[subfolder][index] for subfolder in self.subfolders
        ]
        return img_list, filename_list

    def _set_patient_pattern(self, pattern):
        """
        This set patient_pattern using re library.
        :param pattern:
        :return:
        """
        assert isinstance(pattern, str), pattern
        self._pattern = pattern
        self._re_pattern = re.compile(self._pattern)

    def _get_group_name(self, path: Union[Path, str]) -> str:
        if not hasattr(self, "_re_pattern"):
            raise RuntimeError(
                "Calling `_get_group_name` before setting `set_patient_pattern`"
            )
        if isinstance(path, str):
            path = Path(path)
        try:
            group_name = self._re_pattern.search(path.stem).group(0)
        except AttributeError:
            raise AttributeError(
                f"Cannot match pattern: {self._pattern} for path: {str(path)}"
            )
        return group_name

    def get_group_list(self):

        return sorted(
            list(
                set(
                    [
                        self._get_group_name(filename)
                        for filename in self.get_filenames()
                    ]
                )
            )
        )

    def set_transform(self, transform: SequentialWrapper) -> None:
        if not isinstance(transform, SequentialWrapper):
            raise TypeError(
                f"`transform` must be instance of `SequentialWrapper`, given {type(transform)}."
            )
        self._transform = transform

    @property
    def transform(self) -> Optional[SequentialWrapper]:
        return self._transform

    @classmethod
    def _make_dataset(
        cls, root: str, mode: str, subfolders: List[str], verbose=True
    ) -> Dict[str, List[str]]:
        assert mode in cls.dataset_modes, mode
        for subfolder in subfolders:
            assert (
                Path(root, mode, subfolder).exists()
                and Path(root, mode, subfolder).is_dir()
            ), os.path.join(root, mode, subfolder)

        items = [
            os.listdir(Path(os.path.join(root, mode, subfoloder)))
            for subfoloder in subfolders
        ]
        # clear up extension
        items = sorted(
            [
                [x for x in item if allow_extension(x, cls.allow_extension)]
                for item in items
            ]
        )
        assert set(map_(len, items)).__len__() == 1, map_(len, items)

        imgs = {}
        for subfolder, item in zip(subfolders, items):
            imgs[subfolder] = sorted(
                [os.path.join(root, mode, subfolder, x_path) for x_path in item]
            )
        assert (
            set(map_(len, imgs.values())).__len__() == 1
        ), "imgs list have component with different length."

        for subfolder in subfolders:
            if verbose:
                print(f"found {len(imgs[subfolder])} images in {subfolder}\t")
        return imgs

