import functools
import random
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable, List, Tuple, TypeVar, Iterable, Union
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import Compose, InterpolationMode

from augment.pil_augment import Identity
from utils.utils import fix_all_seed_for_transforms
from . import pil_augment

_pil2pil_transform_type = Callable[[Image.Image], Image.Image]
_pil2tensor_transform_type = Callable[[Image.Image], Tensor]
_pil_list = List[Image.Image]

T = TypeVar("T")

__all__ = ["SequentialWrapper", "SequentialWrapperTwice"]


def get_transform(transform) -> Iterable[Callable[[T], T]]:
    if isinstance(transform, Compose):
        for x in transform.transforms:
            yield from get_transform(x)
    else:
        yield transform


@lru_cache()
def get_interpolation(interp: str):
    return {"bilinear": InterpolationMode.BILINEAR, "nearest": InterpolationMode.NEAREST}[interp]


@contextmanager
def switch_interpolation(transforms: Callable[[T], Union[T, Tensor]], *, interp: str):
    assert interp in ("bilinear", "nearest"), interp
    previous_inters = OrderedDict()
    transforms = get_transform(transforms)
    interpolation = get_interpolation(interp)
    for id_, t in enumerate(transforms):
        if hasattr(t, "interpolation"):
            previous_inters[id_] = t.interpolation
            t.interpolation = interpolation
    try:
        yield
    finally:
        transforms = get_transform(transforms)
        for id_, t in enumerate(transforms):
            if hasattr(t, "interpolation"):
                t.interpolation = previous_inters[id_]


def random_int() -> int:
    return random.randint(0, int(1e5))


def transform_(image, transform, seed):
    with fix_all_seed_for_transforms(seed):
        return transform(image)


def warning_suppress(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper

class FixRandomSeed:
    def __init__(self, random_seed: int = 0):
        self.random_seed = random_seed
        self.randombackup = random.getstate()
        self.npbackup = np.random.get_state()

    def __enter__(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def __exit__(self, *_):
        np.random.set_state(self.npbackup)
        random.setstate(self.randombackup)


class SequentialWrapper:
    """
    This is the wrapper for synchronized image transformation
    The idea is to define two transformations for images and targets, with randomness.
    The randomness is garanted by the same random seed
    """

    def __init__(
        self,
        img_transform: Callable = None,
        target_transform: Callable = None,
        if_is_target: Union[List[bool], Tuple[bool, ...]] = [],
    ) -> None:
        super().__init__()
        self.img_transform = img_transform if img_transform is not None else Identity()
        self.target_transform = (
            target_transform if target_transform is not None else Identity()
        )
        self.if_is_target = if_is_target

    def __call__(
        self, *imgs, random_seed=None
    ) -> List[Union[Image.Image, torch.Tensor, np.ndarray]]:
        # assert cases
        assert len(imgs) == len(
            self.if_is_target
        ), f"len(imgs) should match len(if_is_target), given {len(imgs)} and {len(self.if_is_target)}."
        # assert cases ends
        random_seed: int = int(random.randint(0, 1e8)) if random_seed is None else int(
            random_seed
        )  # type ignore

        _imgs: List[Image.Image] = []
        for img, if_target in zip(imgs, self.if_is_target):
            with FixRandomSeed(random_seed):
                _img = self._transform(if_target)(img)
            _imgs.append(_img)
        return _imgs

    def __repr__(self):
        return (
            f"img_transform:{self.img_transform}\n"
            f"target_transform:{self.target_transform}.\n"
            f"is_target: {self.if_is_target}"
        )

    def _transform(self, is_target: bool) -> Callable:
        assert isinstance(is_target, bool)
        return self.img_transform if not is_target else self.target_transform

# class SequentialWrapper:
#
#     def __init__(self, com_transform: _pil2pil_transform_type = None,
#                  image_transform: _pil2tensor_transform_type = pil_augment.ToTensor(),
#                  target_transform: _pil2tensor_transform_type = pil_augment.ToLabel()) -> None:
#         """
#         image -> comm_transform -> img_transform -> Tensor
#         target -> comm_transform -> target_transform -> Tensor
#         :param com_transform: common geo-transformation
#         :param image_transform: transformation only applied for images
#         :param target_transform: transformation only applied for targets
#         """
#         self._com_transform = com_transform
#         self._image_transform = image_transform
#         self._target_transform = target_transform
#
#     def __call__(self, images: _pil_list, targets: _pil_list = None, com_seed: int = None,
#                  img_seed: int = None, target_seed: int = None) -> Tuple[List[Tensor], List[Tensor]]:
#         com_seed = com_seed or random_int()
#         img_seed = img_seed or random_int()
#         target_seed = target_seed or random_int()
#
#         image_list_after_transform, target_list_after_transform = images, targets or []
#
#         if self._com_transform:
#             # comm is the optional
#             with switch_interpolation(self._com_transform, interp="bilinear"):
#                 image_list_after_transform = [transform_(image, self._com_transform, com_seed)
#                                               for image in image_list_after_transform]
#             if targets is not None:
#                 with switch_interpolation(self._com_transform, interp="nearest"):
#                     target_list_after_transform = [transform_(target, self._com_transform, com_seed)
#                                                    for target in target_list_after_transform]
#
#         image_list_after_transform = [transform_(image, self._image_transform, img_seed)
#                                       for image in image_list_after_transform]
#
#         if targets is not None:
#             with switch_interpolation(self._target_transform, interp="nearest"):
#                 target_list_after_transform = [transform_(target, self._target_transform, target_seed)
#                                                for target in target_list_after_transform]
#
#         return image_list_after_transform, target_list_after_transform
#
#     def __repr__(self):
#         return (
#             f"comm_transform:{self._com_transform}\n"
#             f"img_transform:{self._image_transform}.\n"
#             f"target_transform: {self._target_transform}"
#         )


class SequentialWrapperTwice(SequentialWrapper):

    def __init__(self, com_transform: _pil2pil_transform_type = None,
                 image_transform: _pil2tensor_transform_type = pil_augment.ToTensor(),
                 target_transform: _pil2tensor_transform_type = pil_augment.ToLabel(),
                 total_freedom=True) -> None:
        """
        :param total_freedom: if True, the two-time generated images are using different seeds for all aspect,
                              otherwise, the images are used different random seed only for img_seed
        """
        super().__init__(com_transform, image_transform, target_transform)
        self._total_freedom = total_freedom

    def __call__(self, image_list: _pil_list, target_list: _pil_list = None, seed: int = None, **kwargs) -> \
        Tuple[List[Tensor], List[Tensor]]:
        seed = seed or random_int()

        with fix_all_seed_for_transforms(seed):
            comm_seed1, comm_seed2 = random_int(), random_int()
            img_seed1, img_seed2 = random_int(), random_int()
            target_seed1, target_seed2 = random_int(), random_int()

            if self._total_freedom:
                image_list1, target_list1 = super(SequentialWrapperTwice, self).__call__(image_list, target_list,
                                                                                         comm_seed1, img_seed1,
                                                                                         target_seed1)
                image_list2, target_list2 = super(SequentialWrapperTwice, self).__call__(image_list, target_list,
                                                                                         comm_seed2, img_seed2,
                                                                                         target_seed2)
                return [*image_list1, *image_list2], [*target_list1, *target_list2]

            image_list1, target_list1 = super(SequentialWrapperTwice, self).__call__(image_list, target_list,
                                                                                     comm_seed1, img_seed1,
                                                                                     target_seed1)
            image_list2, target_list2 = super(SequentialWrapperTwice, self).__call__(image_list, target_list,
                                                                                     comm_seed1, img_seed2,
                                                                                     target_seed1)
            return [*image_list1, *image_list2], [*target_list1, *target_list2]
