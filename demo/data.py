import typing as t

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose

from utils import DATA_PATH
from utils.sampler import InfiniteRandomSampler


class InverseColor:
    def __call__(self, image: Tensor):
        return 1 - image


def get_source_data(train=True):
    return MNIST(root=DATA_PATH, train=train, transform=ToTensor(), download=True)


def get_target_data(train=True):
    return MNIST(root=DATA_PATH, train=train, transform=Compose([ToTensor(), InverseColor()]), download=True)


def get_data(mode="source") -> t.Tuple[DataLoader, DataLoader]:
    assert mode in ("source", "target")
    if mode == "source":
        tra_data = get_source_data(train=True)
        test_data = get_source_data(train=False)
    elif mode == "target":
        tra_data = get_target_data(train=True)
        test_data = get_target_data(train=False)
    else:
        raise ValueError(mode)
    tra_loader = DataLoader(
        tra_data,
        sampler=InfiniteRandomSampler(tra_data, shuffle=True),
        num_workers=4,
        batch_size=100,
        persistent_workers=True
    )
    test_loader = DataLoader(test_data, num_workers=4, shuffle=False, batch_size=100)
    return tra_loader, test_loader
