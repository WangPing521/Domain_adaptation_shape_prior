from contextlib import AbstractContextManager

import torch
from torch import Tensor

from loss.entropy import Entropy

entropy_criterion = Entropy(reduction="none", eps=1e-8)

def imsat_loss(prediction: Tensor):
    pred = prediction.moveaxis(0, 1).reshape(prediction.shape[1], -1)
    margin = pred.mean(1, keepdims=True)

    mi = -entropy_criterion(pred.t()).mean() + entropy_criterion(margin.t()).mean()

    return -mi


def cluster_alignment_criterion(source_cluster, target_cluster):
    return torch.mean((source_cluster.mean(dim=0) - target_cluster.mean(dim=0)) ** 2)


class nullcontext(AbstractContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *excinfo):
        pass
