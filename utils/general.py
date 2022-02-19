# in this file, no dependency on the other module.
import collections
import numbers
import sys
import types
from functools import partial
from functools import reduce
from multiprocessing import Pool
from operator import and_
from pathlib import Path
from typing import TypeVar, Any, Set, Callable, List, Iterable

from numpy import ndarray
import numpy as np
import six
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from typing_extensions import Protocol

T_path = TypeVar("T_path", str, Path)
A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)

def is_np_array(val):
    """
    Checks whether a variable is a numpy array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    out : bool
        True if the variable is a numpy array. Otherwise False.

    """
    # using np.generic here via isinstance(val, (np.ndarray, np.generic)) seems to also fire for scalar numpy values
    # even though those are not arrays
    return isinstance(val, np.ndarray)


def is_np_scalar(val):
    """
    Checks whether a variable is a numpy scalar.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    out : bool
        True if the variable is a numpy scalar. Otherwise False.

    """
    # Note that isscalar() alone also fires for thinks like python strings
    # or booleans.
    # The isscalar() was added to make this function not fire for non-scalar
    # numpy types. Not sure if it is necessary.
    return isinstance(val, np.generic) and np.isscalar(val)


def is_single_integer(val):
    """
    Checks whether a variable is an integer.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is an integer. Otherwise False.

    """
    return isinstance(val, numbers.Integral) and not isinstance(val, bool)


def is_single_float(val):
    """
    Checks whether a variable is a float.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a float. Otherwise False.

    """
    return (
        isinstance(val, numbers.Real)
        and not is_single_integer(val)
        and not isinstance(val, bool)
    )


def is_single_number(val):
    """
    Checks whether a variable is a number, i.e. an integer or float.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a number. Otherwise False.

    """
    return is_single_integer(val) or is_single_float(val)


def is_iterable(val):
    """
    Checks whether a variable is iterable.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is an iterable. Otherwise False.

    """
    return isinstance(val, collections.Iterable)


# TODO convert to is_single_string() or rename is_single_integer/float/number()
def is_string(val):
    """
    Checks whether a variable is a string.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a string. Otherwise False.

    """
    return isinstance(val, six.string_types)


def is_single_bool(val):
    """
    Checks whether a variable is a boolean.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a boolean. Otherwise False.

    """
    return isinstance(val, bool)


def is_integer_array(val):
    """
    Checks whether a variable is a numpy integer array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a numpy integer array. Otherwise False.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.integer)


def is_float_array(val):
    """
    Checks whether a variable is a numpy float array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a numpy float array. Otherwise False.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.floating)


def is_callable(val):
    """
    Checks whether a variable is a callable, e.g. a function.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a callable. Otherwise False.

    """
    # python 3.x with x <= 2 does not support callable(), apparently
    if sys.version_info[0] == 3 and sys.version_info[1] <= 2:
        return hasattr(val, "__call__")
    else:
        return callable(val)


def is_generator(val):
    """
    Checks whether a variable is a generator.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a generator. Otherwise False.

    """
    return isinstance(val, types.GeneratorType)


def is_tuple_or_list(val):
    """
    Checks whether a variable is a list or a tuple
    :param val: The variable to check
    :return: True if the variable is a list or a tuple, otherwise False
    """
    return isinstance(val, (list, tuple))

def is_map(value):
    return isinstance(value, collections.abc.Mapping)


def is_path(value):
    return isinstance(value, (str, Path))


def is_numeric(value):
    return isinstance(value, (int, float, Tensor, ndarray))


class Identical(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, m):
        return m


def identical(x: Any) -> Any:
    """
    identical function
    :param x: function x
    :return: function x
    """
    return x


# Assert utils
def uniq(a: Tensor) -> Set:
    """
    return unique element of Tensor
    Use python Optimized mode to skip assert statement.
    :rtype set
    :param a: input tensor
    :return: Set(a_npized)
    """
    return set([x.item() for x in a.unique()])


def sset(a: Tensor, sub: Iterable) -> bool:
    """
    if a tensor is the subset of the other
    :param a:
    :param sub:
    :return:
    """
    return uniq(a).issubset(sub)


def eq(a: Tensor, b: Tensor) -> bool:
    """
    if a and b are equal for torch.Tensor
    :param a:
    :param b:
    :return:
    """
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    """
    check if the matrix is the probability distribution
    :param t:
    :param axis:
    :return:
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, rtol=1e-4, atol=1e-4)


def one_hot(t: Tensor, axis=1) -> bool:
    """
    check if the Tensor is one hot.
    The tensor shape can be float or int or others.
    :param t:
    :param axis: default = 1
    :return: bool
    """
    return simplex(t, axis) and sset(t, [0, 1])


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert a.dtype == torch.int, a.dtype
    assert b.dtype == torch.int, b.dtype
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b


def probs2class(probs: Tensor, class_dim: int = 1) -> Tensor:
    assert simplex(probs, axis=class_dim)
    res = probs.argmax(dim=class_dim)
    return res


# @profile
def class2one_hot(seg: Tensor, C: int, class_dim: int = 1) -> Tensor:
    """
    make segmentation mask to be onehot
    """
    assert sset(seg, list(range(C)))

    return F.one_hot(seg.long(), C).moveaxis(-1, class_dim)


def probs2one_hot(probs: Tensor, class_dim: int = 1) -> Tensor:
    C = probs.shape[class_dim]
    assert simplex(probs, axis=class_dim)
    res = class2one_hot(probs2class(probs, class_dim=class_dim), C, class_dim=class_dim)
    assert res.shape == probs.shape
    assert one_hot(res, class_dim)
    return res


def logit2one_hot(logit: Tensor, class_dim: int = 1) -> Tensor:
    probs = F.softmax(logit, class_dim)
    return probs2one_hot(probs, class_dim)

def path2Path(path: T_path) -> Path:
    assert isinstance(path, (Path, str)), type(path)
    return Path(path) if isinstance(path, str) else path


def path2str(path: T_path) -> str:
    assert isinstance(path, (Path, str)), type(path)
    return str(path)


# functions
def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    with Pool() as pool:
        return list(pool.map(fn, iter))


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x


def assert_list(func: Callable[[A], bool], Iters: Iterable) -> bool:
    """
    List comprehensive assert for a function and a list of iterables.
    >>> assert assert_list(simplex, [torch.randn(2,10)]*10)
    :param func: assert function
    :param Iters:
    :return:
    """
    return reduce(and_, [func(x) for x in Iters])

def average_list(input_list):
    return sum(input_list) / len(input_list)

def average_iter(a_list):
    return sum(a_list) / float(len(a_list))


def iter_average(input_iter: Iterable):
    return sum(input_iter) / len(tuple(input_iter))


def to_float(value):
    if torch.is_tensor(value):
        return float(value.item())
    elif type(value).__module__ == "numpy":
        return float(value.item())
    elif type(value) in (float, int, str):
        return float(value)
    elif isinstance(value, collections.Mapping):
        return {k: to_float(o) for k, o in value.items()}
    elif isinstance(value, (tuple, list, collections.UserList)):
        return [to_float(o) for o in value]
    else:
        raise TypeError(f"{value.__class__.__name__} cannot be converted to float.")


def to_numpy(tensor):
    if (
        is_np_array(tensor)
        or is_np_scalar(tensor)
        or isinstance(tensor, numbers.Number)
    ):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif isinstance(tensor, collections.Mapping):
        return {k: to_numpy(o) for k, o in tensor.items()}
    elif isinstance(tensor, (tuple, list, collections.UserList)):
        return [to_numpy(o) for o in tensor]
    else:
        raise TypeError(f"{tensor.__class__.__name__} cannot be convert to numpy")


def to_torch(ndarray):
    if torch.is_tensor(ndarray):
        return ndarray
    elif type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray, numbers.Number):
        return torch.tensor(ndarray)
    elif isinstance(ndarray, collections.Mapping):
        return {k: to_torch(o) for k, o in ndarray.items()}
    elif isinstance(ndarray, (tuple, list, collections.UserList)):
        return [to_torch(o) for o in ndarray]
    else:
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))


def allow_extension(path: str, extensions: List[str]) -> bool:
    try:
        return Path(path).suffixes[0] in extensions
    except:
        return False

def to_device(obj, device, non_blocking=True):
    """
    Copy an object to a specific device asynchronizedly. If the param `main_stream` is provided,
    the copy stream will be synchronized with the main one.

    Args:
        obj (Iterable[Tensor] or Tensor): a structure (e.g., a list or a dict) containing pytorch tensors.
        dev (int): the target device.
        main_stream (stream): the main stream to be synchronized.

    Returns:
        a deep copy of the data structure, with each tensor copied to the device.

    """
    # Adapted from: https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/_functions.py
    if torch.is_tensor(obj):
        v = obj.to(device, non_blocking=non_blocking)
        return v
    elif isinstance(obj, collections.abc.Mapping):
        return {k: to_device(o, device, non_blocking) for k, o in obj.items()}
    elif isinstance(obj, (tuple, list, collections.UserList)):
        return [to_device(o, device, non_blocking) for o in obj]
    else:
        raise TypeError(f"{obj.__class__.__name__} cannot be converted to {device}")


class SizedIterable(Protocol):
    def __len__(self):
        pass

    def __next__(self):
        pass

    def __iter__(self):
        pass


class CriterionType(Protocol):

    def __call__(self, *args: Tensor, **kwargs) -> Tensor:
        pass
