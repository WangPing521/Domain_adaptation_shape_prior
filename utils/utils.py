# dictionary helper functions
import argparse
import collections
import collections.abc as container_abcs
import functools
import os
import random
import warnings
from contextlib import contextmanager
from copy import deepcopy as dcopy
from enum import Enum
from functools import reduce
from itertools import repeat
from pathlib import Path, PosixPath
from pprint import pprint
from typing import List, Dict, Any, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter  # noqa

from meters import MeterInterface, UniversalDice, AverageValueMeter
from utils.general import path2str, path2Path

logger_format = "<green>{time:MM/DD HH:mm:ss.SS}</green> | <level>{level: ^7}</level> |" \
                "{process.name:<5}.{thread.name:<5}: " \
                "<cyan>{name:<8}</cyan>:<cyan>{function:<10}</cyan>:<cyan>{line:<4}</cyan>" \
                " - <level>{message}</level>"


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, container_abcs.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def nice_dict(input_dict: Dict[str, Union[int, float]]) -> str:
    """
    this function is to return a nice string to dictionary displace propose.
    :param input_dict: dictionary
    :return: string
    """
    assert isinstance(
        input_dict, dict
    ), f"{input_dict} should be a dict, given {type(input_dict)}."
    is_flat_dict = True
    for k, v in input_dict.items():
        if isinstance(v, dict):
            is_flat_dict = False
            break
    flat_dict = input_dict if is_flat_dict else flatten_dict(input_dict, sep="")
    string_list = [f"{k}:{v:.3f}" for k, v in flat_dict.items()]
    return ", ".join(string_list)


def get_dataset(dataloader):
    if isinstance(dataloader, _BaseDataLoaderIter):
        return dataloader._dataset  # noqa
    elif isinstance(dataloader, DataLoader):
        return dataloader.dataset
    else:
        raise NotImplementedError(type(dataloader))


def multiply_iter(iter_a, iter_b):
    return [x * y for x, y in zip(iter_a, iter_b)]


def weighted_average_iter(a_list, weight_list):
    sum_weight = sum(weight_list) + 1e-16
    return sum(multiply_iter(a_list, weight_list)) / sum_weight


def pairwise_distances(x, y=None, recall_func=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
           recall function is a function to manipulate the distance.
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    if recall_func:
        return recall_func(dist)
    return dist


@contextmanager
def plt_interactive():
    plt.ion()
    try:
        yield
    finally:
        plt.ioff()


def extract_model_state_dict(trainer_checkpoint_path: str, *, keyword="_model"):
    trainer_state = torch.load(trainer_checkpoint_path, map_location="cpu")
    return trainer_state[keyword]


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


# reproducibility
def set_deterministic(enable=True):
    torch.backends.cudnn.benchmark = not enable  # noqa
    try:
        torch.use_deterministic_algorithms(enable)
    except:
        try:
            torch.set_deterministic(enable)
        finally:
            return


def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


@contextmanager
def fix_all_seed_for_transforms(seed):
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    fix_all_seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)  # noqa
        torch.random.set_rng_state(torch_state)  # noqa


@contextmanager
def fix_all_seed_within_context(seed):
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_support = torch.cuda.is_available()
    if cuda_support:
        torch_cuda_state = torch.cuda.get_rng_state()
        torch_cuda_state_all = torch.cuda.get_rng_state_all()
    fix_all_seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)  # noqa
        torch.random.set_rng_state(torch_state)  # noqa
        if cuda_support:
            torch.cuda.set_rng_state(torch_cuda_state)  # noqa
            torch.cuda.set_rng_state_all(torch_cuda_state_all)  # noqa


def ntuple(n):
    def parse(x):
        if isinstance(x, str):
            return tuple(repeat(x, n))
        if isinstance(x, container_abcs.Iterable):
            x = list(x)
            if len(x) == 1:
                return tuple(repeat(x[0], n))
            else:
                if len(x) != n:
                    raise RuntimeError(f"inconsistent shape between {x} and {n}")
            return x

        return tuple(repeat(x, n))

    return parse


_single = ntuple(1)
_pair = ntuple(2)
_triple = ntuple(3)
_quadruple = ntuple(4)


def adding_writable_sink(save_dir):
    abs_save_dir = os.path.abspath(save_dir)
    from loguru import logger
    logger.add(os.path.join(abs_save_dir, "loguru.log"), level="TRACE", backtrace=False, diagnose=True,
               format=logger_format)


def fix_seed(func):
    functools.wraps(func)

    def func_wrapper(*args, **kwargs):
        with fix_all_seed_within_context(1):
            return func(*args, **kwargs)

    return func_wrapper


def class_name(class_) -> str:
    return class_.__class__.__name__


def get_lrs_from_optimizer(optimizer: Optimizer) -> List[float]:
    return [p["lr"] for p in optimizer.param_groups]


@contextmanager
def disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    # let the track_running_stats to be inverse
    model.apply(switch_attr)
    # return the model
    try:
        yield
    # let the track_running_stats to be inverse
    finally:
        model.apply(switch_attr)


def get_model(model):
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
        return model.module
    elif isinstance(model, nn.Module):
        return model
    raise TypeError(type(model))


class switch_plt_backend:

    def __init__(self, env="agg") -> None:
        super().__init__()
        self.env = env

    def __enter__(self):
        self.prev = matplotlib.get_backend()
        matplotlib.use(self.env, force=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        matplotlib.use(self.prev, force=True)

    def __call__(self, func):
        functools.wraps(func)

        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


@contextmanager
def ignore_exception(*exceptions, log=True):
    if len(exceptions) == 0:
        exceptions = (Exception,)
    try:
        yield
    except exceptions as e:
        if log:
            logger.warning(e)
        else:
            pass


class YAMLargParser(object):
    """
        parse command line args for yaml type.
    """

    def __new__(
            cls,
            verbose: bool = True,
            k_v_sep1: str = ":",
            k_v_sep2: str = "=",
            hierarchy: str = ".",
            type_sep: str = "!",
    ) -> Dict[str, Any]:
        cls.k_v_sep1 = k_v_sep1
        cls.k_v_sep2 = k_v_sep2
        cls.type_sep = type_sep
        cls.hierachy = hierarchy
        cls.verbose = verbose
        args: List[str] = cls._setup()  # return a list of string using space, default by argparser.
        yaml_args: List[Dict[str, Any]] = [cls.parse_string(
            f, sep_1=cls.k_v_sep1, sep_2=cls.k_v_sep2, type_sep=cls.type_sep
        ) for f in args]
        hierarchical_dict_list = [cls.parse_hierachy(d) for d in yaml_args]
        merged_dict = cls.merge_dict(hierarchical_dict_list)
        if cls.verbose:
            print("-> Received Args:")
            pprint(merged_dict)

        return merged_dict

    @classmethod
    def _setup(cls) -> List[str]:
        parser = argparse.ArgumentParser("Augment parser for yaml config")
        parser.add_argument("strings", nargs="*", type=str, default=[""])
        args: argparse.Namespace = parser.parse_args()
        return (
            args.strings
        )  # return a list of string using space, default by argparser.

    @staticmethod
    def parse_string(string, sep_1=":", sep_2="=", type_sep="!") -> Dict[str, Any]:
        """
        support yaml parser of type:
        key:value
        key=value
        key:!type=value
        to be {key:value} or {key:type(value)}
        where `:` is the `sep_1`, `=` is the `sep_2` and `!` is the `type_sep`
        :param string:
        :param sep_1:
        :param sep_2:
        :param type_sep:
        :return: dict
        """
        if string == "":
            return {}

        if type_sep in string:
            # key:!type=value
            # assert sep_1 in string and sep_2 in string, f"Only support key:!type=value, given {string}."
            # assert string.find(sep_1) < string.find(sep_2), f"Only support key:!type=value, given {string}."
            string = string.replace(sep_1, ": ")
            string = string.replace(sep_2, " ")
            string = string.replace(type_sep, " !!")
        else:
            # no type here, so the input should be like key=value or key:value
            # assert (sep_1 in string) != (sep_2 in string), f"Only support a=b or a:b type, given {string}."
            string = string.replace(sep_1, ": ")
            string = string.replace(sep_2, ": ")

        return yaml.safe_load(string)

    @staticmethod
    def parse_hierachy(k_v_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            assert len(k_v_dict) <= 1
            if len(k_v_dict) == 0:
                return {}
        except TypeError:
            return {}
        key = list(k_v_dict.keys())[0]
        value = k_v_dict[key]
        keys = key.split(".")
        keys.reverse()
        for k in keys:
            d = {}
            d[k] = value
            value = dcopy(d)
        return dict(value)

    @staticmethod
    def merge_dict(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        args = reduce(lambda x, y: dict_merge(x, y, True), dict_list)
        return args


def dict_merge(dct: Dict[str, Any], merge_dct: Dict[str, Any], re=True):
    """
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    # dct = dcopy(dct)
    if merge_dct is None:
        if re:
            return dct
        else:
            return
    for k, v in merge_dct.items():
        if (
                k in dct
                and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)
        ):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    if re:
        return dcopy(dct)


def yaml_load(yaml_path: Union[Path, str], verbose=False) -> Dict[str, Any]:
    """
    load yaml file given a file string-like file path. return must be a dictionary.
    :param yaml_path:
    :param verbose:
    :return:
    """
    assert isinstance(yaml_path, (Path, str, PosixPath)), type(yaml_path)
    with open(path2str(yaml_path), "r") as stream:
        data_loaded: dict = yaml.safe_load(stream)
    if verbose:
        print(f"Loaded yaml path:{str(yaml_path)}")
        pprint(data_loaded)
    return data_loaded


class ConfigManger:
    DEFAULT_CONFIG = ""

    def __init__(
            self, DEFAULT_CONFIG_PATH: str = None, verbose=True, integrality_check=True
    ) -> None:
        self.parsed_args: Dict[str, Any] = YAMLargParser(verbose=verbose)
        if DEFAULT_CONFIG_PATH is None:
            warnings.warn(
                "No default yaml is provided, only used for parser input arguments.",
                UserWarning,
            )
            # stop running the following code, just self.parserd_args is validated
            return
        self.SET_DEFAULT_CONFIG_PATH(DEFAULT_CONFIG_PATH)
        if self.parsed_args.get("Config"):
            if Path(self.parsed_args["Config"]).is_dir():
                self.parsed_args["Config"] = os.path.join(
                    self.parsed_args["Config"], "config.yaml"
                )

        self.default_config: Dict[str, Any] = yaml_load(
            self.parsed_args.get("Config", self.DEFAULT_CONFIG), verbose=verbose
        )
        self.merged_config: Dict[str, Any] = dict_merge(
            self.default_config, self.parsed_args
        )
        if integrality_check:
            self._check_integrality(self.merged_config)
        if verbose:
            print("Merged args:")
            pprint(self.merged_config)

    @classmethod
    def SET_DEFAULT_CONFIG_PATH(cls, default_config_path: str) -> None:
        """
        check if the default config exits.
        :param default_config_path:
        :return: None
        """
        path: Path = Path(default_config_path)
        assert path.exists(), path
        assert path.is_file(), path
        assert path.with_suffix(".yaml") or path.with_suffix(".yml")
        cls.DEFAULT_CONFIG = str(default_config_path)

    @staticmethod
    def _check_integrality(merged_dict=Dict[str, Any]):
        # assert merged_dict.get(
        #     "Arch"
        # ), f"Merged dict integrity check failed,{merged_dict.keys()}"
        # assert merged_dict.get(
        #     "Optim"
        # ), f"Merged dict integrity check failed,{merged_dict.keys()}"
        assert merged_dict.get(
            "Scheduler"
        ), f"Merged dict integrity check failed,{merged_dict.keys()}"
        assert merged_dict.get(
            "Trainer"
        ), f"Merged dict integrity check failed,{merged_dict.keys()}"

    @property
    def config(self):
        try:
            # for those having the default config
            config = self.merged_config
        except AttributeError:
            # for those just use the command line
            config = self.parsed_args
        from collections import defaultdict

        return defaultdict(lambda: None, config)


def _warnings(*args, **kwargs):
    if len(args) > 0:
        warnings.warn(f"Received unassigned args with args: {args}.", UserWarning)
    if len(kwargs) > 0:
        kwarg_str = ", ".join([f"{k}:{v}" for k, v in kwargs.items()])
        warnings.warn(f"Received unassigned kwargs: \n{kwarg_str}", UserWarning)


def write_yaml(
        dictionary: Dict, save_dir: Union[Path, str], save_name: str, force_overwrite=True
) -> None:
    save_path = path2Path(save_dir) / save_name
    if save_path.exists():
        if force_overwrite is False:
            save_name = (
                    save_name.split(".")[0] + "_copy" + "." + save_name.split(".")[1]
            )
    with open(str(save_dir / save_name), "w") as outfile:  # type: ignore
        yaml.dump(dictionary, outfile, default_flow_style=False)


def set_environment(environment_dict: Dict[str, str] = None, verbose=True) -> None:
    if environment_dict:
        for k, v in environment_dict.items():
            os.environ[k] = str(v)
            if verbose:
                print(f"setting environment {k}:{v}")


class ModelMode(Enum):
    """ Different mode of model """

    TRAIN = "TRAIN"  # during training
    EVAL = "EVAL"  # eval mode. On validation data
    PRED = "PRED"

    @staticmethod
    def from_str(mode_str):
        """ Init from string
            :param mode_str: ['train', 'eval', 'predict']
        """
        if mode_str == "train":
            return ModelMode.TRAIN
        elif mode_str == "eval":
            return ModelMode.EVAL
        elif mode_str == "predict":
            return ModelMode.PRED
        else:
            raise ValueError("Invalid argument mode_str {}".format(mode_str))


def meters_register(c):
    meters = MeterInterface()
    report_axis = list(range(1, c))

    with meters.focus_on("train"):
        meters.register_meter("lr", AverageValueMeter())
        # train dice
        meters.register_meter(
            f"train_dice", UniversalDice(C=c, report_axis=report_axis))
        # meters.register_meter(
        #     f"trainT_dice", UniversalDice(C=c, report_axis=report_axis))

        # loss
        meters.register_meter(
            "total_loss", AverageValueMeter()
        )
        meters.register_meter(
            "s_loss", AverageValueMeter()
        )
        meters.register_meter(
            "align_loss", AverageValueMeter()
        )
        meters.register_meter(
            "cluster_loss", AverageValueMeter()
        )

        # weight
        meters.register_meter(
            "weight", AverageValueMeter()
        )

    with meters.focus_on("val"):
        meters.register_meter(
            f"valT_dice", UniversalDice(C=c, report_axis=report_axis)
        )
        meters.register_meter(
            f"test_dice", UniversalDice(C=c, report_axis=report_axis)
        )
    return meters
