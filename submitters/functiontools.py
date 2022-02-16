import re

from utils import PROJECT_PATH
from utils.general import path2Path
from copy import deepcopy as dcopy
from typing import Union, Any, Dict, Iterable
from collections.abc import Mapping
from pathlib import Path
import yaml
import torch
import numpy as np
from itertools import product
from loguru import logger
import time
import random
import string

mapType = Mapping
DATA_PATH = str(Path(PROJECT_PATH) / ".data")

class SubmitError(RuntimeError):
    pass

def random_string(N=20):
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(N))

def randomString():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))

def move_dataset():
    return f" find {DATA_PATH}  " + "-name '*.zip' -exec cp {} $SLURM_TMPDIR \;"

def match_narval(name):
    match = re.compile(r"ng\d+").match(name)
    if match:
        return True
    return False

def on_cc() -> bool:
    """return if running on Compute Canada"""
    import socket
    hostname = socket.gethostname()
    # on beluga
    if "beluga" in hostname or "blg" in hostname:
        return True
    # on cedar
    if "cedar" in hostname or "cdr" in hostname:
        return True
    # on graham
    if "gra" in hostname:
        return True
    if "narval" in hostname or match_narval(hostname):
        return True
    return False

def on_cedar() -> bool:
    import socket
    hostname = socket.gethostname()
    if "cedar" in hostname or "cdr" in hostname:
        return True
    return False

def dictionary_merge_by_hierachy(dictionary1: Dict[str, Any], new_dictionary: Dict[str, Any] = None, deepcopy=True,
                                 hook_after_merge=None):
    """
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into``dct``.
    :return: None
    """
    if deepcopy:
        dictionary1, new_dictionary = dcopy(dictionary1), dcopy(new_dictionary)
    if new_dictionary is None:
        return dictionary1
    for k, v in new_dictionary.items():
        if k in dictionary1 and isinstance(dictionary1[k], mapType) and isinstance(new_dictionary[k], mapType):
            dictionary1[k] = dictionary_merge_by_hierachy(dictionary1[k], new_dictionary[k], deepcopy=False)
        else:
            dictionary1[k] = new_dictionary[k]
    if hook_after_merge:
        dictionary1 = hook_after_merge(dictionary1)
    return dictionary1

def yaml_write(
        dictionary: Dict, save_dir: Union[Path, str], save_name: str, force_overwrite=True
) -> str:
    save_path = path2Path(save_dir) / save_name
    path2Path(save_dir).mkdir(exist_ok=True, parents=True)
    if save_path.exists():
        if force_overwrite is False:
            save_path = (
                    save_name.split(".")[0] + "_copy" + "." + save_name.split(".")[1]
            )
    with open(str(save_path), "w") as outfile:  # type: ignore
        yaml.dump(dictionary, outfile, default_flow_style=False)
    return str(save_path)

def _create_sbatch_prefix(*, account: str, time: int = 4, job_name="default_job_name", nodes=1, gres=None,
                          cpus_per_task=6, mem: int = 16, mail_user="13220931629@163.com"):

    if gres is None:
        gres = "gpu:1"
        if on_cedar():
            gres = "gpu:p100:1"

    return (
        f"#!/bin/bash \n"
        f"#SBATCH --time=0-{time}:00 \n"
        f"#SBATCH --account={account} \n"
        f"#SBATCH --cpus-per-task={cpus_per_task} \n"
        f"#SBATCH --gres={gres} \n"
        f"#SBATCH --job-name={job_name} \n"
        f"#SBATCH --nodes={nodes} \n"
        f"#SBATCH --mem={mem}000M \n"
        f"#SBATCH --mail-user={mail_user} \n"
        f"#SBATCH --mail-type=FAIL \n"
    )

def is_true_iterator(value):
    if isinstance(value, Iterable):
        if not isinstance(value, (str, np.ndarray, torch.Tensor)):
            return True
    return False

def grid_search(max_num: int = None, **kwargs, ):
    max_N = 1
    for k, v in kwargs.copy().items():
        if is_true_iterator(v):
            max_N = max(max_N, len(v))
    for k, v in kwargs.copy().items():
        if is_true_iterator(v):
            kwargs[k] = iter(v)
        else:
            kwargs[k] = [v]
    result = []
    for value in product(*kwargs.values()):
        result.append(dict(zip(kwargs.keys(), value)))

    logger.info(f"Found {len(result)} combination of parameters.")

    if max_num is None:
        time.sleep(2)
        for case in result:
            yield case
    else:
        if len(result) <= max_num:
            for case in result:
                yield case
        else:
            logger.info(f"Randomly choosing {max_num} combination of parameters.")
            time.sleep(2)
            index = np.random.permutation(range(len(result)))[:max_num].tolist()
            index.sort()
            for i in index:
                yield result[i]
