import os
from typing import Union, List
from submitters.functiontools import grid_search
from utils import PROJECT_PATH
from utils.utils import yaml_load
from collections.abc import Iterable
from pathlib import Path

OPT_PATH = str(Path(PROJECT_PATH, "opt"))
Path(OPT_PATH).mkdir(exist_ok=True, parents=True)

lr_zooms = [0.0001, 0.00005, 0.00001]

class ScriptGenerator:

    def __init__(self, *, save_dir) -> None:
        super().__init__()
        self.conditions = []
        self._save_dir = save_dir
        self.data_opt = Path(OPT_PATH) / ("mmwhs.yaml")


    def grid_search_on(self, *, seed: int, **kwargs):
        pass

    @staticmethod
    def _get_hyper_param_string(**kwargs):
        def to_str(v):
            if isinstance(v, Iterable) and (not isinstance(v, str)):
                return "_".join([str(x) for x in v])
            return v

        list_string = [f"{k}_{to_str(v)}" for k, v in kwargs.items()]
        prefix = "/".join(list_string)
        return prefix

class BaselineGenerator(ScriptGenerator):

    def __init__(self, *, save_dir, model_checkpoint=None) -> None:
        super().__init__(save_dir=save_dir)
        self._model_checkpoint = model_checkpoint
        self.conditions.append(f"Trainer.checkpoint_path={self._model_checkpoint or 'null'}")

    def generate_single_script(self, save_dir, seed, lr, align_layer, clusters, weight, cluster_weight, map_x, map_y):
        return f"python main.py Trainer.name=baseline Trainer.save_dir={save_dir} " \
               f" Optim.lr={lr:.7f} seed={str(seed)} " \
               f" {' '.join(self.conditions)}  "

    def grid_search_on(self, *, seed: Union[int, List[int]], **kwargs):
        jobs = []

        for param in grid_search(**{**kwargs, **{"seed": seed}}):
            random_seed = param.pop("seed")
            sub_save_dir = self._get_hyper_param_string(**param)
            true_save_dir = os.path.join(self._save_dir, "Seed_" + str(random_seed), sub_save_dir)

            job = " && ".join(
                [self.generate_single_script(save_dir=os.path.join(true_save_dir, "baseline", f"_{lr:02d}"),
                                             lr=lr, seed=random_seed)
                 for lr in lr_zooms])

            jobs.append(job)
        return jobs