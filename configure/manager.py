from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy as dcp
from functools import reduce, partial
from pprint import pprint
from typing import List, Dict

from easydict import EasyDict as edict

from utils.general import path2Path
from utils.types import typePath
from ._merge_checker import merge_checker as _merge_checker
from .dictionary_utils import dictionary_merge_by_hierachy
from .yaml_parser import yamlArgParser, yaml_load

__config_dictionary__: OrderedDict = OrderedDict()

__all__ = ["ConfigManager", "get_config"]


def _load_yaml(config_path: typePath, verbose=False):
    config_path_ = path2Path(config_path)
    assert config_path_.is_file(), config_path
    return yaml_load(config_path_, verbose=verbose)


class ConfigManager:
    def __init__(self, *path: str, verbose: bool = True,
                 strict: bool = False, _test_message=None) -> None:
        if isinstance(path[0], (list, tuple)):
            path = path[0]
        self._parsed_args, parsed_config_path, parsed_extra_args_list = \
            yamlArgParser().parse(_test_message)
        self._path = parsed_config_path or path
        self._configs: List[Dict] = self.load_yaml(verbose=False)
        self._parsed_args_merge_check = self.merge_check(strict=strict)
        from .dictionary_utils import remove_dictionary_callback
        self._merged_config = reduce(
            partial(dictionary_merge_by_hierachy, deepcopy=True, hook_after_merge=remove_dictionary_callback),
            [*self._configs, self._parsed_args]
        )
        if verbose:
            self.show_configs()
            self.show_merged_config()

    def load_yaml(self, verbose=False) -> List[Dict]:
        config_list = [{}]
        if self._path:
            config_list = [_load_yaml(p, verbose=verbose) for p in self._path]
        return config_list

    def merge_check(self, strict=True):
        try:
            _merge_checker(
                base_dictionary=reduce(partial(dictionary_merge_by_hierachy, deepcopy=True), self._configs),
                coming_dictionary=self._parsed_args
            )
        except RuntimeError as e:
            if strict:
                raise e

    @contextmanager
    def __call__(self, config=None, scope="base"):
        assert scope not in __config_dictionary__, scope
        config = self.config if config is None else config
        __config_dictionary__[scope] = config
        try:
            yield config
        finally:
            del __config_dictionary__[scope]

    @property
    def parsed_config(self):
        return edict(dcp(self._parsed_args))

    @property
    def unmerged_configs(self):
        return [edict(x) for x in dcp(self._configs)]

    @property
    def merged_config(self):
        return edict(dcp(self._merged_config))

    @property
    def config(self):
        return self.merged_config

    def show_configs(self):
        print("parsed configs:")

        for i, (n, d) in enumerate(zip(self._path, self._configs)):
            print(f">>>>>>>>>>>({i}): {n} start>>>>>>>>>")
            pprint(d)
        else:
            print(f">>>>>>>>>> end >>>>>>>>>")

    def show_merged_config(self):
        print("merged configure:")
        pprint(self.merged_config)

    @property
    def path(self) -> List[str]:
        return [str(x) for x in self._path]


def get_config(scope):
    return __config_dictionary__[scope]
