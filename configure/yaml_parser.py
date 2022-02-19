import argparse
from copy import deepcopy
from functools import reduce
from pathlib import Path
from pprint import pprint
from typing import List, Dict, Any, Tuple, Optional, Union

import yaml

from utils.general import path2Path
from .dictionary_utils import dictionary_merge_by_hierachy

__all__ = ["yamlArgParser", "str2bool", "yaml_load", "yaml_write"]



dType = Dict[str, Any]


class yamlArgParser:
    """
    parse command line args for yaml type.

    parsed_dict = YAMLArgParser()
    input:
    trainer.lr:!seq=[{1:2},{'yes':True}] lr.yes=0.94 lr.no=False
    output:
    {'lr': {'no': False, 'yes': 0.94}, 'trainer': {'lr': [{1: 2}, {'yes': True}]}}

    """

    def __init__(self, k_v_sep1: str = ":", k_v_sep2: str = "=", hierarchy: str = ".", type_sep: str = "!", ):
        self.__k_v_sep1 = k_v_sep1
        self.__k_v_sep2 = k_v_sep2
        self.__type_sep = type_sep
        self.__hierachy = hierarchy

    def parse(self, test_message=None) -> Tuple[dType, Optional[str], Optional[List[str]]]:
        parsed_args, base_filepath, extra_variable_list = self._setup(test_message)
        yaml_args: List[Dict[str, Any]] = [self.parse_string2flatten_dict(f) for f in parsed_args]
        hierarchical_dict_list = [self.create_dictionary_hierachy(d) for d in yaml_args]
        merged_args = self.merge_dict(hierarchical_dict_list)
        return merged_args, base_filepath, extra_variable_list

    @classmethod
    def _setup(cls, test_message: str = None) -> Tuple[List[str], Optional[str], List[str]]:
        parser = argparse.ArgumentParser(
            "Augment parser for yaml config", formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        path_parser = parser.add_argument_group("config path parser")
        path_parser.add_argument(
            "--path", type=str, required=False, default=None, nargs=argparse.ZERO_OR_MORE,
            help="base config path location",
        )
        # parser.add_argument("--opt-path", type=str, default=None, required=False, nargs=argparse.ZERO_OR_MORE,
        #                     help="optional config path locations", )
        parser.add_argument("optional_variables", nargs="*", type=str, default=[""], help="optional variables")
        args, extra_variables = parser.parse_known_args(test_message)
        return args.optional_variables, args.path, extra_variables

    def parse_string2flatten_dict(self, string) -> Dict[str, Any]:
        """
        support yaml parser of type:
        key:value
        key=value
        key:!type=value
        to be {key:value} or {key:type(value)}
        where `:` is the `sep_1`, `=` is the `sep_2` and `!` is the `type_sep`
        :param string: input string
        :param sep_1:
        :param sep_2:
        :param type_sep:
        :return: dict
        """
        if string == "" or len(string) == 0:
            return {}

        if self.__type_sep in string:
            # key:!type=value
            # assert sep_1 in string and sep_2 in string, f"Only support key:!type=value, given {string}."
            # assert string.find(sep_1) < string.find(sep_2), f"Only support key:!type=value, given {string}."
            string = string.replace(self.__k_v_sep1, ": ")
            string = string.replace(self.__k_v_sep2, " ")
            string = string.replace(self.__type_sep, " !!")
        else:
            # no type here, so the input should be like key=value or key:value
            # assert (sep_1 in string) != (sep_2 in string), f"Only support a=b or a:b type, given {string}."
            string = string.replace(self.__k_v_sep1, ": ")
            string = string.replace(self.__k_v_sep2, ": ")

        return yaml.safe_load(string)

    @staticmethod
    def create_dictionary_hierachy(k_v_dict: Dict[str, Any]) -> Dict[str, Any]:
        if k_v_dict is None or len(k_v_dict) == 0:
            return {}
        if len(k_v_dict) > 1:
            raise RuntimeError(k_v_dict)

        key = list(k_v_dict.keys())[0]
        value = k_v_dict[key]
        keys = sorted(key.split("."), reverse=True, key=lambda x: key.split(".").index(x))
        core = {keys[0]: deepcopy(value)}
        for k in keys[1:]:
            core = {k: core}

        return core

    @staticmethod
    def merge_dict(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        args = reduce(lambda x, y: dictionary_merge_by_hierachy(x, y, deepcopy=True), dict_list)
        return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def yaml_load(yaml_path: Union[Path, str], verbose=False) -> Dict[str, Any]:
    """
    load yaml file given a file string-like file path. return must be a dictionary.
    :param yaml_path:
    :param verbose:
    :return:
    """
    yaml_path = path2Path(yaml_path)
    assert path2Path(yaml_path).exists(), yaml_path
    if yaml_path.is_dir():
        if (yaml_path / "config.yaml").exists():
            yaml_path = yaml_path / "config.yaml"
        else:
            raise FileNotFoundError(f"config.yaml does not found in {str(yaml_path)}")

    with open(str(yaml_path), "r") as stream:
        data_loaded: dict = yaml.safe_load(stream)
    if verbose:
        print(f"Loaded yaml path:{str(yaml_path)}")
        pprint(data_loaded)
    return data_loaded


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
