import os
import subprocess
from itertools import cycle
from typing import List, Union
from termcolor import colored
from submitters.functiontools import SubmitError, randomString, _create_sbatch_prefix

accounts = ["def-chdesa", ]

class SlurmSubmitter:
    def __init__(self, work_dir="./", stop_on_error=False, verbose=True, on_local=False,
                 account_list: List[str] = None) -> None:
        self._work_dir = work_dir
        self._env = []
        self._sbatch_kwargs = {}
        self._stop_on_error = stop_on_error
        self._verbose = verbose
        self._on_local = on_local
        self._def_account_iter = cycle(accounts)
        if account_list:
            self._def_account_iter = cycle(account_list)

    @property
    def absolute_work_dir(self) -> str:
        return os.path.abspath(self._work_dir)

    @property
    def env(self) -> str:
        return "\n".join(self._env)

    def configure_sbatch(self, **kwargs):
        self._sbatch_kwargs = kwargs
        self._configure_sbatch_done = True

    def configure_environment(self, cmd_list: Union[str, List[str]] = None):
        if isinstance(cmd_list, str):
            cmd_list = [cmd_list, ]
        self._env = cmd_list

    def submit(self, job: str, *, on_local: bool = None, force_show=False, **kwargs):

        if on_local is None:
            on_local = self._on_local  # take the global parameters

        cd_script = f"cd {self.absolute_work_dir}"

        if "account" not in kwargs:
            kwargs['account'] = next(self._def_account_iter)  # use global parameter

        full_script = "\n".join([
            _create_sbatch_prefix(**{**self._sbatch_kwargs, **kwargs}),  # slurm parameters
            self.env,  # set environment
            cd_script,  # go to the working folder
            job  # run job
        ])

        code = self._write_and_run(full_script, on_local=on_local, verbose=self._verbose, force_show=force_show)

        if code != 0:
            if self._stop_on_error:
                raise SubmitError(code)

    def _write_and_run(self, full_script, *, on_local: bool = False, verbose: bool = False, force_show=False):
        random_name = randomString() + ".sh"
        workdir = self.absolute_work_dir
        random_bash = os.path.join(workdir, random_name)

        if force_show:
            verbose = True
        with open(random_bash, "w") as f:
            f.write(full_script)
        try:
            if verbose:
                print(colored(full_script, "green"), "\n")
                if force_show:
                    return 0
            if on_local:
                code = subprocess.call(f"bash {random_bash}", shell=True)
            else:
                code = subprocess.call(f"sbatch {random_bash}", shell=True)
        finally:
            os.remove(random_bash)
        return code