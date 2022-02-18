import os
from pathlib import Path
import argparse
from easydict import EasyDict as edict
from submitters.functiontools import grid_search, yaml_write, dictionary_merge_by_hierachy, random_string, on_cc, \
    move_dataset
from submitters.jobsubmitters import SlurmSubmitter
from submitters.scriptsgenerators import BaselineGenerator
from utils import PROJECT_PATH
from utils.utils import yaml_load


accounts = "def-chdesa"
CONFIG_PATH = str(Path(PROJECT_PATH, "configs"))
TEMP_DIR = os.path.join(PROJECT_PATH, "scripts", ".temp_config")
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)

seed = 10
clusters_zoom = 5
align_layer="Up_conv2" #"['Up_conv4', 'Up_conv3', 'Up_conv2']
weights = [0, 0.01, 1, 10, 100]
cluster_weight = [0, 0.01, 0.1, 0.5]

class alignScriptGenerator(BaselineGenerator):

    def __init__(self, *, save_dir, model_checkpoint=None) -> None:
        super().__init__(save_dir=save_dir, model_checkpoint=model_checkpoint)

        self.hook_config = yaml_load(os.path.join(CONFIG_PATH, "config.yaml"))

    def get_hook_params(self, align_layer, cluster, weight, cluster_weight):
        return {
            "DA":
                {
                 "align_layer": {"name": align_layer,
                                 "clusters": cluster
                                 }
				 },
            "Scheduler":
                {
                    "RegScheduler":
                        {
                            "max_value": weight
                        },
                    "ClusterScheduler":
                        {
                            "max_value": cluster_weight
                        }
                }
        }

    def generate_single_script(self, save_dir, seed, lr, align_layer, clusters, weight, cluster_weight):
        return f"python main.py seed={str(seed)} Optim.lr={lr:.7f} Trainer.name=align_IndividualBN " \
               f"DA.align_layer.name={str(align_layer)} DA.align_layer.clusters={int(clusters)} " \
               f"Scheduler.RegScheduler.max_value={float(weight)} Trainer.save_dir={save_dir} " \
               f"Scheduler.ClusterScheduler.max_value={float(cluster_weight)} " \
               f" {' '.join(self.conditions)} " \


    def grid_search_on(self, *, seed, **kwargs):
        jobs = []

        for param in grid_search(**{**kwargs, **{"seed": seed}}):
            random_seed = param.pop("seed")
            hook_params = self.get_hook_params(**param)
            sub_save_dir = self._get_hyper_param_string(**param)
            merged_config = dictionary_merge_by_hierachy(self.hook_config, hook_params)
            config_path = yaml_write(merged_config, save_dir=TEMP_DIR, save_name=random_string() + ".yaml")
            true_save_dir = os.path.join(self._save_dir, "Seed_" + str(random_seed), sub_save_dir)

            job = " && ".join([self.generate_single_script(save_dir=os.path.join(true_save_dir, "align_IndividualBN"),
                                             lr=0.00001,seed=random_seed, align_layer=merged_config.get('DA').get('align_layer').get('name'),
                                                          clusters=merged_config.get('DA').get('align_layer').get('clusters'),
                                                          weight=merged_config.get('Scheduler').get('RegScheduler').get('max_value'),
                                                          cluster_weight=merged_config.get('Scheduler').get('ClusterScheduler').get('max_value'),
                                                           )])

            jobs.append(job)
        return jobs

if __name__ == '__main__':
    parser = argparse.ArgumentParser("align method")
    parser.add_argument("--save_dir", required=True, type=str, help="save_dir")
    parser.add_argument("--force-show", action="store_true")
    args = parser.parse_args()

    submittor = SlurmSubmitter(work_dir="", stop_on_error=False, on_local=not on_cc())
    submittor.configure_environment([
        "module load python/3.8 ",
        f"source ~/torchenv37/bin/activate ",
        'if [ $(which python) == "/usr/bin/python" ]',
        "then",
        "exit 9",
        "fi",
        "export OMP_NUM_THREADS=1",
        "export PYTHONOPTIMIZE=1",
        "export PYTHONWARNINGS=ignore ",
        "export CUBLAS_WORKSPACE_CONFIG=:16:8 ",
        "export LOGURU_LEVEL=TRACE",
        "echo $(pwd)",
        move_dataset(),
        "python -c 'import torch; print(torch.randn(1,1,1,1,device=\"cuda\"))'",
        "nvidia-smi"
    ])
    submittor.configure_sbatch(mem=16)
    save_dir = args.save_dir
    data_config_path = yaml_load(Path(CONFIG_PATH) / ('config' + ".yaml"))
    data_opt = edict(data_config_path)
    force_show = args.force_show

    script_generator = alignScriptGenerator(save_dir=save_dir)

    jobs = script_generator.grid_search_on(seed=seed, cluster=clusters_zoom,
                                           weight=weights, cluster_weight=cluster_weight, align_layer=align_layer)

    for j in jobs:
        submittor.submit(j, account=accounts, force_show=force_show, time=6)

