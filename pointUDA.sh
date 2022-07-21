#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=0721_pointUDA_mmwhs
declare -a StringArray=(
#-----------

"python PointCloudUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.001 Trainer.save_dir=${save_dir}/bs63_201dis_seed1"
#"python PointCloudUDA_main.py seed=20 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.001 Trainer.save_dir=${save_dir}/bs63_201dis_seed2"
#"python PointCloudUDA_main.py seed=30 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.001 Trainer.save_dir=${save_dir}/bs63_201dis_seed3"

"python PointCloudUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.01 Trainer.save_dir=${save_dir}/bs63_101dis_seed1"
#"python PointCloudUDA_main.py seed=20 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.01 Trainer.save_dir=${save_dir}/bs63_101dis_seed2"
#"python PointCloudUDA_main.py seed=30 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.01 Trainer.save_dir=${save_dir}/bs63_101dis_seed3"

"python PointCloudUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.2 Trainer.save_dir=${save_dir}/bs63_02dis_seed1"
#"python PointCloudUDA_main.py seed=20 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.2 Trainer.save_dir=${save_dir}/bs63_02dis_seed2"
#"python PointCloudUDA_main.py seed=30 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.2 Trainer.save_dir=${save_dir}/bs63_02dis_seed3"

"python PointCloudUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.5 Trainer.save_dir=${save_dir}/bs63_05dis_seed1"
#"python PointCloudUDA_main.py seed=20 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.5 Trainer.save_dir=${save_dir}/bs63_05dis_seed2"
#"python PointCloudUDA_main.py seed=30 Optim.lr=0.00001 DataLoader.batch_size=63 weights=0.5 Trainer.save_dir=${save_dir}/bs63_05dis_seed3"


)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
