#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=def-chdesa
save_dir=0716_MTUDA_runs_
declare -a StringArray=(
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.01 Trainer.save_dir=${save_dir}/lkd1_cons101_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.01 Trainer.save_dir=${save_dir}/lkd1_cons101_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.01 Trainer.save_dir=${save_dir}/lkd1_cons101_seed3"
##
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=0.1 weights.consistency.max_value=0.1  Trainer.save_dir=${save_dir}/lk01_cons01_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=0.1 weights.consistency.max_value=0.1  Trainer.save_dir=${save_dir}/lk01_cons01_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=0.1 weights.consistency.max_value=0.1  Trainer.save_dir=${save_dir}/lk01_cons01_seed3"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=0.1 weights.consistency.max_value=0.05 Trainer.save_dir=${save_dir}/lk01_cons105_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=0.1 weights.consistency.max_value=0.05 Trainer.save_dir=${save_dir}/lk01_cons105_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=0.1 weights.consistency.max_value=0.05 Trainer.save_dir=${save_dir}/lk01_cons105_seed3"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=0.1 weights.consistency.max_value=0.01 Trainer.save_dir=${save_dir}/lk01_cons101_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=0.1 weights.consistency.max_value=0.01 Trainer.save_dir=${save_dir}/lk01_cons101_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=0.1 weights.consistency.max_value=0.01 Trainer.save_dir=${save_dir}/lk01_cons101_seed3"


)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
