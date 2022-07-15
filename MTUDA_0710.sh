#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=def-chdesa
save_dir=0715_MTUDA
declare -a StringArray=(
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=1 weights.consistency=1    Trainer.save_dir=${save_dir}/lkd1_cons1_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=1 weights.consistency=0.5  Trainer.save_dir=${save_dir}/lkd1_cons05_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=1 weights.consistency=0.1  Trainer.save_dir=${save_dir}/lkd1_cons01_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=1 weights.consistency=0.05 Trainer.save_dir=${save_dir}/lkd1_cons105_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=1 weights.consistency=0.01 Trainer.save_dir=${save_dir}/lkd1_cons101_seed1"
#
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.5 weights.consistency=1    Trainer.save_dir=${save_dir}/lkd05_cons1_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.5 weights.consistency=0.5  Trainer.save_dir=${save_dir}/lkd05_cons05_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.5 weights.consistency=0.1  Trainer.save_dir=${save_dir}/lkd05_cons01_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.5 weights.consistency=0.05 Trainer.save_dir=${save_dir}/lkd05_cons105_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.5 weights.consistency=0.01 Trainer.save_dir=${save_dir}/lkd05_cons101_seed1"
#
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.1 weights.consistency=1    Trainer.save_dir=${save_dir}/lkd1_cons1_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.1 weights.consistency=0.5  Trainer.save_dir=${save_dir}/lkd1_cons05_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.1 weights.consistency=0.1  Trainer.save_dir=${save_dir}/lkd1_cons01_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.1 weights.consistency=0.05 Trainer.save_dir=${save_dir}/lkd1_cons105_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.1 weights.consistency=0.01 Trainer.save_dir=${save_dir}/lkd1_cons101_seed1"
#
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.05 weights.consistency=1    Trainer.save_dir=${save_dir}/lkd105_cons1_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.05 weights.consistency=0.5  Trainer.save_dir=${save_dir}/lkd105_cons05_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.05 weights.consistency=0.1  Trainer.save_dir=${save_dir}/lkd105_cons01_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.05 weights.consistency=0.05 Trainer.save_dir=${save_dir}/lkd105_cons105_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.05 weights.consistency=0.01 Trainer.save_dir=${save_dir}/lkd105_cons101_seed1"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.01 weights.consistency=1    Trainer.save_dir=${save_dir}/lkd101_cons1_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.01 weights.consistency=0.5  Trainer.save_dir=${save_dir}/lkd101_cons05_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.01 weights.consistency=0.1  Trainer.save_dir=${save_dir}/lkd101_cons01_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.01 weights.consistency=0.05 Trainer.save_dir=${save_dir}/lkd101_cons105_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=8 weights.lkd_weight=0.01 weights.consistency=0.01 Trainer.save_dir=${save_dir}/lkd101_cons101_seed1"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
