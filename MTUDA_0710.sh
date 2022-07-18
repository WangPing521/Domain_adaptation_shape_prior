#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=0717_MTUDA_mmwhs
declare -a StringArray=(
#-----------
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.5 Trainer.checkpoint_path=runs/${save_dir}/bs63_lkd1_cons05_seed1 Trainer.save_dir=${save_dir}/bs63_lkd1_cons05_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.5 Trainer.checkpoint_path=runs/${save_dir}/bs63_lkd1_cons05_seed2 Trainer.save_dir=${save_dir}/bs63_lkd1_cons05_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.5 Trainer.checkpoint_path=runs/${save_dir}/bs63_lkd1_cons05_seed3 Trainer.save_dir=${save_dir}/bs63_lkd1_cons05_seed3"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
