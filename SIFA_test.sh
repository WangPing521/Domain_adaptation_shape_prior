#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=rrg-ebrahimi
save_dir=0603_SIFA
declare -a StringArray=(
#todo check lr, batch size, weight_cyc, weight_seg, weight_disc
"python SIFA_main.py seed=10 Data.kfold=1 Optim.lr=0.00001 Optim.disc_lr=0.0000001 DataLoader.batch_size=63 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_401lr601dlr_c1s2d205"
"python SIFA_main.py seed=10 Data.kfold=2 Optim.lr=0.00001 Optim.disc_lr=0.0000001 DataLoader.batch_size=63 weights.cyc_weight=1 weights.seg_weight=10 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_401lr601dlr_c1s10d205"
"python SIFA_main.py seed=10 Data.kfold=3 Optim.lr=0.00001 Optim.disc_lr=0.0000001 DataLoader.batch_size=63 weights.cyc_weight=10 weights.seg_weight=10 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_401lr601dlr_c10s10d205"


)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
