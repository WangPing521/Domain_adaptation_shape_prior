#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=def-chdesa
save_dir=0625_SIFA_lr
declare -a StringArray=(
#todo check lr  (note: maybe the batch size is too large for this method, leading to out of memory)
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/501lr_502dlr_SIFA32_c1s2d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.0000002 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/501lr_602dlr_SIFA32_c1s2d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.0000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/501lr_605dlr_SIFA32_c1s2d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.00001 Optim.disc_lr=0.000001 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/401lr_501dlr_SIFA32_c1s2d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.00001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/401lr_505dlr_SIFA32_c1s2d205_seed1"

"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=48 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/501lr_502dlr_SIFA48_c1s2d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.0000002 DataLoader.batch_size=48 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/501lr_602dlr_SIFA48_c1s2d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.0000005 DataLoader.batch_size=48 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/501lr_605dlr_SIFA48_c1s2d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.00001 Optim.disc_lr=0.000001 DataLoader.batch_size=48 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/401lr_501dlr_SIFA48_c1s2d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.00001 Optim.disc_lr=0.000005 DataLoader.batch_size=48 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/401lr_505dlr_SIFA48_c1s2d205_seed1"


)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
