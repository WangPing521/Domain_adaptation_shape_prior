#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=rrg-ebrahimi
save_dir=0603_SIFA
declare -a StringArray=(
#todo check lr  (note: maybe the batch size is too large for this method, leading to out of memory)
"python SIFA_main.py seed=10 Data.kfold=1 Optim.lr=0.00001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_502dlr_c1s2d205_fold1"
"python SIFA_main.py seed=10 Data.kfold=2 Optim.lr=0.00001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_502dlr_c1s2d205_fold2"
"python SIFA_main.py seed=10 Data.kfold=3 Optim.lr=0.00001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_502dlr_c1s2d205_fold3"

"python SIFA_main.py seed=10 Data.kfold=1 Optim.lr=0.00001 Optim.disc_lr=0.000001 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_501lr_c1s2d205_fold1"
"python SIFA_main.py seed=10 Data.kfold=2 Optim.lr=0.00001 Optim.disc_lr=0.000001 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_501lr_c1s2d205_fold2"
"python SIFA_main.py seed=10 Data.kfold=3 Optim.lr=0.00001 Optim.disc_lr=0.000001 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_501lr_c1s2d205_fold3"

"python SIFA_main.py seed=10 Data.kfold=1 Optim.lr=0.00001 Optim.disc_lr=0.0000001 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_601dlr_c1s2d205_fold1"
"python SIFA_main.py seed=10 Data.kfold=2 Optim.lr=0.00001 Optim.disc_lr=0.0000001 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_601dlr_c1s2d205_fold2"
"python SIFA_main.py seed=10 Data.kfold=3 Optim.lr=0.00001 Optim.disc_lr=0.0000001 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_601dlr_c1s2d205_fold3"

#todo check batch size, weight_cyc, weight_seg, weight_disc

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
