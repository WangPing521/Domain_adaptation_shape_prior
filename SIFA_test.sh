#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=rrg-ebrahimi
save_dir=0606_SIFA
declare -a StringArray=(
#todo check lr  (note: maybe the batch size is too large for this method, leading to out of memory)
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=8 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA8_c1s2d205_seed1"
"python SIFA_main.py seed=20 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=8 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA8_c1s2d205_seed2"
"python SIFA_main.py seed=30 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=8 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA8_c1s2d205_seed3"

"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA32_c1s2d205_seed1"
"python SIFA_main.py seed=20 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA32_c1s2d205_seed2"
"python SIFA_main.py seed=30 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA32_c1s2d205_seed3"

"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=48 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA48_c1s2d205_seed1"
"python SIFA_main.py seed=20 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=48 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA48_c1s2d205_seed2"
"python SIFA_main.py seed=30 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=48 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA48_c1s2d205_seed3"

"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=63 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA63_c1s2d205_seed1"
"python SIFA_main.py seed=20 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=63 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA63_c1s2d205_seed2"
"python SIFA_main.py seed=30 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=63 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA63_c1s2d205_seed3"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
