#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=rrg-ebrahimi
save_dir=0606_SIFA
declare -a StringArray=(
#todo check lr  (note: maybe the batch size is too large for this method, leading to out of memory)
#"python SIFA_main.py seed=10 Data.kfold=1 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c1s2d205_fold1"
#"python SIFA_main.py seed=10 Data.kfold=2 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c1s2d205_fold2"
#"python SIFA_main.py seed=10 Data.kfold=3 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c1s2d205_fold3"

#todo check weight_cyc, weight_seg, weight_disc
#"python SIFA_main.py seed=10 Data.kfold=1 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=4 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c10s4d205_fold1"
#"python SIFA_main.py seed=10 Data.kfold=2 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=4 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c10s4d205_fold2"

#"python SIFA_main.py seed=10 Data.kfold=1 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=4 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c10s4d201_fold1"
#"python SIFA_main.py seed=10 Data.kfold=2 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=4 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c10s4d201_fold2"

"python SIFA_main.py seed=10 Data.kfold=1 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=5 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c10s5d205_fold1"
"python SIFA_main.py seed=10 Data.kfold=2 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=5 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c10s5d205_fold2"

"python SIFA_main.py seed=10 Data.kfold=1 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=5 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c10s5d201_fold1"
"python SIFA_main.py seed=10 Data.kfold=2 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=5 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c10s5d201_fold2"

"python SIFA_main.py seed=10 Data.kfold=1 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=10 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c10s10d205_fold1"
"python SIFA_main.py seed=10 Data.kfold=2 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=10 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c10s10d205_fold2"

"python SIFA_main.py seed=10 Data.kfold=1 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=10 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c10s10d201_fold1"
"python SIFA_main.py seed=10 Data.kfold=2 Optim.lr=0.000001 Optim.disc_lr=0.000002 DataLoader.batch_size=32 weights.cyc_weight=10 weights.seg_weight=10 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c10s10d201_fold2"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
