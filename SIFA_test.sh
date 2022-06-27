#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=def-chdesa
save_dir=0626_SIFA_weights
declare -a StringArray=(
#todo check lr  (note: maybe the batch size is too large for this method, leading to out of memory)
#"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/501lr_505dlr_SIFA32_c1s2d205_seed1"
#todo check weights  (note: maybe the batch size is too large for this method, leading to out of memory)
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.0005 Trainer.save_dir=${save_dir}/SIFA32_c1s1d305_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/SIFA32_c1s1d201_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/SIFA32_c1s1d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/SIFA32_c1s1d101_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.05   Trainer.save_dir=${save_dir}/SIFA32_c1s1d105_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.1    Trainer.save_dir=${save_dir}/SIFA32_c1s1d01_seed1"
#--
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.0005 Trainer.save_dir=${save_dir}/SIFA32_c1s2d305_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/SIFA32_c1s2d201_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/SIFA32_c1s2d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/SIFA32_c1s2d101_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.05   Trainer.save_dir=${save_dir}/SIFA32_c1s2d105_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.1    Trainer.save_dir=${save_dir}/SIFA32_c1s2d01_seed1"
#--
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=4 weights.disc_weight=0.0005 Trainer.save_dir=${save_dir}/SIFA32_c1s4d305_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=4 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/SIFA32_c1s4d201_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=4 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/SIFA32_c1s4d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=4 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/SIFA32_c1s4d101_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=4 weights.disc_weight=0.05   Trainer.save_dir=${save_dir}/SIFA32_c1s4d105_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.seg_weight=4 weights.disc_weight=0.1    Trainer.save_dir=${save_dir}/SIFA32_c1s4d01_seed1"

# ---------
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=1 weights.disc_weight=0.0005 Trainer.save_dir=${save_dir}/SIFA32_c2s1d305_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=1 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/SIFA32_c2s1d201_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=1 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/SIFA32_c2s1d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=1 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/SIFA32_c2s1d101_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=1 weights.disc_weight=0.05   Trainer.save_dir=${save_dir}/SIFA32_c2s1d105_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=1 weights.disc_weight=0.1    Trainer.save_dir=${save_dir}/SIFA32_c2s1d01_seed1"
#--
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=2 weights.disc_weight=0.0005 Trainer.save_dir=${save_dir}/SIFA32_c2s2d305_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=2 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/SIFA32_c2s2d201_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=2 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/SIFA32_c2s2d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=2 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/SIFA32_c2s2d101_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=2 weights.disc_weight=0.05   Trainer.save_dir=${save_dir}/SIFA32_c2s2d105_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=2 weights.disc_weight=0.1    Trainer.save_dir=${save_dir}/SIFA32_c2s2d01_seed1"
#--
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=4 weights.disc_weight=0.0005 Trainer.save_dir=${save_dir}/SIFA32_c2s4d305_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=4 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/SIFA32_c2s4d201_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=4 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/SIFA32_c2s4d205_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=4 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/SIFA32_c2s4d101_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=4 weights.disc_weight=0.05   Trainer.save_dir=${save_dir}/SIFA32_c2s4d105_seed1"
"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=2 weights.seg_weight=4 weights.disc_weight=0.1    Trainer.save_dir=${save_dir}/SIFA32_c2s4d01_seed1"



)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
