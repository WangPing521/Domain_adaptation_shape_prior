#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=rrg-ebrahimi
save_dir=SIFA_0519
declare -a StringArray=(
# align
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=5 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_401lr501dlr_c1s2d205"

#"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=5 weights.cyc_weight=1 weights.seg_weight=5 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c1s5d201"
#"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=5 weights.cyc_weight=1 weights.seg_weight=5 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c1s5d205"

#"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=5 weights.cyc_weight=1 weights.seg_weight=10 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c1s10d201"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=5 weights.cyc_weight=1 weights.seg_weight=10 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_401lr501dlr_c1s10d205"
#"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=5 weights.cyc_weight=1 weights.seg_weight=10 weights.disc_weight=0.01  Trainer.save_dir=${save_dir}/SIFA_c1s10d101"

"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=5 weights.cyc_weight=10 weights.seg_weight=10 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_401lr501dlr_c10s10d205"

"python SIFA_main.py seed=123 Optim.disc_lr=0.00001 Optim.disc_lr=0.0000001 DA.source=MRI DA.target=CT DA.batchsize_indicator=5 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_401lr601dlr_c1s2d205"
"python SIFA_main.py seed=123 Optim.disc_lr=0.00001 Optim.disc_lr=0.0000001 DA.source=MRI DA.target=CT DA.batchsize_indicator=5 weights.cyc_weight=1 weights.seg_weight=10 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_401lr601dlr_c1s10d205"
"python SIFA_main.py seed=123 Optim.disc_lr=0.00001 Optim.disc_lr=0.0000001 DA.source=MRI DA.target=CT DA.batchsize_indicator=5 weights.cyc_weight=10 weights.seg_weight=10 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_401lr601dlr_c10s10d205"


)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
