#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=SIFA_0512
declare -a StringArray=(
# align
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=2 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c1s1d201"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=2 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.0001 Trainer.save_dir=${save_dir}/SIFA_c1s1d301"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=2 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.00001 Trainer.save_dir=${save_dir}/SIFA_c1s1d401"

"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=2 weights.cyc_weight=10 weights.seg_weight=1 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c10s1d201"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=2 weights.cyc_weight=10 weights.seg_weight=1 weights.disc_weight=0.0001 Trainer.save_dir=${save_dir}/SIFA_c10s1d301"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=2 weights.cyc_weight=10 weights.seg_weight=1 weights.disc_weight=0.00001 Trainer.save_dir=${save_dir}/SIFA_c10s1d401"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
