#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=SIFA_0516
declare -a StringArray=(
# align
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c1s1d201"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c1s1d205"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=1 weights.disc_weight=0.01  Trainer.save_dir=${save_dir}/SIFA_c1s1d101"

"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c1s2d201"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c1s2d205"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=2 weights.disc_weight=0.01  Trainer.save_dir=${save_dir}/SIFA_c1s2d101"

"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=5 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c1s5d201"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=5 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c1s5d205"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=5 weights.disc_weight=0.01  Trainer.save_dir=${save_dir}/SIFA_c1s5d101"

"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=10 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c1s10d201"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=10 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c1s10d205"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=1 weights.seg_weight=10 weights.disc_weight=0.01  Trainer.save_dir=${save_dir}/SIFA_c1s10d101"

"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=1 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c10s1d201"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=1 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c10s1d205"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=1 weights.disc_weight=0.01  Trainer.save_dir=${save_dir}/SIFA_c10s1d101"

"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=2 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c10s2d201"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=2 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c10s2d205"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=2 weights.disc_weight=0.01  Trainer.save_dir=${save_dir}/SIFA_c10s2d101"

"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=5 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c10s5d201"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=5 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c10s5d205"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=5 weights.disc_weight=0.01  Trainer.save_dir=${save_dir}/SIFA_c10s5d101"

"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=10 weights.disc_weight=0.001 Trainer.save_dir=${save_dir}/SIFA_c10s10d201"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=10 weights.disc_weight=0.005 Trainer.save_dir=${save_dir}/SIFA_c10s10d205"
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 weights.cyc_weight=10 weights.seg_weight=10 weights.disc_weight=0.01  Trainer.save_dir=${save_dir}/SIFA_c10s10d101"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
