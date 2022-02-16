#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=baseline_test

declare -a StringArray=(
#  "python main.py Optim.lr=0.00001 Trainer.name=baseline Trainer.save_dir=${save_dir}/CT2MRI_401lr"
#  "python main.py Optim.lr=0.000005 Trainer.name=baseline Trainer.save_dir=${save_dir}/CT2MRI_505lr"
#  "python main.py Optim.lr=0.000001 Trainer.name=baseline Trainer.save_dir=${save_dir}/CT2MRI_501lr"
  "python main.py Optim.lr=0.0000005 Trainer.name=baseline Trainer.save_dir=${save_dir}/CT2MRI_605lr"
  "python main.py Optim.lr=0.0000001 Trainer.name=baseline Trainer.save_dir=${save_dir}/CT2MRI_601lr"


)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


