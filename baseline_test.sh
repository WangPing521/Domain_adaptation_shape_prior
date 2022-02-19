#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=rrg-ebrahimi
save_dir=baseline

declare -a StringArray=(
  "python main.py seed=123 Optim.lr=0.00001 Trainer.name=baseline DA.source=CT DA.target=MRI  Trainer.save_dir=${save_dir}/CT2MRI_401lr_seed1"
  "python main.py seed=123 Optim.lr=0.000001 Trainer.name=baseline DA.source=CT DA.target=MRI Trainer.save_dir=${save_dir}/CT2MRI_501lr_seed1"

  "python main.py seed=123 Optim.lr=0.00001 Trainer.name=baseline DA.source=MRI DA.target=CT  Trainer.save_dir=${save_dir}/MRI2CT_401lr_seed1"
  "python main.py seed=123 Optim.lr=0.000001 Trainer.name=baseline DA.source=MRI DA.target=CT Trainer.save_dir=${save_dir}/MRI2CT_501lr_seed1"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


