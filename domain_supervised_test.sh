#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=supervised_joint_test

declare -a StringArray=(
  "python main.py seed=123 Optim.lr=0.000001 Trainer.name=supervised DA.source=CT DA.target=MRI Trainer.save_dir=${save_dir}/CT2MRI_supervised_seed1"
  "python main.py seed=213 Optim.lr=0.000001 Trainer.name=supervised DA.source=CT DA.target=MRI Trainer.save_dir=${save_dir}/CT2MRI_supervised_seed2"
  "python main.py seed=321 Optim.lr=0.000001 Trainer.name=supervised DA.source=CT DA.target=MRI Trainer.save_dir=${save_dir}/CT2MRI_supervised_seed3"

  "python main.py seed=123 Optim.lr=0.000001 Trainer.name=supervised DA.source=MRI DA.target=CT Trainer.save_dir=${save_dir}/MRI2CT_supervised_seed1"
  "python main.py seed=213 Optim.lr=0.000001 Trainer.name=supervised DA.source=MRI DA.target=CT Trainer.save_dir=${save_dir}/MRI2CT_supervised_seed2"
  "python main.py seed=321 Optim.lr=0.000001 Trainer.name=supervised DA.source=MRI DA.target=CT Trainer.save_dir=${save_dir}/MRI2CT_supervised_seed3"

  "python main.py seed=123 Optim.lr=0.00001 Trainer.name=supervised DA.source=MRI DA.target=CT Trainer.save_dir=${save_dir}/MRI2CT_401lr_supervised_seed1"
  "python main.py seed=213 Optim.lr=0.00001 Trainer.name=supervised DA.source=MRI DA.target=CT Trainer.save_dir=${save_dir}/MRI2CT_401lr_supervised_seed2"
  "python main.py seed=321 Optim.lr=0.00001 Trainer.name=supervised DA.source=MRI DA.target=CT Trainer.save_dir=${save_dir}/MRI2CT_401lr_supervised_seed3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


