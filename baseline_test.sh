#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=baseline_test

declare -a StringArray=(
  "python main.py Optim.lr=0.0001 Trainer.name=baseline    DataLoader.shuffle=True Trainer.save_dir=${save_dir}/CT2MRI_301lr_shufleT"
  "python main.py Optim.lr=0.00001 Trainer.name=baseline   DataLoader.shuffle=True Trainer.save_dir=${save_dir}/CT2MRI_401lr_shufleT"
  "python main.py Optim.lr=0.000001 Trainer.name=baseline  DataLoader.shuffle=True Trainer.save_dir=${save_dir}/CT2MRI_501lr_shufleT"
  "python main.py Optim.lr=0.0000001 Trainer.name=baseline DataLoader.shuffle=True Trainer.save_dir=${save_dir}/CT2MRI_601lr_shufleT"

  "python main.py Optim.lr=0.0001 Trainer.name=baseline    DataLoader.shuffle=False Trainer.save_dir=${save_dir}/CT2MRI_301lr_shufleF"
  "python main.py Optim.lr=0.00001 Trainer.name=baseline   DataLoader.shuffle=False Trainer.save_dir=${save_dir}/CT2MRI_401lr_shufleF"
  "python main.py Optim.lr=0.000001 Trainer.name=baseline  DataLoader.shuffle=False Trainer.save_dir=${save_dir}/CT2MRI_501lr_shufleF"
  "python main.py Optim.lr=0.0000001 Trainer.name=baseline DataLoader.shuffle=False Trainer.save_dir=${save_dir}/CT2MRI_601lr_shufleF"


)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


