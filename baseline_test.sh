#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=baseline_test_mri

declare -a StringArray=(
  "python main.py Optim.lr=0.0001 Trainer.name=baseline    DA.source=MRI DA.target=CT DataLoader.shuffle=True Trainer.save_dir=${save_dir}/MRI2CT_301lr_shufleT"
  "python main.py Optim.lr=0.00001 Trainer.name=baseline   DA.source=MRI DA.target=CT DataLoader.shuffle=True Trainer.save_dir=${save_dir}/MRI2CT_401lr_shufleT"
  "python main.py Optim.lr=0.000001 Trainer.name=baseline  DA.source=MRI DA.target=CT DataLoader.shuffle=True Trainer.save_dir=${save_dir}/MRI2CT_501lr_shufleT"
  "python main.py Optim.lr=0.0000001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=True Trainer.save_dir=${save_dir}/MRI2CT_601lr_shufleT"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


