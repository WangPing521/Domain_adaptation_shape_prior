#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=baseline

declare -a StringArray=(
  "python main.py seed=123 Optim.lr=0.000001 Trainer.name=baseline DA.source=CT DA.target=MRI DataLoader.shuffle=True Trainer.save_dir=${save_dir}/CT2MRI_shuffleT_seed1"
  "python main.py seed=213 Optim.lr=0.000001 Trainer.name=baseline DA.source=CT DA.target=MRI DataLoader.shuffle=True Trainer.save_dir=${save_dir}/CT2MRI_shuffleT_seed2"
  "python main.py seed=321 Optim.lr=0.000001 Trainer.name=baseline DA.source=CT DA.target=MRI DataLoader.shuffle=True Trainer.save_dir=${save_dir}/CT2MRI_shuffleT_seed3"

  "python main.py seed=123 Optim.lr=0.000001 Trainer.name=baseline DA.source=CT DA.target=MRI DataLoader.shuffle=False Trainer.save_dir=${save_dir}/CT2MRI_shuffleF_seed1"
  "python main.py seed=213 Optim.lr=0.000001 Trainer.name=baseline DA.source=CT DA.target=MRI DataLoader.shuffle=False Trainer.save_dir=${save_dir}/CT2MRI_shuffleF_seed2"
  "python main.py seed=321 Optim.lr=0.000001 Trainer.name=baseline DA.source=CT DA.target=MRI DataLoader.shuffle=False Trainer.save_dir=${save_dir}/CT2MRI_shuffleF_seed3"


  "python main.py seed=123 Optim.lr=0.000001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=True Trainer.save_dir=${save_dir}/MRI2CT_shuffleT_seed1"
  "python main.py seed=213 Optim.lr=0.000001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=True Trainer.save_dir=${save_dir}/MRI2CT_shuffleT_seed2"
  "python main.py seed=321 Optim.lr=0.000001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=True Trainer.save_dir=${save_dir}/MRI2CT_shuffleT_seed3"

  "python main.py seed=123 Optim.lr=0.000001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=False Trainer.save_dir=${save_dir}/MRI2CT_shuffleF_seed1"
  "python main.py seed=213 Optim.lr=0.000001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=False Trainer.save_dir=${save_dir}/MRI2CT_shuffleF_seed2"
  "python main.py seed=321 Optim.lr=0.000001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=False Trainer.save_dir=${save_dir}/MRI2CT_shuffleF_seed3"

  "python main.py seed=123 Optim.lr=0.00001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=True Trainer.save_dir=${save_dir}/MRI2CT_401lr_shuffleT_seed1"
  "python main.py seed=213 Optim.lr=0.00001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=True Trainer.save_dir=${save_dir}/MRI2CT_401lr_shuffleT_seed2"
  "python main.py seed=321 Optim.lr=0.00001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=True Trainer.save_dir=${save_dir}/MRI2CT_401lr_shuffleT_seed3"

  "python main.py seed=123 Optim.lr=0.00001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=False Trainer.save_dir=${save_dir}/MRI2CT_401lr_shuffleF_seed1"
  "python main.py seed=213 Optim.lr=0.00001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=False Trainer.save_dir=${save_dir}/MRI2CT_401lr_shuffleF_seed2"
  "python main.py seed=321 Optim.lr=0.00001 Trainer.name=baseline DA.source=MRI DA.target=CT DataLoader.shuffle=False Trainer.save_dir=${save_dir}/MRI2CT_401lr_shuffleF_seed3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


