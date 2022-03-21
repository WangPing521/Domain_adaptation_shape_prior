#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=0317_baseline_BS
declare -a StringArray=(

# 1:3 3:12 6:21 15:48 20:63 25:78 30:93 40:123
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=1  Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline3_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=3  Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline12_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=6  Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline21_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=15 Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline48_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=20 Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline63_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=25 Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline78_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=30 Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline93_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=40 Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline123_seed1"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=1  Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline3_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=3  Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline12_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=6  Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline21_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=15 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline48_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=20 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=25 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline78_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=30 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline93_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=40 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline123_seed1"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
