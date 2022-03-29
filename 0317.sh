#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=0329_baselines
declare -a StringArray=(
# batch_align: bs=7:1 21:3 35:5 42:6 56:8 63:9 98:14

"python main.py seed=123 Optim.lr=0.00001  DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=1  Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline7"
"python main.py seed=123 Optim.lr=0.00001  DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=3  Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline21"
"python main.py seed=123 Optim.lr=0.00001  DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=5  Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline35"
"python main.py seed=123 Optim.lr=0.00001  DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=6  Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline42"
"python main.py seed=123 Optim.lr=0.00001  DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=8  Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline56"
"python main.py seed=123 Optim.lr=0.00001  DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=9  Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline63"
"python main.py seed=123 Optim.lr=0.00001  DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=14 Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline98"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=1  Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline7"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=3  Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline21"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=5  Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline35"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=6  Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline42"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=8  Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline56"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=9  Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=14 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline98"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
