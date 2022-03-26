#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=0326_prostate_baseline
declare -a StringArray=(

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=6 Trainer.name=baseline Trainer.save_dir=${save_dir}/mise2state_lower_baseline21_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=15 Trainer.name=baseline Trainer.save_dir=${save_dir}/mise2state_lower_baseline48_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=20 Trainer.name=baseline Trainer.save_dir=${save_dir}/mise2state_lower_baseline63_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=25 Trainer.name=baseline Trainer.save_dir=${save_dir}/mise2state_lower_baseline78_seed1"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=6 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/mise2state_upper_baseline21_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=15 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/mise2state_upper_baseline48_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=20 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/mise2state_upper_baseline63_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=25 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/mise2state_upper_baseline78_seed1"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=6 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline21_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=15 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline48_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=20 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline63_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=25 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline78_seed1"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=6 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline21_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=15 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline48_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=20 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline63_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=25 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline78_seed1"



)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
