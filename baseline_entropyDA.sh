#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=rrg-ebrahimi
save_dir=baseline_entropyDA

declare -a StringArray=(
  "python main.py seed=123 Optim.lr=0.00001 Trainer.name=entda DA.source=CT DA.target=MRI  Scheduler.RegScheduler.max_value=0.1 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01reg"
  "python main.py seed=123 Optim.lr=0.00001 Trainer.name=entda DA.source=CT DA.target=MRI  Scheduler.RegScheduler.max_value=0.01 Trainer.save_dir=${save_dir}/CT2MRI_401lr_101reg"
  "python main.py seed=123 Optim.lr=0.00001 Trainer.name=entda DA.source=CT DA.target=MRI  Scheduler.RegScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_201reg"
  "python main.py seed=123 Optim.lr=0.00001 Trainer.name=entda DA.source=CT DA.target=MRI  Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_301reg"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


