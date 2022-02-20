#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=main_jobs_base

declare -a StringArray=(
# CT2MRI
# lr = 0.00001
# baseline
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=baseline Trainer.save_dir=${save_dir}/CT2MRI_lower_baseline_seed1"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/CT2MRI_upper_baseline_seed1"

# entropy DA
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.5 Trainer.save_dir=${save_dir}/CT2MRI_entDA_05reg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.1 Trainer.save_dir=${save_dir}/CT2MRI_entDA_01reg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.01 Trainer.save_dir=${save_dir}/CT2MRI_entDA_101reg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_entDA_201reg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/CT2MRI_entDA_301reg"

# sup + 0align + 0cluster
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_0Areg_0Creg_prediction"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


