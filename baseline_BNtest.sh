#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=rrg-ebrahimi
save_dir=main_jobs_base_MR2CT_BN

declare -a StringArray=(
# CT2MRI
#------------------MR2CT
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=1 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline3_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=1 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline3_seed1"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline21_seed1"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline21_seed1"


)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


