#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=0325_baseline

declare -a StringArray=(
#------------------MR2CT
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline21_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline21_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline21_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=15 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline48_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=15 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline48_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=15 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline48_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=20 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline63_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=20 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline63_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=20 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline63_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=25 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline78_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=25 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline78_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=25 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline78_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=30 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline93_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=30 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline93_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=30 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline93_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=40 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline123_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=40 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline123_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=40 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline123_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline21_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline21_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline21_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=15 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline48_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=15 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline48_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=15 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline48_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=20 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=20 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=20 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=25 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline78_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=25 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline78_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=25 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline78_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=30 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline93_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=30 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline93_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=30 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline93_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=40 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline123_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=40 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline123_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=40 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline123_seed3"

# entropy
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_601reg"
#"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.checkpoint_path=runs/${save_dir}/MRI2CT_entDA_601reg_run2 Trainer.save_dir=${save_dir}/MRI2CT_entDA_601reg_run2"
#"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.checkpoint_path=runs/${save_dir}/MRI2CT_entDA_601reg_run3 Trainer.save_dir=${save_dir}/MRI2CT_entDA_601reg_run3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_601reg"
#"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.checkpoint_path=runs/${save_dir}/MRI2CT_entDA_701reg_run2 Trainer.save_dir=${save_dir}/MRI2CT_entDA_701reg_run2"
#"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_801reg_run3_1"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


