#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=0404_ent_psda

declare -a StringArray=(
#------------------MR2CT
#"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=9 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline63_seed1"
#"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=9 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=9 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline63_seed3"


#"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=9 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63_seed1"
#"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=9 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=9 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63_seed3"

# entropy
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_701reg"
#"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_701reg_run2"
#"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_701reg_run3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_801reg"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_801reg_run2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_801reg_run3"

# PLDA # lowerbaseline and ent
#"python pseudoDA_main.py seed=123 Optim.lr=0.000001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_baseline"
"python pseudoDA_main.py seed=123 Optim.lr=0.000001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_entda"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


