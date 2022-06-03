#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=0602_baselines

declare -a StringArray=(

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=1 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=1 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=1 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=1 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=1 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=1 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed3"


#"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.batchsize_indicator=9 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63_seed1"
#"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.batchsize_indicator=9 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.batchsize_indicator=9 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63_seed3"

# entropy
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_701reg"
#"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_701reg_run2"
#"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_701reg_run3"

# PLDA # lowerbaseline and ent
#"python pseudoDA_main.py seed=123 Optim.lr=0.000001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_baseline"
"python pseudoDA_main.py seed=123 Optim.lr=0.000001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_entda_seed1"
"python pseudoDA_main.py seed=231 Optim.lr=0.000001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_entda_seed2"
"python pseudoDA_main.py seed=321 Optim.lr=0.000001 DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_entda_seed3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


