#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=priorbased_BS

declare -a StringArray=(

#------------------MR2CT
#cluster: prior
#echeduler: ent

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior21_401Ent_401prior"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=15 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior48_401Ent_401prior"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=20 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior63_401Ent_401prior"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=15 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior78_401Ent_401prior"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=30 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior93_401Ent_401prior"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=40 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior123_401Ent_401prior"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


