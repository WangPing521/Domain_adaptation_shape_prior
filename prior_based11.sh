#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=rrg-ebrahimi
save_dir=priorbased333

declare -a StringArray=(

#------------------MR2CT
#cluster: prior
#echeduler: ent

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_401Ent_401prior"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_401Ent_401prior2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_401Ent_401prior3"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/prior_401Ent_301prior"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/prior_401Ent_301prior2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/prior_401Ent_301prior3"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.001  Trainer.save_dir=${save_dir}/prior_401Ent_201prior"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.001  Trainer.save_dir=${save_dir}/prior_401Ent_201prior2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.001  Trainer.save_dir=${save_dir}/prior_401Ent_201prior3"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/prior_401Ent_101prior"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/prior_401Ent_101prior2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/prior_401Ent_101prior3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


