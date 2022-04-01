#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=rrg-ebrahimi
save_dir=0331_priorbased_BS

declare -a StringArray=(

#------------------MR2CT
#bs=7:1 21:3 35:5 42:6 56:8 63:9

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=3 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior21_401Ent_401prior"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=3 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior21_401Ent_401prior_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=3 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior21_401Ent_401prior_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=5 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior35_401Ent_401prior"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=5 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior35_401Ent_401prior_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=5 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior35_401Ent_401prior_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior42_401Ent_401prior"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior42_401Ent_401prior_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior42_401Ent_401prior_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=8 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior56_401Ent_401prior"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=8 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior56_401Ent_401prior_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=8 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior56_401Ent_401prior_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior63_401Ent_401prior"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior63_401Ent_401prior_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=MRI DA.target=CT DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior63_401Ent_401prior_seed3"


)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


