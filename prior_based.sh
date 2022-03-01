#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=priorbased

declare -a StringArray=(

#------------------MR2CT
#cluster: prior
#echeduler: ent
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_301Ent_501prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_301Ent_501prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_301Ent_501prior3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/prior_301Ent_301prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/prior_301Ent_301prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/prior_301Ent_301prior3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.01 Trainer.save_dir=${save_dir}/prior_301Ent_101prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.01 Trainer.save_dir=${save_dir}/prior_301Ent_101prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.01 Trainer.save_dir=${save_dir}/prior_301Ent_101prior3"


"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_401Ent_501prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_401Ent_501prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_401Ent_501prior3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/prior_401Ent_301prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/prior_401Ent_301prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/prior_401Ent_301prior3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/prior_401Ent_101prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/prior_401Ent_101prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/prior_401Ent_101prior3"


"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_501prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_501prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_501prior3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0001    Trainer.save_dir=${save_dir}/prior_501Ent_301prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0001    Trainer.save_dir=${save_dir}/prior_501Ent_301prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0001    Trainer.save_dir=${save_dir}/prior_501Ent_301prior3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.01    Trainer.save_dir=${save_dir}/prior_501Ent_101prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.01    Trainer.save_dir=${save_dir}/prior_501Ent_101prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.01    Trainer.save_dir=${save_dir}/prior_501Ent_101prior3"


"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/prior_601Ent_501prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/prior_601Ent_501prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/prior_601Ent_501prior3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/prior_601Ent_301prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/prior_601Ent_301prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/prior_601Ent_301prior3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.01   Trainer.save_dir=${save_dir}/prior_601Ent_101prior"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.01   Trainer.save_dir=${save_dir}/prior_601Ent_101prior2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.01   Trainer.save_dir=${save_dir}/prior_601Ent_101prior3"



)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


