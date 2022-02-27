#!/usr/bin/env bash

#set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=rrg-ebrahimi
save_dir=olva

declare -a StringArray=(
# CT2MRI
# lr = 0.00001
#------------------MR2CT
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=10 Scheduler.ClusterScheduler.max_value=10      Trainer.save_dir=${save_dir}/OLVA_10kl_10ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=10 Scheduler.ClusterScheduler.max_value=0.001   Trainer.save_dir=${save_dir}/OLVA_10kl_201ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=10 Scheduler.ClusterScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/OLVA_10kl_301ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=10 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/OLVA_10kl_401ot"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=10      Trainer.save_dir=${save_dir}/OLVA_1kl_10ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0.001   Trainer.save_dir=${save_dir}/OLVA_1kl_201ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/OLVA_1kl_301ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/OLVA_1kl_401ot"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=10      Trainer.save_dir=${save_dir}/OLVA_201kl_10ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.001   Trainer.save_dir=${save_dir}/OLVA_201kl_201ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/OLVA_201kl_301ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/OLVA_201kl_401ot"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=10      Trainer.save_dir=${save_dir}/OLVA_401kl_10ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.001   Trainer.save_dir=${save_dir}/OLVA_401kl_201ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/OLVA_401kl_301ot"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/OLVA_401kl_401ot"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


