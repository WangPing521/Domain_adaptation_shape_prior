#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=0607_EntPrior

declare -a StringArray=(
# based on the previous experiments, the hyperparameters is determined
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_401prior_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_401prior_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_401prior_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_405prior_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_405prior_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_405prior_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_505prior_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_505prior_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_505prior_seed3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


