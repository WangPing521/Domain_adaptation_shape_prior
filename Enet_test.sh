#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=enet_test
declare -a StringArray=(
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed3"
#
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed3"
#

##todo: displacement(0,1,3,5,13)
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp3_405Ent_301joint_seed1"


"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_405prior_seed1"

)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
