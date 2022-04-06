#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=0404_bothlayers
declare -a StringArray=(
# upconv2 projector_clusters:5, 8, 10

# upconv2 projector_clusters:5, 10, 20 ,30
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=5  DA.weight1=0.5 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_305joint_05weight_c5"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=8  DA.weight1=0.5 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_305joint_05weight_c8"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 DA.weight1=0.5 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_305joint_05weight_c10"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.weight1=0.5 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_305joint_05weight_c20"

# upconv2 cross correlations
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.weight1=0.5 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp0_601ent_305joint_05weight_cc"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.weight1=0.5 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/disp0_701ent_305joint_05weight_cc"


#

)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
