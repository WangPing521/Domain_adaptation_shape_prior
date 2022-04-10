#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=0407_bothlayers
declare -a StringArray=(
# upconv2 projector_clusters:5, 10, 20
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.weight1=1  DA.multi_scale=2  DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_1weight_c20"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.weight1=10 DA.multi_scale=2  DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_10weight_c20"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.weight1=20 DA.multi_scale=2  DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_20weight_c20"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.weight1=30 DA.multi_scale=2  DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_30weight_c20"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.weight1=50 DA.multi_scale=2  DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_50weight_c20"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.weight1=100 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_100weight_c20"

# upconv2 cross correlations
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.weight1=1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_1weight_cc"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.weight1=1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_1weight_cc_seed2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.weight1=1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_1weight_cc_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.weight1=10 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_10weight_cc"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.weight1=20 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_20weight_cc"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.weight1=20 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_20weight_cc_seed2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.weight1=20 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_20weight_cc_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.weight1=30 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_30weight_cc"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.weight1=50 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501ent_301joint_50weight_cc"


#

)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
