#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=def-chdesa
save_dir=mmwhs_1115_entWeight
declare -a StringArray=(
#todo bothlayer
# ent weight: [0, 0.000001, 0.00001, 0.00003, 0.00005, 0.0001]
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/200bothlayer_0Ent_407joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/200bothlayer_0Ent_407joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/200bothlayer_0Ent_407joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/200bothlayer_501Ent_407joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/200bothlayer_501Ent_407joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/200bothlayer_501Ent_407joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/200bothlayer_401Ent_407joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/200bothlayer_401Ent_407joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/200bothlayer_401Ent_407joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.00003 Trainer.save_dir=${save_dir}/200bothlayer_403Ent_407joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.00003 Trainer.save_dir=${save_dir}/200bothlayer_403Ent_407joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.00003 Trainer.save_dir=${save_dir}/200bothlayer_403Ent_407joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/200bothlayer_405Ent_407joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/200bothlayer_405Ent_407joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/200bothlayer_405Ent_407joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/200bothlayer_301Ent_407joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/200bothlayer_301Ent_407joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00007 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/200bothlayer_301Ent_407joint_seed3"

)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
