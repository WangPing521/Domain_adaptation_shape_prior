#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=0812_prostate_our
declare -a StringArray=(
#todo output
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501Ent_601joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501Ent_601joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/501Ent_601joint_seed3"

#todo upconv2
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_501Ent_305joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_501Ent_305joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_501Ent_305joint_seed3"

#todo resolutions
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r2_501Ent_601joint_seed1"


#todo bothlayer
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10  Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_10bothlayer_501Ent_601joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_100bothlayer_501Ent_601joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_501Ent_601joint_seed1"

)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
