#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=def-chdesa
save_dir=mmwhs_1119_poolingsize3_upcv2
declare -a StringArray=(
#todo: output layer pooling_size=3
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.001  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r2_401Ent_201joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.001  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r2_401Ent_201joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.001  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r2_401Ent_201joint_seed3"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.0007  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r2_401Ent_307joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.0007  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r2_401Ent_307joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.0007  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r2_401Ent_307joint_seed3"


#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.00007  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r3_401Ent_407joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.00007  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r3_401Ent_407joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.00007  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r3_401Ent_407joint_seed3"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.0001  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r3_401Ent_301joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.0001  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r3_401Ent_301joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] Scheduler.RegScheduler.max_value=0.0001  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/output_r3_401Ent_301joint_seed3"




# todo: Upconv2 layer pooling_size=3
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.01  Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr2_503Ent_101joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.01  Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr2_503Ent_101joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.01  Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr2_503Ent_101joint_seed3"

#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr2_503Ent_205joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr2_503Ent_205joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr2_503Ent_205joint_seed3"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.01  Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr3_503Ent_101joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.01  Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr3_503Ent_101joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.01  Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr3_503Ent_101joint_seed3"

#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr3_503Ent_205joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr3_503Ent_205joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.pool_size=3 DA.displacement=True DA.displace_scale=[1] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccr3_503Ent_205joint_seed3"

)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
