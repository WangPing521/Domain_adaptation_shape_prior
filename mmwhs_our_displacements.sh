#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=mmwhs_upconv2_Newdisp5_0811
declare -a StringArray=(
# \Delta_{0,1,3}
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00003 Trainer.save_dir=${save_dir}/dispNew3_403Ent_405joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00003 Trainer.save_dir=${save_dir}/dispNew3_403Ent_405joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00003 Trainer.save_dir=${save_dir}/dispNew3_403Ent_405joint_seed3"

#######
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3,5] Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00003 Trainer.save_dir=${save_dir}/dispNew5_403Ent_301joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3,5] Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00003 Trainer.save_dir=${save_dir}/dispNew5_403Ent_301joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3,5] Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00003 Trainer.save_dir=${save_dir}/dispNew5_403Ent_301joint_seed3"

## todo: upconv2 displacement(cross-correlation)
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_503Ent_205joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_503Ent_205joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_503Ent_205joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.003 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_503Ent_203joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.003 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_503Ent_203joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.003 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_503Ent_203joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.007 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_503Ent_207joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.007 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_503Ent_207joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.007 Scheduler.ClusterScheduler.max_value=0.000003 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_503Ent_207joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.007 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_501Ent_207joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.007 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_501Ent_207joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.007 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_501Ent_207joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.003 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_501Ent_203joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.003 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_501Ent_203joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.003 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdispNew3_501Ent_203joint_seed3"


#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3,5] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdispNew5_501Ent_201joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3,5] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdispNew5_501Ent_201joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3,5] DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdispNew5_501Ent_201joint_seed3"

)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
