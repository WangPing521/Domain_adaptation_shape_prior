#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=0626

declare -a StringArray=(
#todo output layer, displacement or not
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_405joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_405joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=13 Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp13_401Ent_405joint_seed1"

#--
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_301joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=13 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp13_401Ent_301joint_seed1"

#--
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp0_405Ent_301joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp1_405Ent_301joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp3_405Ent_301joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=13 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp13_405Ent_301joint_seed1"


#todo upconv2, projectors->clusters VS features alignmet,  displacement or not
# projectors with clusters
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.clusters=5 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k5_401Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.clusters=8 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k8_401Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.clusters=10 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k10_401Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.clusters=15 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k15_401Ent_205joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.clusters=5 DA.displace_scale=1 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k5_disp1_401Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.clusters=8 DA.displace_scale=1 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k8_disp1_401Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.clusters=10 DA.displace_scale=1 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k10_disp1_401Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.clusters=15 DA.displace_scale=1 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k15_disp1_401Ent_205joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.clusters=5 DA.displace_scale=5 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k5_disp5_401Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.clusters=8 DA.displace_scale=5 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k8_disp5_401Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.clusters=10 DA.displace_scale=5 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k10_disp5_401Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.clusters=15 DA.displace_scale=5 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k15_disp5_401Ent_205joint_seed1"

# feature alignments, displacements or not
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_cc0_501Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_cc1_501Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=5 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_cc5_501Ent_205joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_cc0_401Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_cc1_401Ent_205joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=5 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_cc5_401Ent_205joint_seed1"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
