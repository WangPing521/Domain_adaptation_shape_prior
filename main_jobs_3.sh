#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=main_jobs_clusterweight

declare -a StringArray=(
# CT2MRI
# lr = 0.00001
# Deconv_1x1, Creg=0.1 0.01 0.001 0.0001
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.1    Trainer.save_dir=${save_dir}/CT2MRI_01Creg_prediction"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.01   Trainer.save_dir=${save_dir}/CT2MRI_101Creg_prediction"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001  Trainer.save_dir=${save_dir}/CT2MRI_201Creg_prediction"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/CT2MRI_301Creg_prediction"

# Up_conv2, K=5, 10, 20, 40, Creg=0.1 0.01 0.001 0.0001
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.1    Trainer.save_dir=${save_dir}/CT2MRI__01Creg_upconv2_5"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.01   Trainer.save_dir=${save_dir}/CT2MRI__101Creg_upconv2_5"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001  Trainer.save_dir=${save_dir}/CT2MRI__201Creg_upconv2_5"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/CT2MRI__301Creg_upconv2_5"

# 10
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.1    Trainer.save_dir=${save_dir}/CT2MRI_01Creg__upconv2_10"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.01   Trainer.save_dir=${save_dir}/CT2MRI_101Creg_upconv2_10"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001  Trainer.save_dir=${save_dir}/CT2MRI_201Creg_upconv2_10"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/CT2MRI_301Creg_upconv2_10"

# 20
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.1    Trainer.save_dir=${save_dir}/CT2MRI_01Creg_upconv2_20"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.01   Trainer.save_dir=${save_dir}/CT2MRI_101Creg_upconv2_20"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001  Trainer.save_dir=${save_dir}/CT2MRI_201Creg_upconv2_20"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/CT2MRI_301Creg_upconv2_20"

# 40
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.1    Trainer.save_dir=${save_dir}/CT2MRI_01Creg_upconv2_40"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.01   Trainer.save_dir=${save_dir}/CT2MRI_101Creg_upconv2_40"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001  Trainer.save_dir=${save_dir}/CT2MRI_201Creg_upconv2_40"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/CT2MRI_301Creg_upconv2_40"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


