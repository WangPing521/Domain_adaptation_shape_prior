#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=align_joint_prediction

declare -a StringArray=(
# CT2MRI
# lr = 0.000001

# sup+ cluster + align  Up_conv2
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=False  Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_201creg_deconv1"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_201creg_5c_0disp"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_201creg_10c_0disp"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_201creg_20c_0disp"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Deconv_1x1 DA.displacement=True Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_201creg_deconv1_disp"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_201creg_5c_disp"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_201creg_10c_disp"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_201creg_20c_disp"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


