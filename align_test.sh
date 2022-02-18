#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=align_joint

declare -a StringArray=(
# CT2MRI
# lr = 0.000001

# Deconv_1x1, K =5;
# sup+cluster_loss
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=1     Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_1creg"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.1   Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_01creg"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_101creg"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_201creg"

"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=1     Trainer.save_dir=${save_dir}/CT2MRI_501lr_0areg_1creg"
"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.1   Trainer.save_dir=${save_dir}/CT2MRI_501lr_0areg_01creg"
"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/CT2MRI_501lr_0areg_101creg"
"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_501lr_0areg_201creg"

# sup+align_loss
"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=False Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_501lr_100areg_0creg_0disp"
"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=False Scheduler.RegScheduler.max_value=10  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_501lr_10areg_0creg_0disp"
"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=False Scheduler.RegScheduler.max_value=1   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_501lr_1areg_0creg_0disp"
"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=False Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_501lr_01areg_0creg_0disp"

"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=True Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_501lr_100areg_0creg_disp"
"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=True Scheduler.RegScheduler.max_value=10  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_501lr_10areg_0creg_disp"
"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=True Scheduler.RegScheduler.max_value=1   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_501lr_1areg_0creg_disp"
"python main.py seed=123 Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=True Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_501lr_01areg_0creg_disp"

# from different layers


# sup+ cluster + align

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


