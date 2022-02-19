#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=align_joint_upconv2

declare -a StringArray=(
# CT2MRI
# lr = 0.000001

# Deconv_1x1, K =5;
# sup+cluster_loss
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=1     Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_1creg_5c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.1   Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_01creg_5c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_101creg_5c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_201creg_5c"
# 10
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=1     Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_1creg_10c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.1   Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_01creg_10c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_101creg_10c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_201creg_10c"

# 20
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=1     Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_1creg_20c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.1   Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_01creg_20c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_101creg_20c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_401lr_0areg_201creg_20c"


# sup+align_loss
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_100areg_0creg_0disp_5c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=10  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_10areg_0creg_0disp_5c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=1   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_1areg_0creg_0disp_5c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_0creg_0disp_5c"

"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_100areg_0creg_disp_5c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=10  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_10areg_0creg_disp_5c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=1   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_1areg_0creg_disp_5c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_0creg_disp_5c"

# 10
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_100areg_0creg_0disp_10c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=10  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_10areg_0creg_0disp_10c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=1   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_1areg_0creg_0disp_10c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_0creg_0disp_10c"

"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_100areg_0creg_disp_10c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=10  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_10areg_0creg_disp_10c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=1   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_1areg_0creg_disp_10c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_0creg_disp_10c"

# 20
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_100areg_0creg_0disp_20c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=10  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_10areg_0creg_0disp_20c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=1   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_1areg_0creg_0disp_20c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_0creg_0disp_20c"

"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_100areg_0creg_disp_20c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=10  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_10areg_0creg_disp_20c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=1   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_1areg_0creg_disp_20c"
"python main.py seed=123 Optim.lr=0.00001 Train.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_401lr_01areg_0creg_disp_20c"

# sup+ cluster + align

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


