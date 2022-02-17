#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=align_test_0215_R

declare -a StringArray=(
# CT2MRI
# lr = 0.000001

# Deconv_1x1, K =5;
# sup+cluster_loss
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0 DA.displacement.map_y=0 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=1     Trainer.save_dir=${save_dir}/CT2MRI_0areg_1creg"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0 DA.displacement.map_y=0 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.1   Trainer.save_dir=${save_dir}/CT2MRI_0areg_01creg"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0 DA.displacement.map_y=0 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.01  Trainer.save_dir=${save_dir}/CT2MRI_0areg_101creg"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0 DA.displacement.map_y=0 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_0areg_201creg"

# sup+align_loss
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0 DA.displacement.map_y=0 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_100areg_0creg_0disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=1 DA.displacement.map_y=1 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_100areg_0creg_1disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=3 DA.displacement.map_y=3 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_100areg_0creg_3disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=5 DA.displacement.map_y=5 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_100areg_0creg_5disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=7 DA.displacement.map_y=7 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_100areg_0creg_7disp"

"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0 DA.displacement.map_y=0 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=10 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10areg_0creg_0disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=1 DA.displacement.map_y=1 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=10 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10areg_0creg_1disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=3 DA.displacement.map_y=3 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=10 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10areg_0creg_3disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=5 DA.displacement.map_y=5 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=10 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10areg_0creg_5disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=7 DA.displacement.map_y=7 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=10 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10areg_0creg_7disp"

"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0 DA.displacement.map_y=0 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1areg_0creg_0disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=1 DA.displacement.map_y=1 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1areg_0creg_1disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=3 DA.displacement.map_y=3 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1areg_0creg_3disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=5 DA.displacement.map_y=5 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1areg_0creg_5disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=7 DA.displacement.map_y=7 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1areg_0creg_7disp"

"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0 DA.displacement.map_y=0 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_01areg_0creg_0disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=1 DA.displacement.map_y=1 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_01areg_0creg_1disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=3 DA.displacement.map_y=3 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_01areg_0creg_3disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=5 DA.displacement.map_y=5 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_01areg_0creg_5disp"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=7 DA.displacement.map_y=7 DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0.1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_01areg_0creg_7disp"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


