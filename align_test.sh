#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=align_test_0215_R

declare -a StringArray=(
# lr = 0.000001     K =20;   Upconv2;  [0,0][3,3][9,9][13.13]
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=9  DA.displacement.map_y=9  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_100reg_upc2_9disp_20c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=13 DA.displacement.map_y=13 DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_100reg_upc2_13disp_20c"

"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0  DA.displacement.map_y=0  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=10000 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10000reg_upc2_0disp_20c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=3  DA.displacement.map_y=3  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=10000 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10000reg_upc2_3disp_20c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=9  DA.displacement.map_y=9  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=10000 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10000reg_upc2_9disp_20c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=13 DA.displacement.map_y=13 DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=10000 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10000reg_upc2_13disp_20c"

"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0  DA.displacement.map_y=0  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1reg_upc2_0disp_20c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=3  DA.displacement.map_y=3  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1reg_upc2_3disp_20c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=9  DA.displacement.map_y=9  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1reg_upc2_9disp_20c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=13 DA.displacement.map_y=13 DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1reg_upc2_13disp_20c"

# k=10
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0  DA.displacement.map_y=0  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_100reg_upc2_0disp_10c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=3  DA.displacement.map_y=3  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_100reg_upc2_3disp_10c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=9  DA.displacement.map_y=9  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_100reg_upc2_9disp_10c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=13 DA.displacement.map_y=13 DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_100reg_upc2_13disp_10c"

"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0  DA.displacement.map_y=0  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=10000 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10000reg_upc2_0disp_10c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=3  DA.displacement.map_y=3  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=10000 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10000reg_upc2_3disp_10c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=9  DA.displacement.map_y=9  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=10000 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10000reg_upc2_9disp_10c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=13 DA.displacement.map_y=13 DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=10000 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_10000reg_upc2_13disp_10c"

"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0  DA.displacement.map_y=0  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1reg_upc2_0disp_10c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=3  DA.displacement.map_y=3  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1reg_upc2_3disp_10c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=9  DA.displacement.map_y=9  DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1reg_upc2_9disp_10c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=13 DA.displacement.map_y=13 DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/CT2MRI_1reg_upc2_13disp_10c"


# lr = 0.000001 cluster_weight=0.01, 0.001
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0 DA.displacement.map_y=0 DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0.01 Trainer.save_dir=${save_dir}/CT2MRI_100reg_upc2_0disp_cluster101_20c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=3 DA.displacement.map_y=3 DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0.01 Trainer.save_dir=${save_dir}/CT2MRI_100reg_upc2_3disp_cluster101_20c"

"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=0 DA.displacement.map_y=0 DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_100reg_upc2_0disp_cluster201_20c"
"python main.py Optim.lr=0.000001 Train.name=align_IndividualBN DA.double_bn=True DA.displacement.map_x=3 DA.displacement.map_y=3 DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=100 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/CT2MRI_100reg_upc2_3disp_cluster201_20c"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


