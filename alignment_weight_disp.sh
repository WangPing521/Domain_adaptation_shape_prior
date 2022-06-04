#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=0603_alignment_weight_disp1

declare -a StringArray=(

#todo disp=0, output layer, resolution=256, ent+align
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/disp0_301Ent_305joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/disp0_301Ent_305joint_fold2"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/disp0_301Ent_201joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/disp0_301Ent_201joint_fold2"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/disp0_301Ent_205joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/disp0_301Ent_205joint_fold2"

#----
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405joint_fold2"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301joint_fold2"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_305joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_305joint_fold2"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_201joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_201joint_fold2"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_205joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_205joint_fold2"

#--
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_401joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_401joint_fold2"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_405joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_405joint_fold2"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_301joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_301joint_fold2"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_305joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_305joint_fold2"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_201joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_201joint_fold2"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_205joint_fold1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_205joint_fold2"

#todo disp=1, output layer, resolution=256, ent+align
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301joint"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301joint"
)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
