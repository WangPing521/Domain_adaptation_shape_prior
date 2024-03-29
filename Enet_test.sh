#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=ENET_0822
declare -a StringArray=(
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed3"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed3"

# entprior
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_405prior_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_405prior_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_405prior_seed3"

#entda
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000005 Trainer.save_dir=${save_dir}/entDA63_705reg_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000005 Trainer.save_dir=${save_dir}/entDA63_705reg_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000005 Trainer.save_dir=${save_dir}/entDA63_705reg_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000005 Trainer.save_dir=${save_dir}/entDA63_805reg_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000005 Trainer.save_dir=${save_dir}/entDA63_805reg_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000005 Trainer.save_dir=${save_dir}/entDA63_805reg_seed3"

##todo: displacement(0,1,3)
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp0_505Ent_305joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp0_505Ent_305joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp0_505Ent_305joint_seed3"

#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/disp0_605Ent_305joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/disp0_605Ent_305joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/disp0_605Ent_305joint_seed3"


# todo: disp[1,3]
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_305joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_305joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_305joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0003 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp3_505Ent_303joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0003 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp3_505Ent_303joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0003 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp3_505Ent_303joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp3_505Ent_305joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp3_505Ent_305joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp3_505Ent_305joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0007 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp3_505Ent_307joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0007 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp3_505Ent_307joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0007 Scheduler.ClusterScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/disp3_505Ent_307joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0003 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_303joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0003 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_303joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0003 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_303joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_305joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_305joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_305joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0007 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_307joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0007 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_307joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=ent_our_trainer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=[1,3] Scheduler.RegScheduler.max_value=0.0007 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_307joint_seed3"

)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
