#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=0721_prostate_our
declare -a StringArray=(
# disp1
#---
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1_501Ent_601joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1_501Ent_601joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1_501Ent_601joint_seed3"

#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1_501Ent_705joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1_501Ent_705joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1_501Ent_705joint_seed3"

# upconv2 + output
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_501Ent_501joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_501Ent_605joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_501Ent_601joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_501Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_100bothlayer_501Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_100bothlayer_501Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_100bothlayer_501Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_100bothlayer_501Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_10bothlayer_501Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_10bothlayer_501Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_10bothlayer_501Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_10bothlayer_501Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_1bothlayer_501Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_1bothlayer_501Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_1bothlayer_501Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r3_1bothlayer_501Ent_705joint_seed1"
####
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/disp1r1_200bothlayer_501Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/disp1r1_200bothlayer_501Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/disp1r1_200bothlayer_501Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r1_200bothlayer_501Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/disp1r1_100bothlayer_501Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/disp1r1_100bothlayer_501Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/disp1r1_100bothlayer_501Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r1_100bothlayer_501Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/disp1r1_10bothlayer_501Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/disp1r1_10bothlayer_501Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/disp1r1_10bothlayer_501Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r1_10bothlayer_501Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/disp1r1_1bothlayer_501Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/disp1r1_1bothlayer_501Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/disp1r1_1bothlayer_501Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1r1_1bothlayer_501Ent_705joint_seed1"

#============
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_601Ent_501joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_601Ent_605joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_601Ent_601joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_601Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp1r3_100bothlayer_601Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r3_100bothlayer_601Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r3_100bothlayer_601Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp1r3_100bothlayer_601Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp1r3_10bothlayer_601Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r3_10bothlayer_601Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r3_10bothlayer_601Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp1r3_10bothlayer_601Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp1r3_1bothlayer_601Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r3_1bothlayer_601Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r3_1bothlayer_601Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp1r3_1bothlayer_601Ent_705joint_seed1"

####
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp1r1_200bothlayer_601Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r1_200bothlayer_601Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r1_200bothlayer_601Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp1r1_200bothlayer_601Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp1r1_100bothlayer_601Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r1_100bothlayer_601Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r1_100bothlayer_601Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=100 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp1r1_100bothlayer_601Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp1r1_10bothlayer_601Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r1_10bothlayer_601Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r1_10bothlayer_601Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=10 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp1r1_10bothlayer_601Ent_705joint_seed1"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp1r1_1bothlayer_601Ent_501joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r1_1bothlayer_601Ent_605joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/disp1r1_1bothlayer_601Ent_601joint_seed1"
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=1 Scheduler.RegScheduler.max_value=0.00000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp1r1_1bothlayer_601Ent_705joint_seed1"


)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
