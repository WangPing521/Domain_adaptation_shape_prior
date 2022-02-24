#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=check_both_layers

declare -a StringArray=(
# CT2MRI
# lr = 0.00001
#------------------MR2CT
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r2_disp_prediction"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_upconv2"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_upconv2"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r2_disp_upconv2"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_upconv2_20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_upconv2_20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r2_disp_upconv2_20c"



"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Creg_r1_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Creg_r1_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/MRI2CT_201Creg_r2_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=5 DA.displacement=True Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Creg_r5_disp_prediction"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg301Creg_r1_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg301Creg_r1_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_201Areg301Creg_r2_disp_prediction"


"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight1=0.5 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight1=0.5 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight1=0.5 Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r2_disp_both05"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight1=0.1 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight1=0.1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight1=0.1 Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r2_disp_both01"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight1=0.5 DA.align_layer.clusters=20  Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_both05_up20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight1=0.5 DA.align_layer.clusters=20  Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_both05_up20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight1=0.5 DA.align_layer.clusters=20  Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r2_disp_both05_up20c"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight1=0.1 DA.align_layer.clusters=20  Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_both01_up20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight1=0.1 DA.align_layer.clusters=20  Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_both01_up20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight1=0.1 DA.align_layer.clusters=20  Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r2_disp_both01_up20c"


)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


