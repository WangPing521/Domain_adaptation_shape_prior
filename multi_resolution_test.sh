#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=MR2CT_resolutions_21bs

declare -a StringArray=(
# CT2MRI
#------------------MR2CT
# batch_size = 21    DA.batchsize_indicator=6
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r1_0disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r1_disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=6 DA.displacement=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r6_disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=5 DA.displacement=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r5_disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r4_disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r3_disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r2_disp_prediction"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_0disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=6 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r6_disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=5 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r5_disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r3_disp_prediction"
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_prediction"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=6 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r6_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=5 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r5_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r4_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r3_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r2_disp_prediction"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


