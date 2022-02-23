#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=main_jobs_multi_MR2CT_resolution

declare -a StringArray=(
# CT2MRI
#------------------MR2CT
# batch_size = 21    DA.batchsize_indicator=6
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_mrs_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_mrs_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_mrs_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_mrs_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=5 DA.displacement=True Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_mrs_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=6 DA.displacement=True Scheduler.RegScheduler.max_value=1 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_mrs_disp_prediction"


)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


