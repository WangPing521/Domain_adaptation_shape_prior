#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=nosidplacement

declare -a StringArray=(
# CT2MRI
#------------------MR2CT
# batch_size = 21    DA.batchsize_indicator=6
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r1_0disp_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r1_0disp_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_r1_0disp_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_0disp_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_0disp_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_0disp_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_0disp_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_0disp_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_0disp_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/MRI2CT_501Areg_r1_0disp_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/MRI2CT_501Areg_r1_0disp_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/MRI2CT_501Areg_r1_0disp_seed3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


