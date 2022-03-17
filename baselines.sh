#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=check_entDA

declare -a StringArray=(
# CT2MRI
#------------------MR2CT
#"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline21_seed1"
#"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline21_seed2"
#"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline21_seed3"

#"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline21_seed1"
#"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline21_seed2"
#"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.batchsize_indicator=6 DA.target=CT Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline21_seed3"

# entropy
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_601reg"
#"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.checkpoint_path=runs/${save_dir}/MRI2CT_entDA_601reg_run2 Trainer.save_dir=${save_dir}/MRI2CT_entDA_601reg_run2"
#"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.checkpoint_path=runs/${save_dir}/MRI2CT_entDA_601reg_run3 Trainer.save_dir=${save_dir}/MRI2CT_entDA_601reg_run3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_601reg"
#"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.checkpoint_path=runs/${save_dir}/MRI2CT_entDA_701reg_run2 Trainer.save_dir=${save_dir}/MRI2CT_entDA_701reg_run2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_801reg_run3_1"

## sup + 0align + 0cluster
##"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=False DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_1bn_prediction"
#"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=False DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_1bn_prediction_run2"
#"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=False DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_1bn_prediction_run3"
#
##"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_2bn_prediction"
#"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_2bn_prediction_run2"
#"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_2bn_prediction_run3"
#
##"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=5 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_2bn_upconv2_5c"
#"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=5 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_2bn_upconv2_5c_run2"
#"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=5 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_2bn_upconv2_5c_run3"
#
##"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_2bn_upconv2_20c"
#"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_2bn_upconv2_20c_run2"
#"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_0Areg_0Creg_2bn_upconv2_20c_run3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


