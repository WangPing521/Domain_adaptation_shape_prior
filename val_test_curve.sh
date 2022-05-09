#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=splits_crossvalidation
declare -a StringArray=(
# align
"python main.py seed=123 Optim.lr=0.00001 Data.seed=12 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/fold1_run1"
"python main.py seed=231 Optim.lr=0.00001 Data.seed=12 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/fold1_run2"
"python main.py seed=321 Optim.lr=0.00001 Data.seed=12 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/fold1_run3"

"python main.py seed=123 Optim.lr=0.00001 Data.seed=123 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/fold2_run1"
"python main.py seed=231 Optim.lr=0.00001 Data.seed=123 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/fold2_run2"
"python main.py seed=321 Optim.lr=0.00001 Data.seed=123 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/fold2_run3"

"python main.py seed=123 Optim.lr=0.00001 Data.seed=13 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/fold3_run1"
"python main.py seed=231 Optim.lr=0.00001 Data.seed=13 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/fold3_run2"
"python main.py seed=321 Optim.lr=0.00001 Data.seed=13 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/fold3_run3"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
