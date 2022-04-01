#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=0331_CT2MRI_recheck
declare -a StringArray=(
# batch_align: bs=21:3 35:5 42:6 56:8 63:9
# baselines
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=CT DA.target=MRI DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/CT2MRI_lower_baseline63"
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=CT DA.target=MRI DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/CT2MRI_upper_baseline63"

# priorbased
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=CT DA.target=MRI DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior63_401Ent_401prior"

# align
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=CT DA.target=MRI DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301regjoint63"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=CT DA.target=MRI DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301MAEregjoint63"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=CT DA.target=MRI DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_301MAEregjoint63"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
