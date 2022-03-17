#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="../CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=0317_baseline_BS
declare -a StringArray=(

"python ../main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=6  Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline21_seed1"
"python ../main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=10 Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline33_seed1"
"python ../main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=20 Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline63_seed1"
"python ../main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=40 Trainer.name=baseline Trainer.save_dir=${save_dir}/MRI2CT_lower_baseline123_seed1"

"python ../main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=6  Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline21_seed1"
"python ../main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=10 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline33_seed1"
"python ../main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=20 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline63_seed1"
"python ../main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=MRI DA.target=CT DA.batchsize_indicator=40 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/MRI2CT_upper_baseline123_seed1"

#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=20 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_63bs_0Ent_301MAEreg_seed1"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=20 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1_63bs_501Ent_401MAEreg_seed1"
#"python main.py seed=231 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=20 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp1_63bs_501Ent_401MAEreg_seed2"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
