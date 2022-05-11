#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=rebuttal_CT2MRI_priorent
declare -a StringArray=(
# baselines
#"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=False DA.source=CT DA.target=MRI DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/CT2MRI_lower_baseline63_seed1"
#"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=False DA.source=CT DA.target=MRI DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/CT2MRI_lower_baseline63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=False DA.source=CT DA.target=MRI DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/CT2MRI_lower_baseline63_seed3"

# priorbased
"python main.py seed=123 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=CT DA.target=MRI DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior63_401Ent_401prior_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=CT DA.target=MRI DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior63_401Ent_401prior_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.double_bn=True Trainer.name=priorbased DA.source=CT DA.target=MRI DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior63_401Ent_401prior_seed3"

# align
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=CT DA.target=MRI DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301regjoint63_seed1"
#"python main.py seed=231 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=CT DA.target=MRI DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301regjoint63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=CT DA.target=MRI DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301regjoint63_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=CT DA.target=MRI DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301regjoint63_seed1"
#"python main.py seed=231 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=CT DA.target=MRI DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301regjoint63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=CT DA.target=MRI DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301regjoint63_seed3"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
