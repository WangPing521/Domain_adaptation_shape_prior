#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=test_33BS_disp0
declare -a StringArray=(
# prior based
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_401Ent_401prior"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_33BS_401Ent_401prior"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=20 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_63BS_401Ent_401prior"

# disp=0 large batch size
# disp=0 align: mae
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=20 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_63bs_401MAEreg"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_33bs_401Ent_301MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_33bs_501Ent_301MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp0_33bs_601Ent_301MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/disp0_33bs_701Ent_301MAEreg"
# disp=0 ent on joint


)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
