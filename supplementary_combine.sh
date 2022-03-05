#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=rrg-ebrahimi
save_dir=NoGradS_ent_align_begin20_combine
declare -a StringArray=(

# combination layers    rerun
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.weight2=0.01 Scheduler.RegScheduler.begin_epoch=20 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/combine_disp0_21bs_501Ent_301MAEreg_101weight"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.weight2=0.01 Scheduler.RegScheduler.begin_epoch=20 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/combine_disp0_21bs_501Ent_401MAEreg_101weight"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.weight2=0.01 Scheduler.RegScheduler.begin_epoch=20 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/combine_disp0_21bs_601Ent_301MAEreg_101weight"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.weight2=0.01 Scheduler.RegScheduler.begin_epoch=20 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/combine_disp0_21bs_601Ent_401MAEreg_101weight"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.source=MRI DA.target=CT DA.batchsize_indicator=20 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.weight2=0.01 Scheduler.RegScheduler.begin_epoch=20 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/combine_disp0_63bs_501Ent_301MAEreg_101weight"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.source=MRI DA.target=CT DA.batchsize_indicator=20 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.weight2=0.01 Scheduler.RegScheduler.begin_epoch=20 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/combine_disp0_63bs_501Ent_401MAEreg_101weight"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.source=MRI DA.target=CT DA.batchsize_indicator=20 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.weight2=0.01 Scheduler.RegScheduler.begin_epoch=20 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/combine_disp0_63bs_601Ent_301MAEreg_101weight"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=combinationlayer DA.source=MRI DA.target=CT DA.batchsize_indicator=20 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.weight2=0.01 Scheduler.RegScheduler.begin_epoch=20 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/combine_disp0_63bs_601Ent_401MAEreg_101weight"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
