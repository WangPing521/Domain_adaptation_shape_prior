#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=0317_alignjoint_gtS
declare -a StringArray=(

#with ent
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6  DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.statistic=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301MAEregjointgtS21_pred"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6  DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301MAEregjoint21_pred"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=15 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.statistic=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301MAEregjointgtS48_pred"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=15 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301MAEregjoint48_pred"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=20 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.statistic=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301MAEregjointgtS63_pred"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=20 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301MAEregjoint63_pred"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=25 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.statistic=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301MAEregjointgtS78_pred"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=25 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301MAEregjoint78_pred"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=30 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.statistic=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301MAEregjointgtS93_pred"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=30 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301MAEregjoint93_pred"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=40 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.statistic=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301MAEregjointgtS123_pred"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=40 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301MAEregjoint123_pred"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
