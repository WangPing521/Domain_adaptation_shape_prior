#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=0326_prostate_align
declare -a StringArray=(

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=15 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_305MAEregjoint48_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=15 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301MAEregjoint48_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=15 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405MAEregjoint48_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=15 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_401MAEregjoint48_pred"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=15 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0005  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_305MAEregjoint48_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=15 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301MAEregjoint48_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=15 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_405MAEregjoint48_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=15 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_401MAEregjoint48_pred"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=20 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_305MAEregjoint63_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=20 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_301MAEregjoint63_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=20 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405MAEregjoint63_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=20 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_401MAEregjoint63_pred"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=20 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0005  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_305MAEregjoint63_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=20 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001  Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301MAEregjoint63_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=20 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_405MAEregjoint63_pred"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=20 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_401MAEregjoint63_pred"

)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
