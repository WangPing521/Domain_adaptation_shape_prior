#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=rrg-ebrahimi
save_dir=olva
common="Scheduler.RegScheduler.begin_epoch=0 Scheduler.RegScheduler.max_epoch=0 Scheduler.ClusterScheduler.begin_epoch=0 Scheduler.ClusterScheduler.max_epoch=0  "
declare -a StringArray=(
  # CT2MRI
  # lr = 0.00001
  #------------------MR2CT
  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0   Trainer.enable_sampling=true   Trainer.save_dir=${save_dir}/OLVA_0kl_0ot/sample_true"
  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0 Scheduler.ClusterScheduler.max_value=0   Trainer.enable_sampling=false   Trainer.save_dir=${save_dir}/OLVA_0kl_0ot/sample_false"

  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.1    Scheduler.ClusterScheduler.max_value=1.0 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_1.0kl_0.1ot"
  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.01   Scheduler.ClusterScheduler.max_value=1.0 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_1.0kl_101ot"
  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.001  Scheduler.ClusterScheduler.max_value=1.0 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_1.0kl_201ot"
  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=1.0 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_1.0kl_301ot"

  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.1    Scheduler.ClusterScheduler.max_value=0.1 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_0.1kl_0.1ot"
  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.01   Scheduler.ClusterScheduler.max_value=0.1 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_0.1kl_101ot"
  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.001  Scheduler.ClusterScheduler.max_value=0.1 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_0.1kl_201ot"
  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.1 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_0.1kl_301ot"

  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.1    Scheduler.ClusterScheduler.max_value=0.01 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_0.01kl_0.1ot"
  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.01   Scheduler.ClusterScheduler.max_value=0.01 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_0.01kl_101ot"
  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.001  Scheduler.ClusterScheduler.max_value=0.01 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_0.01kl_201ot"
  "python main.py ${common} seed=123 Optim.lr=0.00001 Trainer.name=ottrainer DA.align_layer.name=Deconv_1x1  DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.double_bn=True Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.01 Trainer.enable_sampling=true  Trainer.save_dir=${save_dir}/OLVA_0.01kl_301ot"


)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
