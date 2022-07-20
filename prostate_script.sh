#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=0720_prostate_SOTA_new
declare -a StringArray=(
#todo test is val, and val is test. change the data in two datasts
# Data.kfold=0, only as indicator to allow performing validation
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed3"

#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed3"

# entda
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/entDA_605reg_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/entDA_605reg_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/entDA_605reg_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/entDA_401reg_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/entDA_401reg_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/entDA_401reg_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/entDA_505reg_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/entDA_505reg_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/entDA_505reg_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/entDA_601reg_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/entDA_601reg_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/entDA_601reg_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00000005 Trainer.save_dir=${save_dir}/entDA_705reg_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00000005 Trainer.save_dir=${save_dir}/entDA_705reg_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00000005 Trainer.save_dir=${save_dir}/entDA_705reg_seed3"


# ent+ prior
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_601prior_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_601prior_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_601prior_seed3"


)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
