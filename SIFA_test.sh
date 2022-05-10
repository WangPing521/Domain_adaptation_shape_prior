#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=rrg-ebrahimi
save_dir=SIFA_0509
declare -a StringArray=(
# align
"python SIFA_main.py seed=123 DA.source=MRI DA.target=CT DA.batchsize_indicator=2
 Scheduler.RegScheduler_advs.min_value=0.1  Scheduler.RegScheduler_advs.max_value=0.1
 Scheduler.RegScheduler_cyc.min_value=10    Scheduler.RegScheduler_cyc.max_value=10
 Scheduler.RegScheduler_seg2.min_value=0.1  Scheduler.RegScheduler_seg2.max_value=0.1
 Scheduler.RegScheduler_advp1.min_value=0.1 Scheduler.RegScheduler_advp1.max_value=0.1
 Scheduler.RegScheduler_advss.min_value=0.1 Scheduler.RegScheduler_advss.max_value=0.1
 Trainer.save_dir=${save_dir}/SIFA_case1"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
