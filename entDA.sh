#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=0317_entDA_BS_weight

declare -a StringArray=(

# entropy 6:21 15:48 20:63 25:78 30:93 40:123
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA21_601reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA21_701reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA21_801reg"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=15 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA48_601reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=15 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA48_701reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=15 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA48_801reg"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=20 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA63_601reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=20 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA63_701reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=20 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA63_801reg"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=25 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA78_601reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=25 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA78_701reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=25 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA78_801reg"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=30 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA93_601reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=30 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA93_701reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=30 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA93_801reg"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=40 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA123_601reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=40 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA123_701reg"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=40 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA123_801reg"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


