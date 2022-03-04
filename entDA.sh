#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=rrg-ebrahimi
save_dir=entDA_weightsearch

declare -a StringArray=(

# entropy
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_501reg"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_501reg_run2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_501reg_run3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_601reg"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_601reg_run2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_601reg_run3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_701reg"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_701reg_run2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_701reg_run3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_801reg"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_801reg_run2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_801reg_run3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_901reg"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_901reg_run2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_901reg_run3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_1001reg"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_1001reg_run2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.0000000001 Trainer.save_dir=${save_dir}/MRI2CT_entDA_1001reg_run3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


