#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=check_both_layer

declare -a StringArray=(
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_both001"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both001"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=3 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r3_disp_both001"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both001"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r5_disp_both001"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r6_disp_both001"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_both001"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r2_disp_both001"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=3 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r3_disp_both001"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r4_disp_both001"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r5_disp_both001"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r6_disp_both001"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=3 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r3_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r5_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r6_disp_both05"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r2_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=3 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r3_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r4_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r5_disp_both05"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r6_disp_both05"


"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=3 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r3_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r5_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r6_disp_both01"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=1 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r2_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=3 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r3_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r4_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r5_disp_both01"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r6_disp_both01"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


