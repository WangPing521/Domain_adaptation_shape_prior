#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=rrg-ebrahimi
save_dir=check_both_layer_disp1_runs

declare -a StringArray=(
#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both001_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both001_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both001_seed3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both001_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both001_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both001_seed3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r6_disp_both001_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r6_disp_both001_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.01 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r6_disp_both001_seed3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both05_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both05_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both05_seed3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both05_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both05_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both05_seed3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r4_disp_both05_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r4_disp_both05_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r4_disp_both05_seed3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r5_disp_both05_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r5_disp_both05_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.5 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r5_disp_both05_seed3"


#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both01_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both01_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=2 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_both01_seed3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=3 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r3_disp_both01_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=3 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r3_disp_both01_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=3 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r3_disp_both01_seed3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both01_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both01_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=4 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.0001   Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r4_disp_both01_seed3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r5_disp_both01_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r5_disp_both01_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=5 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r5_disp_both01_seed3"

#"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r6_disp_both01_seed1"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r6_disp_both01_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=combinationlayer DA.batchsize_indicator=6 DA.double_bn=True DA.multi_scale=6 DA.displacement=True DA.weight2=0.1 Scheduler.RegScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r6_disp_both01_seed3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


