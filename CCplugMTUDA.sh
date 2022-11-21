#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=8
account=def-chdesa
save_dir=1121_mmwhs_MTUDAplugCC

declare -a StringArray=(
#MTUDA
#
"python MTUDA_main.py seed=10 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=False noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.001 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC201CC_lkd1_cons01_disp0_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=False noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.001 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC201CC_lkd1_cons01_disp0_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=False noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.001 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC201CC_lkd1_cons01_disp0_seed3"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=False noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.0005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC305CC_lkd1_cons01_disp0_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=False noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.0005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC305CC_lkd1_cons01_disp0_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=False noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.0005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC305CC_lkd1_cons01_disp0_seed3"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=False noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.00005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC405CC_lkd1_cons01_disp0_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=False noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.00005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC405CC_lkd1_cons01_disp0_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=False noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.00005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC405CC_lkd1_cons01_disp0_seed3"

#
"python MTUDA_main.py seed=10 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.001 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC201CC_lkd1_cons01_disp1_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.001 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC201CC_lkd1_cons01_disp1_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.001 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC201CC_lkd1_cons01_disp1_seed3"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.0005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC305CC_lkd1_cons01_disp1_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.0005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC305CC_lkd1_cons01_disp1_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.0005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC305CC_lkd1_cons01_disp1_seed3"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.00005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC405CC_lkd1_cons01_disp1_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.00005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC405CC_lkd1_cons01_disp1_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.00005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC405CC_lkd1_cons01_disp1_seed3"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1,3] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.001 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC201CC_lkd1_cons01_disp3_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1,3] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.001 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC201CC_lkd1_cons01_disp3_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1,3] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.001 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC201CC_lkd1_cons01_disp3_seed3"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1,3] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.0005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC305CC_lkd1_cons01_disp3_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1,3] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.0005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC305CC_lkd1_cons01_disp3_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1,3] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.0005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC305CC_lkd1_cons01_disp3_seed3"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1,3] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.00005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC405CC_lkd1_cons015_disp3_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1,3] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.00005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC405CC_lkd1_cons015_disp3_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 Trainer.name=MTUDAplugCCtrainer DA.displacement=True DA.dis_scale=[1,3] noise=0.01 DataLoader.batch_size=63 weights.ccalignScheduler.max_value=0.00005 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/plugCC405CC_lkd1_cons015_disp3_seed3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


