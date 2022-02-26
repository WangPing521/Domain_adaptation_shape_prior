#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=psuedoDA

declare -a StringArray=(
# CT2MRI
# lr = 0.00001
#------------------MR2CT
"python pseudoDA_main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.save_dir=${save_dir}/pseudoDA_seed1"
"python pseudoDA_main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.save_dir=${save_dir}/pseudoDA_seed2"
"python pseudoDA_main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Trainer.save_dir=${save_dir}/pseudoDA_seed3"


)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


