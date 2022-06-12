#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=0612_PLDA

declare -a StringArray=(

# do not nedd cross validation
"python pseudoDA_main.py seed=10 Data_input.dataset=mmwhs Data.kfold=0 Optim.lr=0.00001 DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_seed1"
"python pseudoDA_main.py seed=20 Data_input.dataset=mmwhs Data.kfold=0 Optim.lr=0.00001 DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_seed2"
"python pseudoDA_main.py seed=30 Data_input.dataset=mmwhs Data.kfold=0 Optim.lr=0.00001 DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_seed3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


