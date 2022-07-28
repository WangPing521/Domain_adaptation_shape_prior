#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=def-chdesa
save_dir=SOTA_mmwhs_MTUDA

declare -a StringArray=(

#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed3"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=False DA.batchsize_indicator=9 Data.kfold=0 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed3"
#
##EntDA
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/entDA63_701reg_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/entDA63_701reg_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 DA.batchsize_indicator=9 Trainer.name=entda DA.double_bn=True Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/entDA63_701reg_seed3"
#
##pseudo_DA
#"python pseudoDA_main.py seed=10 Data_input.dataset=mmwhs Data.kfold=0 Optim.lr=0.00001 DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_seed1"
#"python pseudoDA_main.py seed=20 Data_input.dataset=mmwhs Data.kfold=0 Optim.lr=0.00001 DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_seed2"
#"python pseudoDA_main.py seed=30 Data_input.dataset=mmwhs Data.kfold=0 Optim.lr=0.00001 DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_seed3"
#
##Entprior
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_405prior_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_405prior_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs DA.double_bn=True Data.kfold=0 Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/EntPrior_401Ent_405prior_seed3"
#
##SIFA
## cyc_weight: 1 2
## cyc_Tweight: 0.5  1
## seg_weight: 1 2 4
## discSeg_weight: 0.0001 0.0005 0.001 0.005 0.1
## disc_weight: 0.0001 0.0005 0.001 0.005 0.1
#"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=2 weights.discSeg_weight=0.0005 weights.disc_weight=0.01 Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg2_disSeg305_dis101_seed1"
#"python SIFA_main.py seed=20 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=2 weights.discSeg_weight=0.0005 weights.disc_weight=0.01 Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg2_disSeg305_dis101_seed2"
#"python SIFA_main.py seed=30 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=2 weights.discSeg_weight=0.0005 weights.disc_weight=0.01 Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg2_disSeg305_dis101_seed3"
#
#MTUDA
"python MTUDA_main.py seed=10 Optim.lr=0.00001 noise=0.01 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.5 Trainer.checkpoint_path=runs/${save_dir}/bs63_lkd1_cons05_seed1 Trainer.save_dir=${save_dir}/bs63_lkd1_cons05_seed1"
"python MTUDA_main.py seed=20 Optim.lr=0.00001 noise=0.01 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.5 Trainer.checkpoint_path=runs/${save_dir}/bs63_lkd1_cons05_seed2 Trainer.save_dir=${save_dir}/bs63_lkd1_cons05_seed2"
"python MTUDA_main.py seed=30 Optim.lr=0.00001 noise=0.01 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.5 Trainer.checkpoint_path=runs/${save_dir}/bs63_lkd1_cons05_seed3 Trainer.save_dir=${save_dir}/bs63_lkd1_cons05_seed3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


