#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=0721_pseudoDA_prostate
declare -a StringArray=(
#todo test is val, and val is test. change the data in two datasts
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed3"
#
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed3"
#
## entda
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/entDA_605reg_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/entDA_605reg_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/entDA_605reg_seed3"
#
## ent+ prior
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_601prior_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_601prior_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=0 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_601prior_seed3"

#pseudo_DA
"python pseudoDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Optim.lr=0.00001 DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_seed1"
"python pseudoDA_main.py seed=20 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Optim.lr=0.00001 DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_seed2"
"python pseudoDA_main.py seed=30 Data_input.dataset=prostate Data_input.num_class=2 Data.kfold=0 Optim.lr=0.00001 DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_seed3"

#todo SIFA
#"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis201_seed1"
#"python SIFA_main.py seed=20 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis201_seed2"
#"python SIFA_main.py seed=30 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis201_seed3"

#"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis205_seed1"
#"python SIFA_main.py seed=20 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis205_seed2"
#"python SIFA_main.py seed=30 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis205_seed3"

#"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0005 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg305_dis101_seed1"
#"python SIFA_main.py seed=20 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0005 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg305_dis101_seed2"
#"python SIFA_main.py seed=30 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0005 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg305_dis101_seed3"

#todo MTUDA


)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
