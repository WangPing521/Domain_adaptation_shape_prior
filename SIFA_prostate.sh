#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=def-chdesa
save_dir=0707_SIFA_prostate_runs
declare -a StringArray=(
# cyc_weight: 1 2
# cyc_Tweight: 0.5  1
# seg_weight: 1 2 4
# discSeg_weight: 0.0001 0.0005 0.001 0.005 0.1
# disc_weight: 0.0001 0.0005 0.001 0.005 0.1

#todo check weights  (note: maybe the batch size is too large for this method, leading to out of memory)
#"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis201_seed1"
"python SIFA_main.py seed=20 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis201_seed2"
"python SIFA_main.py seed=30 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.001  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis201_seed3"

#"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis205_seed1"
"python SIFA_main.py seed=20 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis205_seed2"
"python SIFA_main.py seed=30 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0001 weights.disc_weight=0.005  Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg301_dis205_seed3"

#"python SIFA_main.py seed=10 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0005 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg305_dis101_seed1"
"python SIFA_main.py seed=20 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0005 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg305_dis101_seed2"
"python SIFA_main.py seed=30 Data.kfold=0 Optim.lr=0.000001 Optim.disc_lr=0.000005 Data_input.dataset=prostate Data_input.num_class=2 DataLoader.batch_size=32 weights.cyc_weight=1 weights.cyc_Tweight=0.5 weights.seg_weight=1 weights.discSeg_weight=0.0005 weights.disc_weight=0.01   Trainer.save_dir=${save_dir}/cyc1_Tcyc05_seg1_disSeg305_dis101_seed3"

######
###cyc1_Tcyc05_seg4_ (25)

###cyc1_Tcyc1_seg1_ (25)

###cyc1_Tcyc1_seg2_ (25)

###cyc1_Tcyc1_seg4_ (25)

###cyc2_Tcyc05_seg1_ (25)   cyc2_Tcyc05_seg2_ (25)   cyc2_Tcyc05_seg4_ (25)
###cyc2_Tcyc1_seg1_ (25)   cyc2_Tcyc1_seg2_ (25)   cyc2_Tcyc1_seg4_ (25)
)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
