#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=def-chdesa
save_dir=0717_MTUDA_prostate_new
declare -a StringArray=(
#===
"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.5 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons05_seed1"
"python MTUDA_main.py seed=20 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.5 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons05_seed2"
"python MTUDA_main.py seed=30 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.5 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons05_seed3"

"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.005 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons205_seed1"
"python MTUDA_main.py seed=20 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.005 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons205_seed2"
"python MTUDA_main.py seed=30 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.005 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons205_seed3"

"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.01 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons101_seed1"
"python MTUDA_main.py seed=20 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.01 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons101_seed2"
"python MTUDA_main.py seed=30 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.01 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons101_seed3"

"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons01_seed1"
"python MTUDA_main.py seed=20 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons01_seed2"
"python MTUDA_main.py seed=30 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000005 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1 Trainer.save_dir=${save_dir}/505_bs63_lkd1_cons01_seed3"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
