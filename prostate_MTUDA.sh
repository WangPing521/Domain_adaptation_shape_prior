#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=def-chdesa
save_dir=0722_MTUDA_prostate
declare -a StringArray=(
#===
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.0001 Trainer.save_dir=${save_dir}/501MTUDA_lkd1_cons301_seed1"
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.01   Trainer.save_dir=${save_dir}/501MTUDA_lkd1_cons101_seed1"
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1    Trainer.save_dir=${save_dir}/501MTUDA_lkd1_cons01_seed1"
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=1      Trainer.save_dir=${save_dir}/501MTUDA_lkd1_cons1_seed1"
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=2      Trainer.save_dir=${save_dir}/501MTUDA_lkd1_cons2_seed1"
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=4      Trainer.save_dir=${save_dir}/501MTUDA_lkd1_cons4_seed1"
#
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=0.0001 Trainer.save_dir=${save_dir}/501MTUDA_lkd4_cons301_seed1"
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=0.01   Trainer.save_dir=${save_dir}/501MTUDA_lkd4_cons101_seed1"
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=0.1    Trainer.save_dir=${save_dir}/501MTUDA_lkd4_cons01_seed1"
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=1      Trainer.save_dir=${save_dir}/501MTUDA_lkd4_cons1_seed1"
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=2      Trainer.save_dir=${save_dir}/501MTUDA_lkd4_cons2_seed1"
#"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=4      Trainer.save_dir=${save_dir}/501MTUDA_lkd4_cons4_seed1"

"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.0001 Trainer.save_dir=${save_dir}/601MTUDA_lkd1_cons301_seed1"
"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.01   Trainer.save_dir=${save_dir}/601MTUDA_lkd1_cons101_seed1"
"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=0.1    Trainer.save_dir=${save_dir}/601MTUDA_lkd1_cons01_seed1"
"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=1      Trainer.save_dir=${save_dir}/601MTUDA_lkd1_cons1_seed1"
"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=2      Trainer.save_dir=${save_dir}/601MTUDA_lkd1_cons2_seed1"
"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=1 weights.consistency.max_value=4      Trainer.save_dir=${save_dir}/601MTUDA_lkd1_cons4_seed1"

"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=0.0001 Trainer.save_dir=${save_dir}/601MTUDA_lkd4_cons301_seed1"
"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=0.01   Trainer.save_dir=${save_dir}/601MTUDA_lkd4_cons101_seed1"
"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=0.1    Trainer.save_dir=${save_dir}/601MTUDA_lkd4_cons01_seed1"
"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=1      Trainer.save_dir=${save_dir}/601MTUDA_lkd4_cons1_seed1"
"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=2      Trainer.save_dir=${save_dir}/601MTUDA_lkd4_cons2_seed1"
"python MTUDA_main.py seed=10 Data_input.dataset=prostate Data_input.num_class=2 Optim.lr=0.0000001 DataLoader.batch_size=63 weights.lkd_weight.max_value=4 weights.consistency.max_value=4      Trainer.save_dir=${save_dir}/601MTUDA_lkd4_cons4_seed1"


)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
