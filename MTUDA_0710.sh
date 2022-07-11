#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=def-chdesa
save_dir=0710_MTUDA
declare -a StringArray=(
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=1 weights.structual=1    Trainer.save_dir=${save_dir}/cons1_stru1_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=1 weights.structual=0.5  Trainer.save_dir=${save_dir}/cons1_stru05_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=1 weights.structual=0.1  Trainer.save_dir=${save_dir}/cons1_stru01_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=1 weights.structual=0.05 Trainer.save_dir=${save_dir}/cons1_stru105_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=1 weights.structual=0.01 Trainer.save_dir=${save_dir}/cons1_stru101_seed1"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.5 weights.structual=1    Trainer.save_dir=${save_dir}/cons05_stru1_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.5 weights.structual=0.5  Trainer.save_dir=${save_dir}/cons05_stru05_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.5 weights.structual=0.1  Trainer.save_dir=${save_dir}/cons05_stru01_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.5 weights.structual=0.05 Trainer.save_dir=${save_dir}/cons05_stru105_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.5 weights.structual=0.01 Trainer.save_dir=${save_dir}/cons05_stru101_seed1"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.1 weights.structual=1    Trainer.save_dir=${save_dir}/cons01_stru1_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.1 weights.structual=0.5  Trainer.save_dir=${save_dir}/cons01_stru05_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.1 weights.structual=0.1  Trainer.save_dir=${save_dir}/cons01_stru01_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.1 weights.structual=0.05 Trainer.save_dir=${save_dir}/cons01_stru105_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.1 weights.structual=0.01 Trainer.save_dir=${save_dir}/cons01_stru101_seed1"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.05 weights.structual=1    Trainer.save_dir=${save_dir}/cons105_stru1_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.05 weights.structual=0.5  Trainer.save_dir=${save_dir}/cons105_stru05_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.05 weights.structual=0.1  Trainer.save_dir=${save_dir}/cons105_stru01_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.05 weights.structual=0.05 Trainer.save_dir=${save_dir}/cons105_stru105_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.05 weights.structual=0.01 Trainer.save_dir=${save_dir}/cons105_stru101_seed1"

"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.01 weights.structual=1    Trainer.save_dir=${save_dir}/cons101_stru1_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.01 weights.structual=0.5  Trainer.save_dir=${save_dir}/cons101_stru05_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.01 weights.structual=0.1  Trainer.save_dir=${save_dir}/cons101_stru01_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.01 weights.structual=0.05 Trainer.save_dir=${save_dir}/cons101_stru105_seed1"
"python MTUDA_main.py seed=10 Optim.lr=0.00001 DataLoader.batch_size=32 weights.consistency=0.01 weights.structual=0.01 Trainer.save_dir=${save_dir}/cons101_stru101_seed1"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
