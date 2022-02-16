#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=def-chdesa
save_dir=Ping_results

declare -a StringArray=(

  "python main.py --seed 10 --log-save-file singlebn_baseline"
  "python main.py --seed 10 --source-cluster-weight 0.1 --log-save-file singlebn_source_cluster_0.1 "
  "python main.py --seed 10 --source-cluster-weight 1 --log-save-file singlebn_source_cluster_1.0 "
  "python main.py --seed 10 --source-cluster-weight 1 --target-cluster-weight 1 --log-save-file singlebn_source_cluster_1.0_target_cluster_1.0"
  "python main.py --seed 10 --source-cluster-weight 1 --target-cluster-weight 1 --alignment-weight 1.0
  --log-save-file singlebn_source_cluster_1.0_target_cluster_1.0_alignment_1.0"

  "python main.py --seed 10 --double-bn --log-save-file doublebn_baseline"
  "python main.py --seed 10 --double-bn --source-cluster-weight 0.1 --log-save-file doublebn_source_cluster_0.1"
  "python main.py --seed 10 --double-bn --source-cluster-weight 1 --log-save-file doublebn_source_cluster_1.0"
  "python main.py --seed 10 --double-bn --source-cluster-weight 1 --target-cluster-weight 1
  --log-save-file doublebn_source_cluster_1.0_target_cluster_1.0"
  "python main.py --seed 10 --double-bn --source-cluster-weight 1 --target-cluster-weight 1 --alignment-weight 1.0
  --log-save-file doublebn_source_cluster_1.0_target_cluster_1.0_alignment_1.0"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


