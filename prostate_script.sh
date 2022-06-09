#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=0607_prostate_baseline
declare -a StringArray=(
# Data.kfold=10, only as indicator to allow performing validation
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/lower_baseline_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=False DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/upper_baseline_seed3"

# entda
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/entDA_505reg_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/entDA_505reg_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/entDA_505reg_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/entDA_501reg_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/entDA_501reg_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/entDA_501reg_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/entDA_605reg_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/entDA_605reg_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Trainer.save_dir=${save_dir}/entDA_605reg_seed3"

# ent+ prior
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_601prior_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_601prior_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_601prior_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_605prior_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_605prior_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=prostate Data.kfold=10 Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/prior_501Ent_605prior_seed3"

#disp0
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_disp0_401Ent_501joint63"
#"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_disp0_401Ent_501joint63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_disp0_401Ent_501joint63_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_disp0_501Ent_501joint63"
#"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_disp0_501Ent_501joint63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_disp0_501Ent_501joint63_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_gtSdisp0_501Ent_501joint63"
#"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_gtSdisp0_501Ent_501joint63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_gtSdisp0_501Ent_501joint63_seed3"


#PLDA
#"python pseudoDA_main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_401seed1"
#"python pseudoDA_main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_401seed2"
#"python pseudoDA_main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_401seed3"

#"python pseudoDA_main.py seed=123 Optim.lr=0.000001 Data_input.dataset=prostate Data_input.num_class=2 DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_501seed1"
#"python pseudoDA_main.py seed=231 Optim.lr=0.000001 Data_input.dataset=prostate Data_input.num_class=2 DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_501seed2"
#"python pseudoDA_main.py seed=321 Optim.lr=0.000001 Data_input.dataset=prostate Data_input.num_class=2 DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.save_dir=${save_dir}/pseudoDA_501seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/s2m_disp1_401Ent_501joint_r3_seed1"
#"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/s2m_disp1_401Ent_501joint_r3_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/s2m_disp1_401Ent_501joint_r3_seed3"


)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
