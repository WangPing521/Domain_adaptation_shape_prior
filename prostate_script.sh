#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=0411_state2mise_partition7_runs
declare -a StringArray=(

# state2mise partition=7
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline63_seed1"
"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline63_seed2"
"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline63_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline63_seed1"
"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline63_seed2"
"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline63_seed3"

#align change ent's weight
# 9       63
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_disp1_401Ent_501joint_r2_seed1"
"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_disp1_401Ent_501joint_r2_seed2"
"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_disp1_401Ent_501joint_r2_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_disp1_501Ent_501joint_r2_seed1"
"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_disp1_501Ent_501joint_r2_seed2"
"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_disp1_501Ent_501joint_r2_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_gtSdisp1_401Ent_501joint_r4_seed1"
"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_gtSdisp1_401Ent_501joint_r4_seed2"
"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_gtSdisp1_401Ent_501joint_r4_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_gtSdisp1_501Ent_501joint_r4_seed1"
"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_gtSdisp1_501Ent_501joint_r4_seed2"
"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_gtSdisp1_501Ent_501joint_r4_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/s2m_gtSdisp1_601Ent_505joint_r4_seed1"
"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/s2m_gtSdisp1_601Ent_505joint_r4_seed2"
"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=4 DA.displacement=True Scheduler.RegScheduler.max_value=0.000005 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/s2m_gtSdisp1_601Ent_505joint_r4_seed3"


# entda
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/s2m_entDA21_401reg"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/s2m_entDA21_501reg"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/s2m_entDA21_601reg"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/s2m_entDA21_701reg"

# ent+ prior
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_401Ent_301prior"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_401Ent_401prior"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_401Ent_501prior"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_301prior"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_401prior"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_501prior"


# state2mise partition=4
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=8 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline32_seed1"
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=8 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline32_seed1"

# 32
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=8 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000005 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/state2mise_disp0_501Ent_505regjoint32"
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=8 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/state2mise_disp0_501Ent_501regjoint32"
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=8 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.0000001  Trainer.save_dir=${save_dir}/state2mise_disp0_601Ent_501regjoint32"
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=8 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/state2mise_disp0_701Ent_501regjoint32"


)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
