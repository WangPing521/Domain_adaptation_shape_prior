#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=0412_state2mise_p7_entda
declare -a StringArray=(

# state2mise partition=7
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline63_seed1"
#"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline63_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline63_seed1"
#"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline63_seed3"

#align change ent's weight
# 9       63
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_disp0_401Ent_501joint63"
#"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_disp0_401Ent_501joint63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.00001   Trainer.save_dir=${save_dir}/s2m_disp0_401Ent_501joint63_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_disp0_501Ent_501joint63"
#"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_disp0_501Ent_501joint63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_disp0_501Ent_501joint63_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_gtSdisp0_501Ent_501joint63"
#"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_gtSdisp0_501Ent_501joint63_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/s2m_gtSdisp0_501Ent_501joint63_seed3"


# entda
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000000005 Trainer.save_dir=${save_dir}/s2m_entDA63_805reg_seed1"
"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000000005 Trainer.save_dir=${save_dir}/s2m_entDA63_805reg_seed2"
"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.000000005 Trainer.save_dir=${save_dir}/s2m_entDA63_805reg_seed3"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000000005 Trainer.save_dir=${save_dir}/s2m_entDA63_905reg_seed1"
"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000000005 Trainer.save_dir=${save_dir}/s2m_entDA63_905reg_seed2"
"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000000005 Trainer.save_dir=${save_dir}/s2m_entDA63_905reg_seed3"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000000001 Trainer.save_dir=${save_dir}/s2m_entDA63_901reg_seed1"
"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000000001 Trainer.save_dir=${save_dir}/s2m_entDA63_901reg_seed2"
"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=entda DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000000001 Trainer.save_dir=${save_dir}/s2m_entDA63_901reg_seed3"

# ent+ prior
#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_601prior"
#"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_601prior_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_601prior_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_605prior"
#"python main.py seed=231 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_605prior_seed2"
#"python main.py seed=321 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=True Trainer.name=priorbased DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Scheduler.RegScheduler.max_value=0.0000005 Scheduler.ClusterScheduler.max_value=0.000001  Trainer.save_dir=${save_dir}/prior_501Ent_605prior_seed3"


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
