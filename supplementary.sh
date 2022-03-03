#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=rrg-ebrahimi
save_dir=test_prioebased_disp0
declare -a StringArray=(
# prior based
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_401Ent_401prior"

# disp=0 align: mae
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_21bs_401MAEreg"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_21bs_401Ent_401MAEreg"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_21bs_401MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_21bs_501MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_21bs_601MAEreg"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_21bs_501Ent_401MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp0_21bs_601Ent_401MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/disp0_21bs_701Ent_401MAEreg"

# disp=0 align: KL
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_21bs_201KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_21bs_301KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_21bs_401KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_21bs_501KLreg"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_21bs_401Ent_401KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_21bs_501Ent_401KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp0_21bs_601Ent_401KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/disp0_21bs_701Ent_401KLreg"


# disp=0 large batch size
            # disp=0 align: mae
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_10bs_401MAEreg"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_10bs_401Ent_401MAEreg"

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_33bs_301MAEreg"
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_33bs_401MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_33bs_501MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_33bs_601MAEreg"

#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_33bs_401Ent_401MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_33bs_501Ent_401MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp0_33bs_601Ent_401MAEreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=MAE Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/disp0_33bs_701Ent_401MAEreg"

            # disp=0 align: KL
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_33bs_301KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_33bs_401KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.000001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_33bs_501KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.0000001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/disp0_33bs_601KLreg"


"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_33bs_401Ent_401KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_33bs_501Ent_401KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.0000001 Trainer.save_dir=${save_dir}/disp0_33bs_601Ent_401KLreg"
"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=10 DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False DA.align_type=kl Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00000001 Trainer.save_dir=${save_dir}/disp0_33bs_701Ent_401KLreg"

# disp=0 ent on joint


)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
