# Domain_adaptation_shape_prior
-----
*Dataset:

MMWHS (MRI(Source) and CT(Target))

*Network: 

Unet

*Trainers:

align_IBN_trainer.py----->alignment from output layer or Upconv2

align_combinationlayer_trainer.py------->alignment from both output layer and Upconv2 layer

Other trainers correspond to SOTA.

scripts:

mmwhs_our_1.sh

mmwhs_our_2.sh

```bash
python main_nd.py -o Trainer.name=ft -p config/base.yaml
```
