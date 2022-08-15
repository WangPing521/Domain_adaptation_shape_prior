# Domain_adaptation_shape_prior
-----
* Dataset:

MMWHS (MRI(Source) and CT(Target))

* Network: 

Unet

* Trainers:

align_IBN_trainer.py----->alignment from output layer or Upconv2

align_combinationlayer_trainer.py------->alignment from both output layer and Upconv2 layer

Other trainers correspond to SOTA.

* Reproduce our results:

```bash
bash mmwhs_our_1.sh
```
```bash
bash mmwhs_our_2.sh
```
