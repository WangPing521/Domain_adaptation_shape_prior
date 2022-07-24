import warnings
from collections import OrderedDict
import torch
from imageio import imsave
from torch import Tensor
from typing import Union, Iterable
from arch.DomainSpecificBNUnet import switch_bn as _switch_bn, convert2TwinBN
from arch.unet import UNet
from dataset import PromiseInterface
from dataset.mmwhs import mmWHSCTInterface
from demo.criterions import nullcontext
from pathlib import Path
import numpy as np
from utils.utils import fix_all_seed_within_context

torch.backends.cudnn.benchmark = True
def save_images(segs: Tensor, names: Iterable[str], root: Union[str, Path], mode: str, iter: int) -> None:
    (b, w, h) = segs.shape  # type: Tuple[int, int,int] # Since we have the class numbers, we do not need a C axis
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for seg, name in zip(segs, names):
            # mmwhs: name[9:13]
            # prostate: name[4:6]
            save_path = Path(root, f"iter{iter:03d}", mode, name[9:13], name).with_suffix(".png")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            imsave(str(save_path), seg.cpu().numpy().astype(np.uint8))


dataset = 'mmwhs'
double_bn = False
# mmwhs: 5
# prostate: 2
Smodel = UNet(num_classes=5, input_dim=1)
if double_bn:
    Smodel = convert2TwinBN(Smodel)
weight = f'../../PHD_documents/papers_work/domain_adaptation/MMWHS/SOTA/cyc1_Tcyc05_seg2_disSeg305_dis101_seed1/last.pth'
new_state_dict = OrderedDict()
state_dict = torch.load(weight, map_location=torch.device('cpu'))
Smodel.load_state_dict(state_dict.get('model'))
Smodel = Smodel.eval()

if dataset == 'mmwhs':
    target_handler = mmWHSCTInterface(seed=12)
else:
    target_handler = PromiseInterface(seed=12)
target_handler.compile_dataloader_params(batch_size=10, val_batch_size=10,shuffle=False,num_workers=1,pin_memory=False)

with fix_all_seed_within_context(seed=12):
    trainT_loader, valT_loader, test_loader = target_handler.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=9
    )

switch_bn = _switch_bn if True else nullcontext
for batch_idT, data_T in enumerate(test_loader):
    imageT, targetT, filenameT = (
        data_T[0][0].to('cpu'),
        data_T[0][1].to('cpu'),
        data_T[1]
    )
    if double_bn:
        with switch_bn(Smodel, 1):
            preds_T = Smodel(imageT).softmax(1)
    else:
        preds_T = Smodel(imageT).softmax(1)

    # save_images(targetT.squeeze(1), names=filenameT, root='../../PHD_documents/papers_work/domain_adaptation/test_mmwhs',
    #             mode='gt',
    #             iter=100)
    save_images(preds_T.max(1)[1], names=filenameT, root='../../PHD_documents/papers_work/domain_adaptation/test_mmwhs/sifa', mode='predictions',
            iter=100)


