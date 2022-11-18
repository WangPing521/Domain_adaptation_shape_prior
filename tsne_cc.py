from collections import OrderedDict
from utils import tqdm
import torch

from arch.DomainSpecificBNUnet import convert2TwinBN, switch_bn as _switch_bn
from arch.utils import FeatureExtractor
from dataset.mmwhs import mmWHSMRInterface, mmWHSCTInterface
from utils.image_save_utils import draw_pictures
from utils.utils import fix_all_seed_within_context, fix_all_seed
from arch.unet import UNet

fix_all_seed(12)
switch_bn = _switch_bn
device= 'cuda'

with fix_all_seed_within_context(12):
    model = UNet(num_classes=5, input_dim=1)

with fix_all_seed_within_context(12):
    model = convert2TwinBN(model)

weight = f'runs/upconv2cc/last.pth'   #upconv2cc
new_state_dict = OrderedDict()
state_dict = torch.load(weight)
model.load_state_dict(state_dict.get('model'))
model.to(device)
model = model.eval()

extractor = FeatureExtractor(model, feature_names="Up_conv2")
extractor.bind()

handler1 = mmWHSMRInterface(seed = 12)
handler2 = mmWHSCTInterface(seed = 12, kfold=0)

handler1.compile_dataloader_params(batch_size= 1, val_batch_size=20, shuffle=False, num_workers=8, pin_memory= False)
handler2.compile_dataloader_params(batch_size= 1, val_batch_size=20, shuffle=False, num_workers=8, pin_memory= False)

with fix_all_seed_within_context(12):
    trainS_loader = handler1.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=2
    )
    trainT_loader, valT_loader, test_loader = handler2.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=2
    )

batch_indicator = tqdm(range(100))
T_indicator = tqdm(test_loader)

for batch_id, (idx, s_data) in enumerate(zip(batch_indicator, trainS_loader)):
    S_img, S_target, S_filename = (
        s_data[0][0].to(device),
        s_data[0][1].to(device),
        s_data[1],
    )

    with switch_bn(model, 0), extractor.enable_register(True):
        extractor.clear()
        pred_S = model(S_img).softmax(1)
        feature_S = next(extractor.features())

    for i in range(S_target.shape[0]):
        if S_target[i].unique().size()[0] == 5:
            feature_S0 = feature_S[i].cpu().detach().numpy().reshape(16, -1).transpose()
            S_lab0 = S_target[i].cpu().numpy().squeeze(0).reshape(-1)
            feature_S0 = feature_S0[::10]
            S_lab0 = S_lab0[::10]
            draw_pictures(feature_S0, S_lab0, f"entS_{batch_id}_{i}.png", show_legend=True)


for batch_id, t_data in enumerate(T_indicator):

    T_img, T_target, T_filename = (
        t_data[0][0].to(device),
        t_data[0][1].to(device),
        t_data[1],
    )

    with switch_bn(model, 1), extractor.enable_register(True):
        extractor.clear()
        pred_T = model(T_img).softmax(1)
        feature_T = next(extractor.features())

    for i in range(T_target.shape[0]):
        if T_target[i].unique().size()[0] == 5:
            feature_T0 = feature_T[i].cpu().detach().numpy().reshape(16, -1).transpose()
            T_lab0 = T_target[i].cpu().numpy().squeeze(0).reshape(-1)
            feature_T0 = feature_T0[::10]
            T_lab0 = T_lab0[::10]
            draw_pictures(feature_T0, T_lab0, f"TSNE_VIS_CC/T_{batch_id}_{i}.png", show_legend=False)




