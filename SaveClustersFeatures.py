from collections import OrderedDict
import torch
import matplotlib.pyplot as plt

from arch.DomainSpecificBNUnet import switch_bn, convert2TwinBN
from arch.projectors import DenseClusterHead
from arch.unet import UNet
from arch.utils import FeatureExtractor
from configure import ConfigManager
from dataset.mmwhs import mmWHSCTInterface, mmWHSMRInterface
from utils import tqdm
from utils.image_save_utils import tensor2plotable
from torch.nn.functional import normalize
from utils.utils import fix_all_seed, fix_all_seed_within_context

def plot_seg(img, label):
    fig = plt.figure()
    gt_volume = tensor2plotable(label)
    plt.imshow(gt_volume, alpha=1, cmap="viridis", vmin=0, vmax=29) # vmax is determined by the clusters
    plt.savefig(f'prediction_map/{img}.png')

def plot_feature(img, domain_index='s', f_index=0):
    img_volume = img
    fig = plt.figure()
    img_volume = tensor2plotable(img_volume)
    # plt.imshow(img_volume, cmap="gray")
    plt.imshow(img_volume, alpha=1, cmap="viridis") # vmax is determined by the clusters
    plt.savefig(f'prediction_map/{domain_index}_{f_index}.png')

cmanager = ConfigManager("configs/config.yaml", strict=True)
config = cmanager.config
fix_all_seed(config['seed'])

weight = f'runs/last.pth'
new_state_dict = OrderedDict()
state_dict = torch.load(weight)

model = UNet(num_classes=config['Data_input']['num_class'], input_dim=1)

with fix_all_seed_within_context(config['seed']):
    model = convert2TwinBN(model)

if not config['DA']['align_layer']['cc_based']:
    projector=DenseClusterHead(input_dim=model.get_channel_dim('Up_conv2'), num_clusters=30, T=0.5)
    projector.load_state_dict(state_dict.get('projector'))
model.load_state_dict(state_dict.get('model'))


handler1 = mmWHSMRInterface(seed = config["Data"]["seed"])
handler2 = mmWHSCTInterface(seed = config["Data"]["seed"], kfold=config["Data"]["kfold"])

handler1.compile_dataloader_params(**config["DataLoader"])
handler2.compile_dataloader_params(**config["DataLoader"])

with fix_all_seed_within_context(config['Data']['seed']):
    trainS_loader = handler1.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=config['DA']['batchsize_indicator'],
        constrastve = config['DA']['constrastve_sampler']
    )

    trainT_loader, valT_loader, test_loader = handler2.DataLoaders(
        train_transform=None,
        val_transform=None,
        group_val=False,
        use_infinite_sampler=True,
        batchsize_indicator=config['DA']['batchsize_indicator'],
        constrastve=config['DA']['constrastve_sampler']
    )

S_indicator = tqdm(trainS_loader)
T_tra_indicator = tqdm(trainT_loader)
T_test_indicator = tqdm(test_loader)
valT_loader = tqdm(valT_loader)

device='cpu'
model.to(device)
model.eval()

if not config['DA']['align_layer']['cc_based']:
    projector.to(device)
    projector.eval()

extractor = FeatureExtractor(model, feature_names='Up_conv2')
extractor.bind()


for batch_id, (s_data, t_data) in enumerate(zip(trainS_loader, trainT_loader)):
    S_img, S_target, S_filename = (
        s_data[0][0].to(device),
        s_data[0][1].to(device),
        s_data[1],
    )
    T_img, T_target, T_filename = (
        t_data[0][0].to(device),
        t_data[0][1].to(device),
        t_data[1],
    )

    with switch_bn(model, 0), extractor.enable_register(True):
        extractor.clear()
        pred_S = model(S_img).softmax(1)
        feature_S = next(extractor.features())

    with switch_bn(model, 1), extractor.enable_register(True):
        extractor.clear()
        pred_T = model(T_img).softmax(1)
        feature_T = next(extractor.features())

    if not config['DA']['align_layer']['cc_based']:
        clusters_S = projector(feature_S)
        clusters_T = projector(feature_T)
    else:
        clusters_S = [feature_S]
        clusters_T = [feature_T]

    n, d, _, _ = clusters_S[0].shape
    if not config['DA']['align_layer']['cc_based']:
        plot_seg(S_filename[9], clusters_S[0].max(1)[1][9])
        plot_seg(S_filename[10], clusters_S[0].max(1)[1][10])

        plot_seg(T_filename[2], clusters_T[0].max(1)[1][2])
        plot_seg(T_filename[3], clusters_T[0].max(1)[1][3])
    else:
        feature_S[9] = normalize(feature_S[9], p=16.0, dim=[1,2])
        plot_feature(feature_S[9][0], domain_index='s', f_index=0)
        plot_feature(feature_S[9][1], domain_index='s', f_index=1)
        plot_feature(feature_S[9][2], domain_index='s', f_index=2)
        plot_feature(feature_S[9][3], domain_index='s', f_index=3)
        plot_feature(feature_S[9][4], domain_index='s', f_index=4)
        plot_feature(feature_S[9][5], domain_index='s', f_index=5)
        plot_feature(feature_S[9][6], domain_index='s', f_index=6)
        plot_feature(feature_S[9][7], domain_index='s', f_index=7)
        plot_feature(feature_S[9][8], domain_index='s', f_index=8)
        plot_feature(feature_S[9][9], domain_index='s', f_index=9)
        plot_feature(feature_S[9][10], domain_index='s', f_index=10)
        plot_feature(feature_S[9][11], domain_index='s', f_index=11)
        plot_feature(feature_S[9][12], domain_index='s', f_index=12)
        plot_feature(feature_S[9][13], domain_index='s', f_index=13)
        plot_feature(feature_S[9][14], domain_index='s', f_index=14)
        plot_feature(feature_S[9][15], domain_index='s', f_index=15)

        feature_T[2] = normalize(feature_T[2], p=11.0, dim=[1,2])
        plot_feature(feature_T[2][0], domain_index='t', f_index=0)
        plot_feature(feature_T[2][1], domain_index='t', f_index=1)
        plot_feature(feature_T[2][2], domain_index='t', f_index=2)
        plot_feature(feature_T[9][3], domain_index='t', f_index=3)
        plot_feature(feature_T[9][4], domain_index='t', f_index=4)
        plot_feature(feature_T[9][5], domain_index='t', f_index=5)
        plot_feature(feature_T[9][6], domain_index='t', f_index=6)
        plot_feature(feature_T[9][7], domain_index='t', f_index=7)
        plot_feature(feature_T[9][8], domain_index='t', f_index=8)
        plot_feature(feature_T[9][9], domain_index='t', f_index=9)
        plot_feature(feature_T[9][10], domain_index='t', f_index=10)
        plot_feature(feature_T[9][11], domain_index='t', f_index=11)
        plot_feature(feature_T[9][12], domain_index='t', f_index=12)
        plot_feature(feature_T[9][13], domain_index='t', f_index=13)
        plot_feature(feature_T[9][14], domain_index='t', f_index=14)
        plot_feature(feature_T[9][15], domain_index='t', f_index=15)