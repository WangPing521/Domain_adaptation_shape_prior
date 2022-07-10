from collections import OrderedDict
import torch

from arch.unet import UNet, decoderU
from arch.utils import FeatureExtractor
from configure import ConfigManager
from dataset import PromiseInterface, ProstateInterface
from dataset.mmwhs import mmWHSCTInterface, mmWHSMRInterface
from utils.image_save_utils import save_images
from utils.utils import fix_all_seed, fix_all_seed_within_context
from utils import tqdm

cmanager = ConfigManager("configs/cyc_config.yaml", strict=True)
config = cmanager.config
fix_all_seed(config['seed'])

weight = f'runs/SIFA/prostate/last.pth'
new_state_dict = OrderedDict()
state_dict = torch.load(weight)

model_S2T = UNet(input_dim=1, num_classes=1)

model = UNet(num_classes=config['Data_input']['num_class'], input_dim=1)
decoder = decoderU(input_dim=256, out_dim=1)


model_S2T.load_state_dict(state_dict.get('Generator'))
model.load_state_dict(state_dict.get('model'))
decoder.load_state_dict(state_dict.get('decoder'))


if config['Data_input']['dataset'] == 'mmwhs':
    handler1 = mmWHSMRInterface(seed = config["Data"]["seed"])
    handler2 = mmWHSCTInterface(seed = config["Data"]["seed"], kfold=config["Data"]["kfold"])
elif config['Data_input']['dataset'] == 'prostate':
    handler1 = ProstateInterface(seed = config["Data"]["seed"])
    handler2 = PromiseInterface(seed = config["Data"]["seed"])
else:
    raise NotImplementedError(config['Data_input']['dataset'])

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

model_S2T.eval()
model.eval()
decoder.eval()
extractor = FeatureExtractor(model, feature_names=[f"Conv{str(f)}" for f in range(5, 0, -1)])
extractor.bind()

# for batch_id1, data_S in enumerate(S_indicator):
#     imageS, targetS, filenameS = (
#         data_S[0][0],
#         data_S[0][1],
#         data_S[1]
#     )
#
#     S2T = torch.tanh(model_S2T(imageS))
#
#     save_images(S2T.squeeze(1), filenameS, root=config['Trainer']['save_dir'], mode='fake_ct_train', iter=0)
#
#     with extractor.enable_register(True):
#         extractor.clear()
#         pred_T = model(S2T).softmax(1)
#         e_list_T = list(extractor.features())
#     S2T2S = torch.tanh(decoder(e_list_T))
#
#     save_images(S2T2S.squeeze(1), filenameS, root=config['Trainer']['save_dir'], mode='recover_mr_train', iter=0)


# for batch_id2, data_tra_T in enumerate(T_tra_indicator):
#     image_tra_T, target_tra_T, filename_traT = (
#         data_tra_T[0][0],
#         data_tra_T[0][1],
#         data_tra_T[1]
#     )
#     with extractor.enable_register(True):
#         extractor.clear()
#         pred_T = model(image_tra_T).softmax(1)
#         e_list_T = list(extractor.features())
#     T2S = torch.tanh(decoder(e_list_T))
#
#     save_images(T2S.squeeze(1), filename_traT, root=config['Trainer']['save_dir'], mode='fake_mr_train', iter=0)
#
#
#     T2S2T = torch.tanh(model_S2T(T2S))
#
#     save_images(T2S2T.squeeze(1), filename_traT, root=config['Trainer']['save_dir'], mode='recover_ct_train', iter=0)



for batch_id3, data_test_T in enumerate(T_test_indicator):
    image_test_T, target_test_T, filename_testT = (
        data_test_T[0][0],
        data_test_T[0][1],
        data_test_T[1]
    )

    with extractor.enable_register(True):
        extractor.clear()
        pred_T = model(image_test_T).softmax(1)
        e_list_T = list(extractor.features())
    T2S_test = torch.tanh(decoder(e_list_T))

    save_images(T2S_test.squeeze(1), filename_testT, root=config['Trainer']['save_dir'], mode='fake_mr_test', iter=0)
