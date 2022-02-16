import contextlib

from torch import nn
import torch
from arch.unet import UNet

_TwinBNCONTEXT = []

class TwinBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None,
                 dtype=None):
        super(TwinBatchNorm2d, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.bn2 = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self._indicator = 0

    def forward(self, x):
        if len(_TwinBNCONTEXT) == 0:
            raise RuntimeError(f"{self.__class__.__name__} must be used with context manager of `switch_bn`.")
        if self.indicator == 0:
            return self.bn1(x)
        else:
            return self.bn2(x)

    @property
    def indicator(self):
        return self._indicator

    @indicator.setter
    def indicator(self, value):
        assert value in (0, 1)
        self._indicator = value

def convert2TwinBN(module: nn.Module):
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = TwinBatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
            device=module.running_mean.device,
            dtype=module.running_mean.dtype,
        )
    for name, child in module.named_children():
        module_output.add_module(
            name, convert2TwinBN(child)
        )
    del module
    return module_output

@contextlib.contextmanager
def switch_bn(module: nn.Module, indicator: int):
    _TwinBNCONTEXT.append("A")
    previous_state = {n: v.indicator for n, v in module.named_modules() if isinstance(v, TwinBatchNorm2d)}
    for n, v in module.named_modules():
        if isinstance(v, TwinBatchNorm2d):
            v.indicator = indicator
    yield
    for n, v in module.named_modules():
        if isinstance(v, TwinBatchNorm2d):
            v.indicator = previous_state[n]
    _TwinBNCONTEXT.pop()

if __name__ == '__main__':
    unet = UNet()

    domainBN_unet = convert2TwinBN(unet)
    input = torch.randn(1, 3, 224, 224)
    output = domainBN_unet(input)
    print(output)

    # with switch_bn(self._model, 0):
    #     label_logits = self._model(torch.cat([labeled_image, labeled_image_tf]))
    # label_logits, label_tf_logits = torch.chunk(label_logits, 2)
    # with self._bn_context(self._model), switch_bn(self._model, 1):
    #     unlabeled_logits, unlabeled_tf_logits = \
    #         torch.split(self._model(torch.cat([unlabeled_image, unlabeled_image_tf], dim=0)),
    #                     [n_unl, n_unl], dim=0)

    # with switch_bn(self._model, 0):
    #     label_logits = self._model(torch.cat([labeled_image, labeled_image_tf]))
    # label_logits, label_tf_logits = torch.chunk(label_logits, 2)
    # with self._bn_context(self._model), switch_bn(self._model, 1):
    #     unlabeled_logits, unlabeled_tf_logits = \
    #         torch.split(self._model(torch.cat([unlabeled_image, unlabeled_image_tf], dim=0)),
    #                     [n_unl, n_unl], dim=0)
