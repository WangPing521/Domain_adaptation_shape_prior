import contextlib
from typing import Type, Union, List, Optional, Callable

import torch
from torch import Tensor, nn as nn
from torchvision.models import ResNet as _ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck

from arch.unet import _ConvBlock

_TwinBNCONTEXT = []


class ResNet(_ResNet):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000,
                 input_dim: int = 3,
                 zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation, norm_layer)
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

    def _forward_impl(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        x = self.fc(features)

        return x, features


class SimpleNet(nn.Module):

    def __init__(self, num_classes: int = 1000,
                 input_dim: int = 3, ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.conv1 = _ConvBlock(in_ch=input_dim, out_ch=16, )
        self.conv2 = _ConvBlock(in_ch=16, out_ch=64, )
        self.conv3 = _ConvBlock(in_ch=64, out_ch=96, )
        self.fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.conv3(x)
        feature = torch.flatten(self.avgpool(x), start_dim=1)
        return self.fc(feature), feature


class Projector(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.intermediate_dim = intermediate_dim

        self.projector = nn.Sequential(
            nn.Linear(input_dim, out_features=intermediate_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(intermediate_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        return self.projector(features)


def resnet18(input_dim, num_classes) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    """
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, input_dim=input_dim)


@contextlib.contextmanager
def switch_grad(module: nn.Module, enable=True):
    previous_grad = next(module.parameters()).requires_grad
    module.requires_grad_(enable)
    yield
    module.requires_grad_(previous_grad)


if __name__ == '__main__':
    net = resnet18(input_dim=1, num_classes=10)
    input_image = torch.randn(2, 3, 224, 224)
    output = net(input_image)
