# this file is going to implement OLVA paper, which does not exist reference code.

import typing as t
from contextlib import contextmanager

import torch
from torch import Tensor, nn

from arch.unet import UNet, _ConvBlock
from utils.utils import fix_all_seed_within_context


class VAEUNet(UNet):

    def __init__(self, input_dim=3, num_classes=1, max_channel=256, momentum=0.1):
        super().__init__(input_dim, num_classes, max_channel, momentum)
        self._Conv5 = nn.Sequential(
            self._Conv5,
            nn.Conv2d(in_channels=self.get_channel_dim("Conv5"),
                      out_channels=self.get_channel_dim("Conv5"),
                      kernel_size=(1, 1))
        )  # to enable negative values.

        self._Conv5_std = nn.Sequential(
            _ConvBlock(in_ch=self.get_channel_dim("Conv4"), out_ch=self.get_channel_dim("Conv5"),
                       momentum=momentum),
            nn.Conv2d(in_channels=self.get_channel_dim("Conv5"), out_channels=self.get_channel_dim("Conv5"),
                      kernel_size=(1, 1))  # to enable negative values.
        )
        self._enable_sampling = False

    def forward(self, x, **kwargs):
        e5_sampled, features = self.forward_encoder(x)
        return self.forward_decoder(e5_sampled, **features)

    def forward_encoder(self, x, until: str = None):
        e5, e5_logvar, features = self._forward_encoder(x, until=until)
        e5_sampled = e5
        if self._enable_sampling:
            e5_std = torch.exp(e5_logvar / 2)
            e5_sampled = e5 + e5_std * torch.randn_like(e5_std)
        self._e5, self._e5_sampled, self._e5_logvar = e5, e5_sampled, e5_logvar
        return e5_sampled, features

    def _forward_encoder(self, x, until: str = None) -> t.Union[Tensor, t.Tuple[Tensor, Tensor, t.Dict[str, Tensor]]]:
        if until:
            if until not in self.layer_dimension:
                raise KeyError(f"`return_until` should be in {', '.join(self.layer_dimension.keys())},"
                               f" given {until}  ")
            # encoding path
        e1 = self._Conv1(x)  # 16 224 224
        # e1-> Conv1
        if until == "Conv1":
            return e1

        e2 = self._max_pool1(e1)
        e2 = self._Conv2(e2)  # 32 112 112
        # e2 -> Conv2
        if until == "Conv2":
            return e2

        e3 = self._max_pool2(e2)
        e3 = self._Conv3(e3)  # 64 56 56
        # e3->Conv3
        if until == "Conv3":
            return e3

        e4 = self._max_pool3(e3)
        e4 = self._Conv4(e4)  # 128 28 28
        # e4->Conv4
        if until == "Conv4":
            return e4

        _e5 = self._max_pool4(e4)
        e5 = self._Conv5(_e5)  # 256 14 14
        e5_std = self._Conv5_std(_e5)  # 256 14 14
        return e5, e5_std, {"e4": e4, "e3": e3, "e2": e2, "e1": e1}

    def forward_decoder(self, e5, *, e4=None, e3=None, e2=None, e1=None, until: str = None) -> Tensor:
        # decoding + concat path
        d5 = self._Up5(e5)
        e4 = d5 if e4 is None else e4
        d5 = torch.cat((torch.zeros_like(e4), d5), dim=1)

        d5 = self._Up_conv5(d5)  # 128 28 28
        # d5->Up5+Up_conv5

        if until == "Up_conv5":
            return d5

        d4 = self._Up4(d5)
        e3 = d4 if e3 is None else e3
        d4 = torch.cat((torch.zeros_like(e3), d4), dim=1)
        d4 = self._Up_conv4(d4)  # 64 56 56

        if until == "Up_conv4":
            return d4

        # d4->Up4+Up_conv4

        d3 = self._Up3(d4)
        e2 = d3 if e2 is None else e2
        d3 = torch.cat((torch.zeros_like(e2), d3), dim=1)
        d3 = self._Up_conv3(d3)  # 32 112 112

        if until == "Up_conv3":
            return d3

        # d3->Up3+upconv3

        d2 = self._Up2(d3)
        e1 = d2 if e1 is None else e1
        d2 = torch.cat((torch.zeros_like(e1), d2), dim=1)
        d2 = self._Up_conv2(d2)  # 16 224 224

        if until == "Up_conv2":
            return d2

        # d2->up2+upconv2

        d1 = self._Deconv_1x1(d2)  # 4 224 224
        return d1

    @property
    def latent_code_mean(self):
        return self._e5

    @property
    def latent_code_sampled(self):
        return self._e5_sampled

    @property
    def latent_code_log_var(self):
        return self._e5_logvar

    @contextmanager
    def switch_sampling(self, *, enable: bool):
        prev_state = self._enable_sampling
        try:
            self._enable_sampling = enable
            yield
        finally:
            self._enable_sampling = prev_state


def unet2vaeunet(model: "UNet", *, seed=0) -> "VAEUNet":
    # this function convert unet to a variational based unet.
    with fix_all_seed_within_context(seed):
        return VAEUNet(
            input_dim=model._input_dim,
            num_classes=model._num_classes,
            max_channel=model._max_channel,
            momentum=model._momentum
        )
