import torch
import torch.nn as nn

# components in the architecture
def conv_layer(in_channels, out_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride = (1,1), padding = (0,0))

def deconv_layer(in_channels, out_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, residual=True):
        super(residual_block, self).__init__()
        self.conv1 = conv_layer(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class dilated_residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(dilated_residual_block, self).__init__()
        self.dilations = dilations
        self.conv = conv_layer(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)



    def forward(self, data_x):
       pass


# network in the paper
class Encoder_E():
    def __init__(self):
        pass
    def forward(self):
        pass

class Decoder_U():
    def __init__(self):
        pass
    def forward(self):
        pass

class Generator_t(nn.Module):
    pass

class DA_classifier(nn.Module):
    def __init__(self, channel_dim, num_classes):
        super(DA_classifier, self).__init__()
        self.conv = nn.Conv2d(channel_dim, num_classes, kernel_size=(1, 1), stride=(1, 1),
                  padding=(0, 0))


    def forward(self, input):
        output = self.conv(input)
        out_seg = output.resize_images(output, (256, 256))
        return out_seg
