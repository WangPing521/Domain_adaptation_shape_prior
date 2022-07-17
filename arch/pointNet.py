import torch.nn as nn


class PointNet(nn.Module):
    def __init__(self, num_points=300, fc_inch=121, conv_inch=256, ext=False):
        super().__init__()
        self.num_points = num_points
        self.ReLU = nn.LeakyReLU(inplace=True)
        # Final convolution is initialized differently form the rest
        if ext:
            self.conv1 = nn.Conv2d(conv_inch, conv_inch * 2, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(conv_inch * 2, conv_inch, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(conv_inch, self.num_points, kernel_size=6)
        self.final_fc = nn.Linear(fc_inch, 3)
        self._ext = ext

    def forward(self, x):
        if self._ext:
            x = self.ReLU(self.conv1(x))
            x = self.ReLU(self.conv2(x))
        x = self.ReLU(self.final_conv(x))
        x = x.view(x.size(0), x.size(1), -1)
        x = self.final_fc(x)
        return x # [n, 300, 3]

import torch
if __name__ == '__main__':
    img = torch.randn(5, 256, 16, 16)
    pointnet = PointNet()
    out1 = pointnet(img)