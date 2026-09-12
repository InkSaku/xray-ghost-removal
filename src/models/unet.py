"""U-Net for ghost artifact removal.

Input: 2-channel tensor [current_image, previous_image]
Output: 1-channel cleaned image

Residual learning: the network predicts the ghost pattern to subtract
from the ENTIRE image. No masking - the ghost affects all pixels,
so the correction must work at any window width/center setting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class GhostUNet(nn.Module):
    """U-Net that predicts the full ghost pattern to subtract.

    The ghost from the previous exposure affects every pixel.
    The network learns: given the current (contaminated) image and
    the previous image, predict the residual ghost contribution.
    """

    def __init__(self, in_channels=2, base_filters=32):
        super().__init__()
        f = base_filters

        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)

        self.bottleneck = ConvBlock(f * 8, f * 16)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        self.out_conv = nn.Conv2d(f, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        current = x[:, 0:1]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self._up_cat(self.up4(b), e4))
        d3 = self.dec3(self._up_cat(self.up3(d4), e3))
        d2 = self.dec2(self._up_cat(self.up2(d3), e2))
        d1 = self.dec1(self._up_cat(self.up1(d2), e1))

        ghost = self.out_conv(d1)
        cleaned = current - ghost

        return cleaned, ghost

    def _up_cat(self, up, skip):
        if up.shape != skip.shape:
            up = F.interpolate(up, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)
