"""
adapted from: https://github.com/usuyama/pytorch-unet
"""


import torch
import torch.nn as nn

import model_utils


class UNet(nn.Module):
    def __init__(self, n_class=1, p_dropout=0.1):
        super().__init__()
        self.p_dropout = p_dropout
        self.dconv_down1 = self.double_conv(1, 64)
        self.dconv_down2 = self.double_conv(64, 128)
        self.dconv_down3 = self.double_conv(128, 256)
        self.dconv_down4 = self.double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.dconv_up3 = self.double_conv(256 + 512, 256)
        self.dconv_up2 = self.double_conv(128 + 256, 128)
        self.dconv_up1 = self.double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.apply(model_utils.init_weights_)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.p_dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.p_dropout)
        )

    def train(self, mode=True):
        """override nn.module.train() to disable dropout modules during training"""
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.apply(model_utils.unset_dropout_)
        return self

    def eval(self, dropout=True):
        """override nn.module.eval() to enable dropout during evaluation"""
        self.train(False)
        self.training = False
        if dropout:
            self.apply(model_utils.set_dropout_)
        return self

    def enable_dropout(self):
        """enable dropout modules without touching other modules training flag"""
        self.apply(model_utils.set_dropout_)
        return self

    def disable_dropout(self):
        """disable dropout modules without touching other modules training flag"""
        self.apply(model_utils.unset_dropout_)
        return self

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out


class UNet3D(nn.Module):
    """NOTE: completely untested"""
    def __init__(self, n_class=1, p_dropout=0.1):
        super().__init__()
        self.p_dropout = p_dropout
        self.dconv_down1 = self.double_conv3D(1, 64)
        self.dconv_down2 = self.double_conv3D(64, 128)
        self.dconv_down3 = self.double_conv3D(128, 256)
        self.dconv_down4 = self.double_conv3D(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.dconv_up3 = self.double_conv3D(256 + 512, 256)
        self.dconv_up2 = self.double_conv3D(128 + 256, 128)
        self.dconv_up1 = self.double_conv3D(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.apply(model_utils.init_weights_)

    def double_conv3D(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.p_dropout),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.p_dropout)
        )

    def train(self, mode=True):
        """override nn.module.train() to disable dropout modules during training"""
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.apply(model_utils.unset_dropout_)
        return self

    def eval(self, dropout=True):
        """override nn.module.eval() to enable dropout during evaluation"""
        self.train(False)
        self.training = False
        if dropout:
            self.apply(model_utils.set_dropout_)
        return self

    def enable_dropout(self):
        """enable dropout modules without touching other modules training flag"""
        self.apply(model_utils.set_dropout_)
        return self

    def disable_dropout(self):
        """disable dropout modules without touching other modules training flag"""
        self.apply(model_utils.unset_dropout_)
        return self

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

