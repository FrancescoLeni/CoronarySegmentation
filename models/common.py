import torch

import torch.nn as nn
# import torchvision.transforms as transforms

from .models_blocks import CNeXtBlock, CNeXtStem, CNeXtDownSample, ResBlock, ConvNormAct, ResBlockDP, TransformerBlock, \
                           SPPF, C3, UnetBlock, UnetDown, UnetUpBlock, UnetBlock3D, UnetDown3D, UnetUpBlock3D
from .utility_blocks import SelfAttentionModule, PatchMerging, PatchExpanding, LayerNorm, SelfAttentionModuleFC, SelfAttentionModuleLin


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvNormAct(3, 64, 6, 2, 2)
        self.a = ConvNormAct(64, 128, 6, 2, 2)
        self.b = ConvNormAct(128, 256, 6, 2, 2)
        self.start_h = (80 - 64) // 2  # 8
        self.start_w = (80 - 64) // 2  # 8

    def forward(self,x):
        x = self.b(self.a(self.stem(x)))

        return x[:, :, self.start_h:self.start_h + 64, self.start_w:self.start_w + 64]


# ----------------------------------------------------------------------------------------------------------------------
#                   NUOVIIIIIIIIIII
# ----------------------------------------------------------------------------------------------------------------------


class UnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_ch = 2  # also graph dim
        self.ch_dims = [32, 64, 128, 256, 512]
        self.dropout = [0.05, 0.1, 0.2, 0.3, 0.5]
        assert (len(self.ch_dims) == 5)
        self.stem = UnetBlock(
            self.in_ch, self.ch_dims[0], self.dropout[0])
        self.down1 = UnetDown(
            self.ch_dims[0], self.ch_dims[1], self.dropout[1])
        self.down2 = UnetDown(
            self.ch_dims[1], self.ch_dims[2], self.dropout[2])
        self.down3 = UnetDown(
            self.ch_dims[2], self.ch_dims[3], self.dropout[3])
        self.down4 = UnetDown(
            self.ch_dims[3], self.ch_dims[4], self.dropout[4])

    def forward(self, x):     # ->  128x128x1
        x0 = self.stem(x)     # ->  128x128x32
        x1 = self.down1(x0)   # ->  64x64x64
        x2 = self.down2(x1)   # ->  32x32x128
        x3 = self.down3(x2)   # ->  16x16x256
        x4 = self.down4(x3)   # ->  8x8x512
        return x0, x1, x2, x3, x4


class UnetDecoder(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.ch_dims = [32, 64, 128, 256, 512]
        self.n_classes = n_classes
        assert (len(self.ch_dims) == 5)

        self.up1 = UnetUpBlock(
            self.ch_dims[4], self.ch_dims[3], dropout_p=0.0)
        self.up2 = UnetUpBlock(
            self.ch_dims[3], self.ch_dims[2], dropout_p=0.0)
        self.up3 = UnetUpBlock(
            self.ch_dims[2], self.ch_dims[1], dropout_p=0.0)
        self.up4 = UnetUpBlock(
            self.ch_dims[1], self.ch_dims[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ch_dims[0], self.n_classes, 1, 1)

    def forward(self, features):
        x0, x1, x2, x3, x4 = features

        x = self.up1(x4, x3)    # -> 16x16x256
        x = self.up2(x, x2)     # -> 32x32x128
        x = self.up3(x, x1)     # -> 64x64x64
        x = self.up4(x, x0)     # -> 128x128x32
        out = self.out_conv(x)  # -> 128x128xN
        return out


class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.name = 'Unet'

        self.encoder = UnetEncoder()
        self.decoder = UnetDecoder(n_classes)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)  # raw logit output
        return out


class UnetEncoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_ch = 2  # also graph dim
        self.ch_dims = [32, 64, 128, 256, 512]
        self.dropout = [0.05, 0.1, 0.2, 0.3, 0.5]
        assert (len(self.ch_dims) == 5)
        self.stem = UnetBlock3D(
            self.in_ch, self.ch_dims[0], self.dropout[0])
        self.down1 = UnetDown3D(
            self.ch_dims[0], self.ch_dims[1], self.dropout[1])
        self.down2 = UnetDown3D(
            self.ch_dims[1], self.ch_dims[2], self.dropout[2])
        self.down3 = UnetDown3D(
            self.ch_dims[2], self.ch_dims[3], self.dropout[3])
        self.down4 = UnetDown3D(
            self.ch_dims[3], self.ch_dims[4], self.dropout[4])

    def forward(self, x):     # ->  128x128x1
        x0 = self.stem(x)     # ->  128x128x32
        x1 = self.down1(x0)   # ->  64x64x64
        x2 = self.down2(x1)   # ->  32x32x128
        x3 = self.down3(x2)   # ->  16x16x256
        x4 = self.down4(x3)   # ->  8x8x512
        return x0, x1, x2, x3, x4


class UnetDecoder3D(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.ch_dims = [32, 64, 128, 256, 512]
        self.n_classes = n_classes
        assert (len(self.ch_dims) == 5)

        self.up1 = UnetUpBlock3D(
            self.ch_dims[4], self.ch_dims[3], dropout_p=0.0)
        self.up2 = UnetUpBlock3D(
            self.ch_dims[3], self.ch_dims[2], dropout_p=0.0)
        self.up3 = UnetUpBlock3D(
            self.ch_dims[2], self.ch_dims[1], dropout_p=0.0)
        self.up4 = UnetUpBlock3D(
            self.ch_dims[1], self.ch_dims[0], dropout_p=0.0)

        self.out_conv = nn.Conv3d(self.ch_dims[0], self.n_classes, 1, 1)

    def forward(self, features):
        x0, x1, x2, x3, x4 = features

        x = self.up1(x4, x3)    # -> 16x16x256
        x = self.up2(x, x2)     # -> 32x32x128
        x = self.up3(x, x1)     # -> 64x64x64
        x = self.up4(x, x0)     # -> 128x128x32
        out = self.out_conv(x)  # -> 128x128xN
        return out


class UNet3D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.name = 'Unet'

        self.encoder = UnetEncoder3D()
        self.decoder = UnetDecoder3D(n_classes)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)  # raw logit output
        return out


class UNetBig(nn.Module):
    def __init__(self, n_classes=2, features=[64, 128, 256, 512]):
        super().__init__()

        in_channels = 2
        self.name = 'UnetBig'
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.dw = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # stem
        self.stem = self._block(in_channels, features[0])

        # Encoder
        for feature in features:
            self.encoder.append(self._block(feature, feature))
            self.dw.append(nn.Conv2d(feature, 2*feature, 3, 2, 1))

        # Bottleneck
        self.bottleneck = self._block(features[-1]*2, features[-1]*2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._block(feature * 2, feature))

        # Final output
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        encodings = []

        x = self.stem(x)

        # Encoder
        for layer, dw in zip(self.encoder, self.dw):
            x = layer(x) + x
            encodings.append(x)
            x = dw(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            enc = encodings[-(i // 2 + 1)]
            x = torch.cat((enc, x), dim=1)
            x = self.decoder[i + 1](x)

        # Final output
        return self.final_conv(x)