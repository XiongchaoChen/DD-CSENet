import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from networks.SE import *
from utils.Utility import *
from models.utils import nmae

# -----------------------------------------
# ----------- (1) 2D UNet -----------------
# -----------------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=32, out_channels=1, depth=3, wf=5, padding=True,
                 norm='None', up_mode='upconv', residual=False):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.residual = residual

        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf+i), padding, norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf+i), up_mode,
                                            padding, norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=(1,1))

    def forward(self, x):
        input_ = x   # [B, 32, 384, 768]
        blocks = []

        # Contraction
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        # Expension
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        if self.residual:
            out = x + self.last(x)
        else:
            out = self.last(x)

        return out # [B, 32, 384, 768]

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=(3,3), padding=int(padding)))
        if norm == 'BN':
            block.append(nn.BatchNorm2d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm2d(out_size))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(out_size, out_size, kernel_size=(3,3), padding=int(padding)))
        if norm == 'BN':
            block.append(nn.BatchNorm2d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm2d(out_size))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=(2,2), stride=(2,2))
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=(1,1)))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()  # [384, 768]
        diff_z = (layer_height - target_size[0]) // 2  # floor division
        diff_y = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_z:(diff_z + target_size[0]), diff_y:(diff_y + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out



# -----------------------------------------
# ------------ (2) 2D ResUNet -------------
# -----------------------------------------
class ResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, norm='None'):
        super(ResUNet, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, n_filters, kernel_size=(3,3), padding=1, bias=True)

        # encode
        self.RDB1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.RDB2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.RDB3 = RDB(n_filters*1, n_denselayer, growth_rate, norm)

        # decode
        self.up2 = nn.ConvTranspose2d(n_filters*1, n_filters*1, kernel_size=(2,2), stride=(2,2))
        self.RDB_up2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)

        self.up1 = nn.ConvTranspose2d(n_filters*1, n_filters*1, kernel_size=(2,2), stride=(2,2))
        self.RDB_up1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)

        self.conv_out = nn.Conv2d(n_filters, out_channels, kernel_size=(1,1), padding=0, bias=True)

    def forward(self, x):  # [B, 32, 384, 768]
        # encode
        down1 = self.conv_in(x)  # [B, 32, 384, 768]
        RDB1 = self.RDB1(down1)

        down2 = F.avg_pool2d(RDB1, 2)  # [B, 32, 192, 384]
        RDB2 = self.RDB2(down2)

        down3 = F.avg_pool2d(RDB2, 2)  # [B, 32, 96, 192]
        RDB3 = self.RDB3(down3)

        # decode
        up2 = self.up2(RDB3)             # [B, 32, 192, 384]
        RDB_up2 = self.RDB_up2(up2 + RDB2)

        up1 = self.up1(RDB_up2)            # [B, 32, 384, 768]
        RDB_up1 = self.RDB_up1(up1 + RDB1)

        output = self.conv_out(RDB_up1)  # [16, 1, 16,16,16]

        return output


# Residual dense block
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm='None'):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, norm=norm))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=(1,1), padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)

        out = out + x # Residual
        return out


# Make Dense Block
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, norm='None'):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=(3,3), padding=1, bias=False)
        self.norm = norm
        self.bn = nn.BatchNorm2d(growthRate)


    def forward(self, x):
        out = self.conv(x)
        if self.norm == 'BN':
            out = self.bn(out)
        out = F.relu(out)

        out = torch.cat((x, out), 1)
        return out




# -----------------------------------------
# ------------- (3) 2D DuRDN --------------
# -----------------------------------------
class DuRDN(nn.Module):
    def __init__(self, in_channels=64, out_channels=32, n_filters=16, n_denselayer=3, growth_rate=16, norm='None'):
        super(DuRDN, self).__init__()

        self.conv_fuse = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0, bias=True)
        self.conv_in = nn.Conv2d(out_channels, n_filters, kernel_size=(3,3), padding=1, bias=True)

        # encode
        self.RDB1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE1 = ChannelSpatialSELayer2D(n_filters*1, norm='None')

        self.RDB2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE2 = ChannelSpatialSELayer2D(n_filters*1, norm='None')

        self.RDB3 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE3 = ChannelSpatialSELayer2D(n_filters*1, norm='None')

        # decode
        self.up2 = nn.ConvTranspose2d(n_filters*1, n_filters*1, kernel_size=(2,2), stride=(2,2))
        self.conv_up2 = nn.Conv2d(n_filters*2, n_filters*1, kernel_size=(1,1), padding=0, bias=False)
        self.RDB_up2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up2 = ChannelSpatialSELayer2D(n_filters*1, norm='None')

        self.up1 = nn.ConvTranspose2d(n_filters*1, n_filters*1, kernel_size=(2,2), stride=(2,2))
        self.conv_up1 = nn.Conv2d(n_filters*2, n_filters*1, kernel_size=(1,1), padding=0, bias=False)
        self.RDB_up1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up1 = ChannelSpatialSELayer2D(n_filters*1, norm='None')

        self.conv_out = nn.Conv2d(n_filters, out_channels, kernel_size=(1,1), padding=0, bias=True)

    def forward(self, x):  # [B, 32/64, 384, 768]
        # encode
        fuse = self.conv_fuse(x) # [B, 32, 384, 768]
        down1 = self.conv_in(fuse)  # [B, 16, 384, 768]
        RDB1 = self.RDB1(down1)
        SE1 = self.SE1(RDB1)

        down2 = F.avg_pool2d(SE1, 2)  # [B, 16, 192, 384]
        RDB2 = self.RDB2(down2)
        SE2 = self.SE2(RDB2)

        down3 = F.avg_pool2d(SE2, 2)  # [B, 16, 96, 192]
        RDB3 = self.RDB3(down3)
        SE3 = self.SE3(RDB3)

        # decode
        up2 = self.up2(SE3)             # [B, 16, 192, 384]
        up_conv2 = self.conv_up2(torch.cat((up2, SE2), 1))
        RDB_up2 = self.RDB_up2(up_conv2)
        SE_up2 = self.SE_up2(RDB_up2)

        up1 = self.up1(SE_up2)            # [B, 16, 384, 768]
        up_conv1 = self.conv_up1(torch.cat((up1, SE1), 1))
        RDB_up1 = self.RDB_up1(up_conv1)
        SE_up1 = self.SE_up1(RDB_up1)

        # output
        conv_out = self.conv_out(SE_up1)  # [B, 16, 384, 768]
        output = conv_out + fuse

        return output



# Gaussian Weight Initialization
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


