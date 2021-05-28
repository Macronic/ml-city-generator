#!/bin/python
import torch
import torch.nn as nn

from easydict import EasyDict as edict
# Discriminator Code


class DownConBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropuout=0, normalize=0, relu=0, padding_type=0):
        super(DownConBlock, self).__init__()
        layers = []
        if padding_type == 1:
            layers.append(nn.ReflectionPad2d(padding))
            padding = 0
        elif padding_type == 2:
            layers.append(nn.ReplicationPad2d(padding))
            padding = 0
        
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        
        if normalize == 0:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif normalize == 1:
            layers[0] = torch.nn.utils.spectral_norm(layers[0])
        elif normalize == 2:
            layers.append(nn.BatchNorm2d(out_channels, affine=True))

        if dropuout:
            layers.append(nn.Dropout(p=dropuout))

        if relu == 0:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif relu == 1:
            layers.append(nn.ReLU(True))

        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropuout=0, normalize=0, relu=0):
        super(DownSample, self).__init__()
        self.down = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        self.bottlencek = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        if normalize == 0:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        elif normalize == 1:
            self.bn = None
            self.bottlencek = torch.nn.utils.spectral_norm(self.conv1)
        elif normalize == 2:
            self.bn = nn.BatchNorm2d(out_channels, affine=True)
        else:
            self.bn = None

        if relu == 0:
            self.relu = nn.LeakyReLU(0.2, inplace=True)
        elif relu == 1:
            self.relu = nn.ReLU(True)
    
    def forward(self, x):
        x = self.down(x)
        x = self.bottlencek(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, normalize=0, relu=0, padding_type=0):
        super(ResidualBlock, self).__init__()
        if padding_type == 0:
            self.pad1 = None
            pad=1
        elif padding_type == 1:
            self.pad1 = nn.ReflectionPad2d(1)
            pad = 0
        elif padding_type == 2:
            self.pad1 = nn.ReplicationPad2d(1)
            pad = 0
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=pad)
        
        if normalize == 0:
            self.bn1 = nn.InstanceNorm2d(in_channels, affine=True)
        elif normalize == 1:
            self.bn1 = None
            self.conv1 = torch.nn.utils.spectral_norm(self.conv1)
        elif normalize == 2:
            self.bn1 = nn.BatchNorm2d(in_channels, affine=True)
        
        if relu == 0:
            self.relu = nn.LeakyReLU(0.2, inplace=True)
        elif relu == 1:
            self.relu = nn.ReLU(True)
        
        if padding_type == 0:
            self.pad2 = None
            pad=1
        elif padding_type == 1:
            self.pad2 = nn.ReflectionPad2d(1)
            pad = 0
        elif padding_type == 2:
            self.pad2 = nn.ReplicationPad2d(1)
            pad = 0

        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=pad)

        if normalize == 0:
            self.bn2 = nn.InstanceNorm2d(in_channels, affine=True)
        elif normalize == 1:
            self.bn2 = None
            self.conv2 = torch.nn.utils.spectral_norm(self.conv1)
        elif normalize == 2:
            self.bn2 = nn.BatchNorm2d(in_channels, affine=True)

    def forward(self, x):
        residual = x
        if self.pad1:
            out = self.pad1(x)

        out = self.conv1(out)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)
        
        if self.pad2:
            out = self.pad2(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class Discriminator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = edict(config)
        config = self.config

        if config.develop:
            if config.down and config.residual:
                self.main = nn.Sequential(
                    # Image (Cx128x128)
                    DownSample(config.in_channels, 32, 4, 2, 1, config.dropfirst, 3, config.activation),
                    ResidualBlock(32, config.normalize, config.activation, config.padingtype),

                    # Image (32x64x64)
                    DownSample(32, 64, 4, 2, 1, config.dropsecond, config.normalize, config.activation),
                    ResidualBlock(64, config.normalize, config.activation, config.padingtype),

                    # Image (64x32x32)
                    DownSample(64, 128, 4, 2, 1, config.dropfirst, config.normalize, config.activation),
                    ResidualBlock(128, config.normalize, config.activation, config.padingtype),

                    # Image (128x16x16)
                    DownSample(128, 256, 4, 2, 1, config.dropsecond, config.normalize, config.activation),
                    ResidualBlock(256, config.normalize, config.activation, config.padingtype),

                    # Image (256x8x8)
                    DownSample(256, 512, 4, 2, 1, config.dropfirst, config.normalize, config.activation),
                    ResidualBlock(512, config.normalize, config.activation, config.padingtype)
                )
                # Image (512x4x4)
                self.final = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
                # Out Image (Cx4x4)

            elif config.residual:
                self.main = nn.Sequential(
                    # Image (Cx128x128)
                    DownConBlock(config.in_channels, 32, 4, 2, 1, config.dropfirst, 3, config.activation, config.padingtype),
                    ResidualBlock(32, config.normalize, config.activation, config.padingtype),

                    # Image (32x64x64)
                    DownConBlock(32, 64, 4, 2, 1, config.dropsecond, config.normalize, config.activation, config.padingtype),
                    ResidualBlock(64, config.normalize, config.activation, config.padingtype),

                    # Image (64x32x32)
                    DownConBlock(64, 128, 4, 2, 1, config.dropfirst, config.normalize, config.activation, config.padingtype),
                    ResidualBlock(128, config.normalize, config.activation, config.padingtype),

                    # Image (128x16x16)
                    DownConBlock(128, 256, 4, 2, 1, config.dropsecond, config.normalize, config.activation, config.padingtype),
                    ResidualBlock(256, config.normalize, config.activation, config.padingtype),

                    # Image (256x8x8)
                    DownConBlock(256, 512, 4, 2, 1, config.dropfirst, config.normalize, config.activation, config.padingtype),
                    ResidualBlock(512, config.normalize, config.activation, config.padingtype),
                )
                # Image (512x4x4)
                self.final = nn.Sequential(
                        nn.ZeroPad2d((1,0,1,0)),
                        nn.Conv2d(512, 1, 4, padding=1, bias=False)
                )
                # Out Image (Cx4x4)
            else:
                self.main = nn.Sequential(
                    # Image (Cx128x128)
                    DownConBlock(config.in_channels, 32, 4, 2, 1, config.dropfirst, 3, config.activation, config.padingtype),
                    
                    # Image (32x64x64)
                    DownConBlock(32, 64, 4, 2, 1, config.dropsecond, config.normalize, config.activation, config.padingtype),
                    
                    # Image (64x32x32)
                    DownConBlock(64, 128, 4, 2, 1, config.dropfirst, config.normalize, config.activation, config.padingtype),
                    
                    # Image (128x16x16)
                    DownConBlock(128, 256, 4, 2, 1, config.dropsecond, config.normalize, config.activation, config.padingtype),
                    
                    # Image (256x8x8)
                    DownConBlock(256, 512, 4, 2, 1, config.dropfirst, config.normalize, config.activation, config.padingtype)
                )
                # Image (512x4x4)
                self.final = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
                # Out Image (Cx4x4)
        else:
            self.main = nn.Sequential(
                # Image (Cx256x256)
                DownConBlock(config.in_channels, 32, 4, 2, 1, config.dropfirst, 3, config.activation, config.padingtype),
                    
                # Image (32x128x128)
                DownConBlock(32, 64, 4, 2, 1, config.dropsecond, config.normalize, config.activation, config.padingtype),
                    
                # Image (64x64x64)
                DownConBlock(64, 128, 4, 2, 1, config.dropfirst, config.normalize, config.activation, config.padingtype),
                    
                # Image (128x32x32)
                DownConBlock(128, 256, 4, 2, 1, config.dropsecond, config.normalize, config.activation, config.padingtype),
                    
                # Image (256x16x16)
                DownConBlock(256, 512, 4, 2, 1, config.dropfirst, config.normalize, config.activation, config.padingtype),
                    
                # Image (512x8x8)
                DownConBlock(512, 512, 4, 2, 1, config.dropsecond, config.normalize, config.activation, config.padingtype)
            )
            # Image (512x4x4)
            self.final = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
            # Out Image (Cx4x4)
    
    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x = self.main(x)
        x = self.final(x)
        return x
