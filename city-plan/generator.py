#!/bin/python
import torch
import torch.nn as nn
# Generator Code


class UpBlock(nn.Module):
     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, dropuout=0, normalize=0, relu=0):
         super(UpBlock, self).__init__()
         layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias)]

        if normalize == 0:
            layers.append(nn.BatchNorm2d(out_channels, affine=True))
        elif normalize == 1:
            layers[0] = torch.nn.utils.spectral_norm(layers[0])
        elif normalize == 2:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))

        if dropuout:
            layers.append(nn.Dropout(p=dropuout))

        if relu == 0:
            layers.append(nn.ReLU(True))
        elif relu == 1:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

         self.main = nn.Sequential(*layers)
   
     def forward(self, x):
         return self.main(x)


class UpPixelBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale, dropuout=0, normalize=0, relu=0):
        super(UpBlock, self).__init__()
        #X,C,H,W
        #C = C/scale^2
        #H = H*scale
        #W = W*scale
        self.up = nn.PixelShuffle(scale)
        self.bottlencek = nn.Conv2d(in_channels/(scale**2), out_channels, kernel_size=1, bias=False)

        if normalize == 0:
            self.bn = nn.BatchNorm2d(out_channels, affine=True)
        elif normalize == 1:
            self.bn = None
            self.bottlencek = torch.nn.utils.spectral_norm(self.conv1)
        elif normalize == 2:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        
        if dropuout:
            self.drop = nn.Dropout(p=dropuout)
        if relu == 0:
            self.relu = nn.ReLU(True)
        elif relu == 1:
            self.relu = nn.LeakyReLU(0.2, inplace=True)
   
     def forward(self, x):
        x = self.up(x)
        x = self.bottlencek(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, normalize=0, relu=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        if normalize == 0:
            self.bn1 = nn.InstanceNorm2d(out_channels, affine=True)
        elif normalize == 1:
            self.bn1 = None
            self.conv1 = torch.nn.utils.spectral_norm(self.conv1)
        elif normalize == 2:
            self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
        
        if relu == 0:
            self.relu = nn.LeakyReLU(0.2, inplace=True)
        elif relu == 1:
            self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        if normalize == 0:
            self.bn2 = nn.InstanceNorm2d(out_channels, affine=True)
        elif normalize == 1:
            self.bn2 = None
            self.conv2 = torch.nn.utils.spectral_norm(self.conv1)
        elif normalize == 2:
            self.bn2 = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out

class Generator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.develop:
            if config.residual and config.shuffle:
                self.main_module = nn.Sequential(
                    # Z latent (Z)
                    UpPixelBlock(config.latent, 1024, 4, config.dropfirst, config.normalize, config.activation),
                    ResidualBlock(1024, 1, 0),

                    # Z latent (1024x4x4)
                    UpPixelBlock(1024, 512, 2, config.dropsecond, config.normalize, config.activation),
                    ResidualBlock(512, 1, 0),

                    # Z latent (512x8x8)
                    UpPixelBlock(512, 256, 2, config.dropfirst, config.normalize, config.activation),
                    ResidualBlock(256, 1, 0),

                    # Z latent (256x16x16)
                    UpPixelBlock(256, 128, 2, config.dropsecond, config.normalize, config.activation),
                    ResidualBlock(128, 1, 0),

                    # Z latent (128x32x32)
                    UpPixelBlock(128, 64, 2, config.dropfirst, config.normalize, config.activation),
                    ResidualBlock(64, 1, 0),

                    # Z latent (64x64x64)
                    UpPixelBlock(64, 32, 2, config.dropsecond, config.normalize, config.activation),
                    ResidualBlock(32, 1, 0),

                    # Z latent (32x128x128)
                    nn.PixelShuffle(2),
                    nn.ConvTranspose2d(in_channels=32/(2**2), out_channels=config.channels, kernel_size=1, bias=False)
                    # output of main module --> Image (Cx256x256)
                )
            elif residual:
                self.main_module = nn.Sequential(
                    # Z latent (Z)
                    UpBlock(config.latent, 1024, 4, 1, 0, config.bias, config.dropfirst, config.normalize, config.activation),
                    ResidualBlock(1024, 1, 0),

                    # Z latent (1024x4x4)
                    UpBlock(1024, 512, 4, 1, 0, config.bias, config.dropsecond, config.normalize, config.activation),
                    ResidualBlock(512, 1, 0),

                    # Z latent (512x8x8)
                    UpBlock(512, 256, 4, 1, 0, config.bias, config.dropfirst, config.normalize, config.activation),
                    ResidualBlock(256, 1, 0),

                    # Z latent (256x16x16)
                    UpBlock(256, 128, 4, 1, 0, config.bias, config.dropsecond, config.normalize, config.activation),
                    ResidualBlock(128, 1, 0),

                    # Z latent (128x32x32)
                    UpBlock(128, 64, 4, 1, 0, config.bias, config.dropfirst, config.normalize, config.activation),
                    ResidualBlock(64, 1, 0),

                    # Z latent (64x64x64)
                    UpBlock(64, 32, 4, 1, 0, config.bias, config.dropsecond, config.normalize, config.activation),
                    ResidualBlock(32, 1, 0),

                    # Z latent (32x128x128)
                    nn.ConvTranspose2d(in_channels=32, out_channels=config.channels, kernel_size=4, stride=2, padding=1)
                    # output of main module --> Image (Cx256x256)
                )
            else:
                self.main_module = nn.Sequential(
                    # Z latent (Z)
                    UpBlock(config.latent, 1024, 4, 1, 0, config.bias, config.dropfirst, config.normalize, config.activation),
                
                    # Z latent (1024x4x4)
                    UpBlock(1024, 512, 4, 1, 0, config.bias, config.dropsecond, config.normalize, config.activation),
                    
                    # Z latent (512x8x8)
                    UpBlock(512, 256, 4, 1, 0, config.bias, config.dropfirst, config.normalize, config.activation),
                    
                    # Z latent (256x16x16)
                    UpBlock(256, 128, 4, 1, 0, config.bias, config.dropsecond, config.normalize, config.activation),
                    
                    # Z latent (128x32x32)
                    UpBlock(128, 64, 4, 1, 0, config.bias, config.dropfirst, config.normalize, config.activation),
                    
                    # Z latent (64x64x64)
                    UpBlock(64, 32, 4, 1, 0, config.bias, config.dropsecond, config.normalize, config.activation),
                    
                    # Z latent (32x128x128)
                    nn.ConvTranspose2d(in_channels=32, out_channels=config.channels, kernel_size=4, stride=2, padding=1)
                    # output of main module --> Image (Cx256x256)
                )
        else:
            self.main_module = nn.Sequential(
                # Z latent (Z)
                UpBlock(config.latent, 1024, 4, 1, 0, config.bias, config.dropfirst, config.normalize, config.activation),
                
                # Z latent (1024x4x4)
                UpBlock(1024, 512, 4, 1, 0, config.bias, config.dropsecond, config.normalize, config.activation),
                    
                # Z latent (512x8x8)
                UpBlock(512, 256, 4, 1, 0, config.bias, config.dropfirst, config.normalize, config.activation),
                    
                # Z latent (256x16x16)
                UpBlock(256, 128, 4, 1, 0, config.bias, config.dropsecond, config.normalize, config.activation),
                    
                # Z latent (128x32x32)
                UpBlock(128, 64, 4, 1, 0, config.bias, config.dropfirst, config.normalize, config.activation),
                    
                # Z latent (64x64x64)
                UpBlock(64, 32, 4, 1, 0, config.bias, config.dropsecond, config.normalize, config.activation),

                # Z latent (32x128x128)
                UpBlock(32, 16, 4, 1, 0, config.bias, config.dropfirst, config.normalize, config.activation),
                    
                # Z latent (16x256x256)
                nn.ConvTranspose2d(in_channels=16, out_channels=config.channels, kernel_size=4, stride=2, padding=1)
                # output of main module --> Image (Cx512x512)
            )
        
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

