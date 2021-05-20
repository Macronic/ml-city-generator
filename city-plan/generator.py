#!/bin/python
import torch
import torch.nn as nn
# Generator Code

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, dropuout=0, normalize=0, relu=0):
        super(UpBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias)]

        if normalize == 1:
            layers.append(nn.BatchNorm2d(out_channels, affine=True))
        elif normalize == 2:
            layers[0] = torch.nn.utils.spectral_norm(layers[0])
        elif normalize == 3:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))

        if dropuout:
            layers.append(nn.Dropout(p=dropuout))

        if relu == 0:
            layers.append(nn.ReLU(inplace=False))
        elif relu == 1:
            layers.append(nn.LeakyReLU(0.2, inplace=False))

        #layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.ReLU(True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, normalize=0, relu=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        if normalize == 1:
            self.bn1 = nn.InstanceNorm2d(in_channels, affine=True)
        elif normalize == 2:
            self.bn1 = None
            self.conv1 = torch.nn.utils.spectral_norm(self.conv1)
        elif normalize == 3:
            self.bn1 = nn.BatchNorm2d(in_channels, affine=True)
        
        if relu == 0:
            self.relu = nn.LeakyReLU(0.2, inplace=False)
        elif relu == 1:
            self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        if normalize == 1:
            self.bn2 = nn.InstanceNorm2d(in_channels, affine=True)
        elif normalize == 2:
            self.bn2 = None
            self.conv2 = torch.nn.utils.spectral_norm(self.conv2)
        elif normalize == 3:
            self.bn2 = nn.BatchNorm2d(in_channels, affine=True)

        
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
    def __init__(self, channels, latent, develop=True, residual=False):
        super().__init__()

        if develop:
            if residual:
                self.main_module = nn.Sequential(
                    # Z latent (Z)
                    UpBlock(latent, 1024, 4, 1, 0, False, 0, 1, 0),
                    ResidualBlock(1024, 1, 0),

                    # Z latent (1024x4x4)
                    UpBlock(1024, 512, 4, 2, 1, False, 0.5, 1, 0),
                    ResidualBlock(512, 1, 0),

                    # Z latent (512x8x8)
                    UpBlock(512, 256, 4, 2, 1, False, 0, 1, 0),
                    ResidualBlock(256, 1, 0),

                    # Z latent (256x16x16)
                    UpBlock(256, 128, 4, 2, 1, False, 0.5, 1, 0),
                    ResidualBlock(128, 1, 0),

                    # Z latent (128x32x32)
                    UpBlock(128, 64, 4, 2, 1, False, 0, 1, 0),
                    ResidualBlock(64, 1, 0),

                    # Z latent (64x64x64)
                    UpBlock(64, 32, 4, 2, 1, False, 0.5, 1, 0),
                    ResidualBlock(32, 1, 0),

                    # Z latent (32x128x128)
                    nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, stride=2, padding=1)
                    # output of main module --> Image (Cx256x256)
                )
            else:
                self.main_module = nn.Sequential(
                    # Z latent (Z)
                    UpBlock(latent, 1024, 4, 2, 1, False, 0, 1, 0),
                
                    # Z latent (1024x4x4)
                    UpBlock(1024, 512, 4, 2, 1, False, 0.5, 1, 0),
                    
                    # Z latent (512x8x8)
                    UpBlock(512, 256, 4, 2, 1, False, 0, 1, 0),
                    
                    # Z latent (256x16x16)
                    UpBlock(256, 128, 4, 2, 1, False, 0.5, 1, 0),
                    
                    # Z latent (128x32x32)
                    UpBlock(128, 64, 4, 2, 1, False, 0, 1, 0),
                    
                    # Z latent (64x64x64)
                    UpBlock(64, 32, 4, 2, 1, False, 0.5, 1, 0),
                    
                    # Z latent (32x128x128)
                    nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, stride=2, padding=1)
                    # output of main module --> Image (Cx256x256)
                )
        else:
            self.main_module = nn.Sequential(
                # Z latent (Z)
                UpBlock(latent, 1024, 4, 2, 1, False, 0, 1, 0),
                
                # Z latent (1024x4x4)
                UpBlock(1024, 512, 4, 2, 1, False, 0, 1, 0),
                    
                # Z latent (512x8x8)
                UpBlock(512, 256, 4, 2, 1, False, 0, 1, 0),
                    
                # Z latent (256x16x16)
                UpBlock(256, 128, 4, 2, 1, False, 0, 1, 0),
                    
                # Z latent (128x32x32)
                UpBlock(128, 64, 4, 2, 1, False, 0, 1, 0),
                    
                # Z latent (64x64x64)
                UpBlock(64, 32, 4, 2, 1, False, 0, 1, 0),

                # Z latent (32x128x128)
                UpBlock(32, 16, 4, 2, 1, False, 0, 1, 0),
                    
                # Z latent (16x256x256)
                nn.ConvTranspose2d(in_channels=16, out_channels=channels, kernel_size=4, stride=2, padding=1)
                # output of main module --> Image (Cx512x512)
            )
        
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

