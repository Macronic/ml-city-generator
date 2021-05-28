#!/bin/python
import torch
import torch.nn as nn

from easydict import EasyDict as edict
# Generator Code


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, dropuout=0, normalize=0, relu=0):
        super(UpBlock, self).__init__()

        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias))

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
        else:
            self.bn = None
        
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
            self.bn2 = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        residual = x
        if self.pad1:
            out = self.pad1(x)
        else:
            out = x

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

class Generator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = edict(config)
        config = self.config

        if config.develop:
            if config.down and config.residual:
                self.encoders = nn.Sequential(
                    # Image (Cx128x128)
                    DownSample(config.in_channels, 32, 4, 2, 1, config.dropfirst_encoder, 3, config.activation_encoder),
                    ResidualBlock(32, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (32x64x64)
                    DownSample(32, 64, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder),
                    ResidualBlock(64, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (64x32x32)
                    DownSample(64, 128, 4, 2, 1, config.dropfirst_encoder, config.normalize_encoder, config.activation_encoder),
                    ResidualBlock(128, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (128x16x16)
                    DownSample(128, 256, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder),
                    ResidualBlock(256, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (256x8x8)
                    DownSample(256, 512, 4, 2, 1, config.dropfirst_encoder, config.normalize_encoder, config.activation_encoder),
                    ResidualBlock(512, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (512x4x4)
                    DownSample(512, 512, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder),
                    ResidualBlock(512, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (512x2x2)
                    DownSample(512, 512, 4, 2, 1, config.dropfirst_encoder, 3, config.activation_encoder),
                    # Out Image (512x1x1)
                )
            elif config.residual:
                self.encoders = nn.Sequential(
                    # Image (Cx128x128)
                    DownConBlock(config.in_channels, 32, 4, 2, 1, config.dropfirst_encoder, 3, config.activation_encoder, config.padingtype_encoder),
                    ResidualBlock(32, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (32x64x64)
                    DownConBlock(32, 64, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    ResidualBlock(64, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (64x32x32)
                    DownConBlock(64, 128, 4, 2, 1, config.dropfirst_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    ResidualBlock(128, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (128x16x16)
                    DownConBlock(128, 256, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    ResidualBlock(256, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (256x8x8)
                    DownConBlock(256, 512, 4, 2, 1, config.dropfirst_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    ResidualBlock(512, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (512x4x4)
                    DownConBlock(512, 512, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    ResidualBlock(512, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                    # Image (512x2x2)
                    DownConBlock(512, 512, 4, 2, 1, config.dropfirst_encoder, 3, config.activation_encoder, config.padingtype_encoder)
                    # Out Image (512x1x1)
                )
            else:
                self.encoders = nn.Sequential(
                    # Image (Cx128x128)
                    DownConBlock(config.in_channels, 32, 4, 2, 1, config.dropfirst_encoder, 3, config.activation_encoder, config.padingtype_encoder),
                    
                    # Image (32x64x64)
                    DownConBlock(32, 64, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    
                    # Image (64x32x32)
                    DownConBlock(64, 128, 4, 2, 1, config.dropfirst_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    
                    # Image (128x16x16)
                    DownConBlock(128, 256, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    
                    # Image (256x8x8)
                    DownConBlock(256, 512, 4, 2, 1, config.dropfirst_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    
                    # Image (512x4x4)
                    DownConBlock(512, 512, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    
                    # Image (512x2x2)
                    DownConBlock(512, 512, 4, 2, 1, config.dropfirst_encoder, 3, config.activation_encoder, config.padingtype_encoder),
                    # Out Image (512x1x1)
                )

            if config.residual and config.shuffle:
                self.decoders = nn.Sequential(
                    # Image (512x1x1)
                    UpPixelBlock(512, 512, 2, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(512, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (1024x2x2)
                    UpPixelBlock(1024, 512, 2, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(512, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (1024x4x4)
                    UpPixelBlock(1024, 256, 2, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(256, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (512x8x8)
                    UpPixelBlock(512, 128, 2, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(128, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (256x16x16)
                    UpPixelBlock(256, 64, 2, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(64, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (128x32x32)
                    UpPixelBlock(128, 32, 2, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(32, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (64x64x64)
                    UpPixelBlock(64, config.out_channels, 2, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder)
                    # Out Image (Cx128x128)
                )
            elif config.residual:
                self.decoders = nn.Sequential(
                    # Image (512x1x1)
                    UpBlock(512, 512, 2, 1, 0, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(512, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (1024x2x2)
                    UpBlock(1024, 512, 4, 2, 1, config.bias_decoder, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(512, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (1024x4x4)
                    UpBlock(1024, 256, 4, 2, 1, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(256, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (512x8x8)
                    UpBlock(512, 128, 4, 2, 1, config.bias_decoder, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(128, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (256x16x16)
                    UpBlock(256, 64, 4, 2, 1, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(64, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (128x32x32)
                    UpBlock(128, 32, 4, 2, 1, config.bias_decoder, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder),
                    ResidualBlock(32, config.normalize_decoder, config.activation_decoder, config.padingtype_encoder),

                    # Image (64x64x64)
                    nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.ZeroPad2d((1,0,1,0)),
                        nn.Conv2d(64, config.out_channels, 4, padding=1)
                    )
                    #UpBlock(64, config.out_channels, 4, 2, 1, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder)
                    # Out Image (Cx128x128)
                )
            else:
                self.decoders = nn.Sequential(
                    # Image (512x1x1)
                    UpBlock(512, 512, 2, 1, 0, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder),
                    
                    # Image (1024x2x2)
                    UpBlock(1024, 512, 4, 2, 1, config.bias_decoder, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder),
                    
                    # Image (1024x4x4)
                    UpBlock(1024, 256, 4, 2, 1, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder),
                    
                    # Image (512x8x8)
                    UpBlock(512, 128, 4, 2, 1, config.bias_decoder, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder),
                    
                    # Image (256x16x16)
                    UpBlock(256, 64, 4, 2, 1, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder),
                    
                    # Image (128x32x32)
                    UpBlock(128, 32, 4, 2, 1, config.bias_decoder, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder),
                    
                    # Image (64x64x64)
                    UpBlock(64, config.out_channels, 4, 2, 1, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder)
                    # Out Image (Cx128x128)
                )

        else:
            self.encoders = nn.Sequential(
                # Image (Cx256x256)
                DownConBlock(config.in_channels, 32, 4, 2, 1, config.dropfirst_encoder, 3, config.activation_encoder, config.padingtype_encoder),
                    
                # Image (32x128x128)
                DownConBlock(32, 64, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    
                # Image (64x64x64)
                DownConBlock(64, 128, 4, 2, 1, config.dropfirst_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    
                # Image (128x32x32)
                DownConBlock(128, 256, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    
                # Image (256x16x16)
                DownConBlock(256, 512, 4, 2, 1, config.dropfirst_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    
                # Image (512x8x8)
                DownConBlock(512, 512, 4, 2, 1, config.dropsecond_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),

                # Image (512x4x4)
                DownConBlock(512, 512, 4, 2, 1, config.dropfirst_encoder, config.normalize_encoder, config.activation_encoder, config.padingtype_encoder),
                    
                # Image (512x2x2)
                DownConBlock(512, 512, 4, 2, 1, config.dropsecond_encoder, 3, config.activation_encoder, config.padingtype_encoder),
                # Out Image (512x1x1)
            )

            self.decoders = nn.Sequential(
                    # Image (512x1x1)
                    UpBlock(512, 512, 2, 1, 1, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder, config.padingtype_decoder),
                    
                    # Image (1024x2x2)
                    UpBlock(1024, 512, 4, 2, 1, config.bias_decoder, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder, config.padingtype_decoder),
                    
                    # Image (1024x4x4)
                    UpBlock(1024, 512, 4, 2, 1, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder, config.padingtype_decoder),
                    
                    # Image (1024x8x8)
                    UpBlock(1024, 256, 4, 2, 1, config.bias_decoder, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder, config.padingtype_decoder),
                    
                    # Image (512x16x16)
                    UpBlock(512, 128, 4, 2, 1, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder, config.padingtype_decoder),
                    
                    # Image (256x32x32)
                    UpBlock(256, 64, 4, 2, 1, config.bias_decoder, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder, config.padingtype_decoder),
                    
                    # Image (128x64x64)
                    UpBlock(128, 32, 4, 2, 1, config.bias_decoder, config.dropfirst_decoder, config.normalize_decoder, config.activation_decoder, config.padingtype_decoder),
                    
                    # Image (64x128x128)
                    UpBlock(64, config.out_channels, 4, 2, 1, config.bias_decoder, config.dropsecond_decoder, config.normalize_decoder, config.activation_decoder, config.padingtype_decoder)
                    # Out Image (Cx256x256)
                )
        
        self.output = nn.Tanh()

    def forward(self, x):
        skips_cons = []

        for encoder in self.encoders[:-1]:
            x = encoder(x)
            if self.config.residual and 'ResidualBlock' in str(encoder):
                skips_cons.append(x)
            elif self.config.residual == False:
                skips_cons.append(x)

        x = self.encoders[-1](x)

        skips_cons = list(reversed(skips_cons))
        decoders = self.decoders[:-1]
        
        conv = 0
        for decoder in decoders:
            x = decoder(x)
            if self.config.residual and 'ResidualBlock' in str(decoder):
                x = torch.cat((x, skips_cons[conv]), axis=1)
                conv += 1
            elif self.config.residual == False:
                x = torch.cat((x, skips_cons[conv]), axis=1)
                conv += 1

        x = self.decoders[-1](x)
        return self.output(x)

