#!/bin/python
import torch
import torch.nn as nn
# Generator Code

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropuout=False, bias=False, normalize=True):
        super(UpBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropuout:
            layers.append(nn.Dropout(p=0.5))
        #layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.ReLU(True))
        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x)

#class Generator(nn.Module):
#    def __init__(self, module_list: list):
#        super(Generator, self).__init__()
#        self.main = nn.Sequential()

 #       for i, layer in enumerate(module_list):
#            in_features, kernal, stride, padding = layer
#            if i == len(module_list)-1:
#                self.main.add_module(
#                    f"tanh",
#                    nn.Tanh()
#                )
#                break
#            else:
#                if i % 2 == 0:
#                    dropout= False
#                else:
#                    dropout= False
#                self.main.add_module(
#                    f"up_block_{i}",
#                    UpBlock(in_features, module_list[i+1][0], kernal, stride, padding, dropout)
#                )

#    def forward(self, x: torch.Tensor):
#        return self.main(x)

class Generator(torch.nn.Module):
    def __init__(self, channels, in_channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            # State (64x64x64)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),        

            # State (32x128x128)
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),
    
            # State (16x256x256)
            nn.ConvTranspose2d(in_channels=16, out_channels=channels, kernel_size=4, stride=2, padding=1)
            # output of main module --> Image (Cx512x512)
        )

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
