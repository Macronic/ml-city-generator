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
        layers.append(nn.ReLU(True))
        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x)

class Generator(nn.Module):
    def __init__(self, module_list: list):
        super(Generator, self).__init__()
        self.main = nn.Sequential()

        for i, layer in enumerate(module_list):
            in_features, kernal, stride, padding = layer
            if i == len(module_list)-1:
                self.main.add_module(
                    f"tanh",
                    nn.Tanh()
                )
                break
            else:
                if i % 2 == 0:
                    dropout= True
                else:
                    dropout= True
                self.main.add_module(
                    f"up_block_{i}",
                    UpBlock(in_features, module_list[i+1][0], kernal, stride, padding, dropout)
                )

    def forward(self, x: torch.Tensor):
        return self.main(x)

