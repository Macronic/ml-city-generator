#!/bin/python
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Variable
from torch.nn.parameter import Parameter

class ConBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropuout=False, normalize=True):
        super(ConBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropuout:
            layers.append(nn.Dropout(p=0.5))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x)

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1).unsqueeze(-1)
        return x


class Discriminator(nn.Module):
    def __init__(self, classes_num: int,  cnn_list: list, batch_size: int, batch_kernel: int):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential()
        
        for i, layer in enumerate(cnn_list):
            in_features, kernal, stride, padding = layer
            if i == len(cnn_list)-1:
                self.main.add_module(
                    "flatten",
                    nn.Flatten()
                )
                self.main.add_module(
                    "MiniBatchDiscrimination",
                     MinibatchDiscrimination(
                        cnn_list[-1][0],
                        int(cnn_list[-1][0]/2),
                        batch_kernel,
                        batch_size
                    )
                )
                self.main.add_module(
                    "conv1_last",
                    nn.Conv1d(cnn_list[-1][0]+int(cnn_list[-1][0]/2), classes_num, 1)
                )
                self.main.add_module("sigmoid", nn.Sigmoid())
                break
            else:
                if i % 2 == 0:
                    dropout= True
                else:
                    dropout= True

                self.main.add_module(
                    f"conv_block_{i}",
                    ConBlock(in_features, cnn_list[i+1][0], kernal, stride, padding, dropout)
                )


    def forward(self, x):
        return self.main(x)


