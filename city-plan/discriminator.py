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
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
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


#class Discriminator(nn.Module):
#    def __init__(self, classes_num: int,  cnn_list: list, batch_size: int, batch_kernel: int):
#        super(Discriminator, self).__init__()
#
#        self.main = nn.Sequential()
#        self.flatten = nn.Flatten()
#        self.mini_batch = MinibatchDiscrimination(
#                        225,
#                        112,
#                        batch_kernel,
#                        batch_size
#                    )
#        self.last = nn.Conv1d(337, classes_num, 1)
#        self.sigmoid = nn.Sigmoid()
#        for i, layer in enumerate(cnn_list):
#            in_features, kernal, stride, padding = layer
#            if i == len(cnn_list)-1:
                #self.main.add_module(
                #    "flatten",
                #    nn.Flatten()
                #)
                #self.main.add_module(
                #    "MiniBatchDiscrimination",
                #     MinibatchDiscrimination(
                #        cnn_list[-1][0],
                #        int(cnn_list[-1][0]/2),
                #        batch_kernel,
                #        batch_size
                #    )
                #)
                #self.main.add_module(
                #    "conv1_last",
                #    nn.Conv1d(cnn_list[-1][0]+int(cnn_list[-1][0]/2), classes_num, 1)
                #)
                #self.main.add_module("sigmoid", nn.Sigmoid())
#                break
#            else:
#                if i % 2 == 0:
#                    dropout= False
#                else:
#                    dropout= False
#
#                self.main.add_module(
#                    f"conv_block_{i}",
#                    ConBlock(in_features, cnn_list[i+1][0], kernal, stride, padding, dropout)
#                )
#
#
#    def forward(self, x):
#        #print(x.shape)
#       x = self.main(x)
        #x = self.flatten(x)
        #x = self.mini_batch(x)
        #x = self.last(x)
        #x = self.sigmoid(x)
#        return x

class Discriminator(torch.nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx512x512)
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Image (16x256x256)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Image (32x128x128)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Image (64x64x64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Image (128x32x32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)
