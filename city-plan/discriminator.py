#!/bin/python
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Variable
from torch.nn.parameter import Parameter

class ConBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropuout=0, normalize=0, relu=0):
        super(ConBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if normalize == 1:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif normalize == 2:
            layers[0] = torch.nn.utils.spectral_norm(layers[0])
        elif normalize == 3:
            layers.append(nn.BatchNorm2d(out_channels, affine=True))

        if dropuout:
            layers.append(nn.Dropout(p=dropuout))

        if relu == 0:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif relu == 1:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.main(x)

# class MinibatchDiscrimination(nn.Module):
#     def __init__(self, in_features, out_features, kernel_dims, mean=False):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.kernel_dims = kernel_dims
#         self.mean = mean
#         self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
#         init.normal_(self.T, 0, 1)

#     def forward(self, x):
#         # x is NxA
#         # T is AxBxC
#         matrices = x.mm(self.T.view(self.in_features, -1))
#         matrices = matrices.view(-1, self.out_features, self.kernel_dims)

#         M = matrices.unsqueeze(0)  # 1xNxBxC
#         M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
#         norm = torch.abs(M - M_T).sum(3)  # NxNxB
#         expnorm = torch.exp(-norm)
#         o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
#         if self.mean:
#             o_b /= x.size(0) - 1

#         x = torch.cat([x, o_b], 1).unsqueeze(-1)
#         return x


# class Discriminator(nn.Module):
#     def __init__(self, classes_num: int,  cnn_list: list, batch_size: int, batch_kernel: int):
#         super(Discriminator, self).__init__()

#         self.main = nn.Sequential()
#         self.flatten = nn.Flatten()
#         self.mini_batch = MinibatchDiscrimination(
#                         225,
#                         112,
#                         batch_kernel,
#                         batch_size
#                     )
#         self.last = nn.Conv1d(337, classes_num, 1)
#         self.sigmoid = nn.Sigmoid()
#         for i, layer in enumerate(cnn_list):
#             in_features, kernal, stride, padding = layer
#             if i == len(cnn_list)-1:
#                 #self.main.add_module(
#                 #    "flatten",
#                 #    nn.Flatten()
#                 #)
#                 #self.main.add_module(
#                 #    "MiniBatchDiscrimination",
#                 #     MinibatchDiscrimination(
#                 #        cnn_list[-1][0],
#                 #        int(cnn_list[-1][0]/2),
#                 #        batch_kernel,
#                 #        batch_size
#                 #    )
#                 #)
#                 #self.main.add_module(
#                 #    "conv1_last",
#                 #    nn.Conv1d(cnn_list[-1][0]+int(cnn_list[-1][0]/2), classes_num, 1)
#                 #)
#                 #self.main.add_module("sigmoid", nn.Sigmoid())
#                 break
#             else:
#                 if i % 2 == 0:
#                     dropout= False
#                 else:
#                     dropout= False


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropuout=0, normalize=0, relu=0):
        super(DownSample, self).__init__()
        self.down = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        self.bottlencek = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        if normalize == 1:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        elif normalize == 2:
            self.bn = None
            self.bottlencek = torch.nn.utils.spectral_norm(self.conv1)
        elif normalize == 3:
            self.bn = nn.BatchNorm2d(out_channels, affine=True)
        
        if relu == 0:
            self.relu = nn.LeakyReLU(0.2, inplace=False)
        elif relu == 1:
            self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x = self.down(x)
        x = self.bottlencek(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

    
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


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

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


class Discriminator(torch.nn.Module):
    def __init__(self, channels, classes, develop=True, residual=False, down=False, minibatch=False):
        super().__init__()
        
        if develop:
            if residual and down:
                self.main_module = nn.Sequential(
                    # Image (Cx256x256)
                    DownSample(channels, 32, 4, 2, 1, 0, 1, 0),
                    ResidualBlock(32, 1, 0),

                    # Image (32x128x128)
                    DownSample(32, 64, 4, 2, 1, 0.5, 1, 0),
                    ResidualBlock(64, 1, 0),

                    # Image (64x64x64)
                    DownSample(64, 128, 4, 2, 1, 0, 1, 0),
                    ResidualBlock(128, 1, 0),

                    # Image (128x32x32)
                    DownSample(128, 256, 4, 2, 1, 0.5, 1, 0),
                    ResidualBlock(256, 1, 0),

                    # State (256x16x16)
                    DownSample(256, 512, 4, 2, 1, 0, 1, 0),
                    ResidualBlock(512, 1, 0),

                    # State (512x8x8)
                    DownSample(512, 1024, 4, 2, 1, 0.5, 1, 0),
                    ResidualBlock(1024, 1, 0))
                    # output of main module --> State (1024x4x4)

                self.output = nn.Sequential(
                    # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
                    nn.Conv2d(in_channels=1024, out_channels=classes, kernel_size=4, stride=1, padding=0))

            elif residual:
                self.main_module = nn.Sequential(
                    # Image (Cx256x256)
                    ConBlock(channels, 32, 4, 2, 1, 0, 1, 0),
                    ResidualBlock(32, 1, 0),

                    # Image (32x128x128)
                    ConBlock(32, 64, 4, 2, 1, 0.5, 1, 0),
                    ResidualBlock(64, 1, 0),

                    # Image (64x64x64)
                    ConBlock(64, 128, 4, 2, 1, 0, 1, 0),
                    ResidualBlock(128, 1, 0),

                    # Image (128x32x32)
                    ConBlock(128, 256, 4, 2, 1, 0.5, 1, 0),
                    ResidualBlock(256, 1, 0),

                    # State (256x16x16)
                    ConBlock(256, 512, 4, 2, 1, 0, 1, 0),
                    ResidualBlock(512, 1, 0),

                    # State (512x8x8)
                    ConBlock(512, 1024, 4, 2, 1, 0.5, 1, 0),
                    ResidualBlock(1024, 1, 0))
                    # output of main module --> State (1024x4x4)

                self.output = nn.Sequential(
                    # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
                    nn.Conv2d(in_channels=1024, out_channels=classes, kernel_size=4, stride=1, padding=0))
                    
            else:       
                self.main_module = nn.Sequential(
                    # Image (Cx256x256)
                    ConBlock(channels, 32, 4, 2, 1, 0, 1, 0),

                    # Image (32x128x128)
                    ConBlock(32, 64, 4, 2, 1, 0.5, 1, 0),

                    # Image (64x64x64)
                    ConBlock(64, 128, 4, 2, 1, 0, 1, 0),

                    # Image (128x32x32)
                    ConBlock(128, 256, 4, 2, 1, 0.5, 1, 0),

                    # State (256x16x16)
                    ConBlock(256, 512, 4, 2, 1, 0, 1, 0),

                    # State (512x8x8)
                    ConBlock(512, 1024, 4, 2, 1, 0.5, 1, 0)
                    # output of main module --> State (1024x4x4)
                )

                self.output = nn.Sequential(
                    # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
                    nn.Conv2d(in_channels=1024, out_channels=classes, kernel_size=4, stride=1, padding=0))
        else:
            self.main_module = nn.Sequential(
                # Image (Cx512x512)
                ConBlock(channels, 16, 4, 2, 1, 0, 1, 0),

                # Image (16x256x256)
                ConBlock(16, 32, 4, 2, 1, 0.5, 1, 0),

                # Image (32x128x128)
                 ConBlock(32, 64, 4, 2, 1, 0, 1, 0),

                # Image (64x64x64)
                 ConBlock(64, 128, 4, 2, 1, 0.5, 1, 0),

                # Image (128x32x32)
                 ConBlock(128, 256, 4, 2, 1, 0, 1, 0),

                # State (256x16x16)
                 ConBlock(256, 512, 4, 2, 1, 0.5, 1, 0),

                # State (512x8x8)
                 ConBlock(512, 1024, 4, 2, 1, 0, 1, 0),
                # output of main module --> State (1024x4x4)
            
            )
            self.output = nn.Sequential(
                # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
                nn.Conv2d(in_channels=1024, out_channels=classes, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)
