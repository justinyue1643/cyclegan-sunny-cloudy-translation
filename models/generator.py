import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        # MISSING 

        # Conv-InstanceNorm-LeakyRelu (INPUT)
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1)
        self.in1   = nn.InstanceNorm2d(32)
        
        #residual blocks
        # input (BATCH, 32, 256, 256)
        self.res1 = ResidualBlock(32,64,5)
        self.res2 = ResidualBlock(64,128,5)
        self.res3 = ResidualBlock(128,128,5)
        self.res4 = ResidualBlock(128,64,5)
        self.res5 = ResidualBlock(64,32,5)
        self.res6 = ResidualBlock(32,3,5)
        self.seq_res = nn.Sequential([self.res1, self.res2, self.res3, self.res4, self.res5, self.res6])
        # output (BATCH, 3, 256, 256)
        # however, this only gives us weird values, so we want an image that goes from 0->1

        # Conv-InstanceNorm-LeakyRelu (OUTPUT)
        self.conv2 = nn.Conv2d(32,3,kernel_size=3,stride=1)
        self.in2   = nn.InstanceNorm2d(3)
        '''
            CycleGan Generator Architecture:
            --------------------------------
                => 2 stride 2 convolutions -----> downsample image size
                => residual blocks
                => 2 1/2 stride deconvolutions -------> upsample image size using half strides
                => instance normalization on each convolution
                        => as in each convolution in the paper is followed by a batchnorm2d

            Cloudy CycleGan Generator Architecture:
            ---------------------------------------
                => residual blocks
        '''
    def forward(self, x):
        '''
        :notes      cloudy-cyclegan adapts from cyclegan which adapts from AL-CGAN
        '''

        #nn.Tanh restricts values from 0->1, which makes it an "image"

        out = self.conv1(x)
        out = self.in1(out)
        out = F.leaky_relu(out,0.05)

        out = self.seq_res(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = F.leaky_relu(out,0.05)
        out = nn.Tanh(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size):
        #### DEFINITION BASED ON SUPPLEMENTARY MATERIAL OF NEURAL TRANSFER PAPER ####
        #############################################################################
        # 3x3 conv ==> Batch Norm ==> Relu twice
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=kernel_size,stride=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # leaky relu
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=kernel_size,stride=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # another leaky relu


    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out,0.05)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.leaky_relu(out,0.05)
        return out


# this is the residual connection block proposed by [43] of original cyclegan paper.
# see supplementary paper of the cyclegan paper which implements [44]
class ResidualConnection(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = self.fc2(output)
        output = torch.cat(x, output)
        return F.relu(output)
