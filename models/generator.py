import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.conv3 = nn.Conv2d()

        self.res1 = ResidualBlock()
        self.res2 = ResidualBlock()
        self.res3 = ResidualBlock()
        self.res4 = ResidualBlock()
        self.res5 = ResidualBlock()
        self.res6 = ResidualBlock()

        self.output_conv = nn.Conv2d(100, 3)

    def forward(self, x):
        return self.output_conv(x)


class ResidualBlock(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = self.fc2(output)
        output = torch.cat(x, output)
        return F.relu(output)
