import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        """
        :param: n, Size of the patch. 70 is the default arg to reduce number of params
        """
        # Kernel size, strides, and padding values are fixed
        super(Discriminator, self).__init__()
        self.conv1 = PatchGANBlock(3, 64)
        self.conv2 = PatchGANBlock(64, 128)
        self.conv3 = PatchGANBlock(128, 64)
        self.conv4 = PatchGANBlock(64, 1)
        self.model = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4)

    def forward(self, x):
        """
        :param x: Dim size = (B, 3, 256, 256)
        :return: output = (B, 1, 14, 14)
        """
        batch_size = x.shape[0]
        # NOTE: I don't know what the fuck I'm doing. If anything goes wrong, change this first.
        # return torch.mean(self.model(x).squeeze().reshape(batch_size,-1),dim=1)
        # return torch.mean(self.model(x), dim=(1, 2, 3))
        output = self.model(x)
        return F.avg_pool2d(output, output.size()[2:]).squeeze()

class PatchGANBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PatchGANBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2)
        self.in1 = nn.InstanceNorm2d(output_channels)
        self.model = nn.Sequential(self.conv, self.in1)

    def forward(self, x):
        return F.leaky_relu(self.model(x))



if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 256, 256)
    model = Discriminator()
    output = model(dummy_input)
    print(output.shape)

    #assert output.shape == torch.Size([1, 1, 30, 30])
