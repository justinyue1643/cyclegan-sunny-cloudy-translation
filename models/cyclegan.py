import torch
import torch.nn as nn
import torch.nn.functional as F

from generator import Generator
from discriminator import Discriminator

'''
    Idea:
    -------
        - Define 2 Generators
        - Define 2 Discriminators

        - forward pass produces image
'''
class CycleGAN(nn.Module):
    def __init__(self):
        pass
    def forward(self,x):
        pass


#### CONTENT SIMILARITY CODE ####
#################################
'''
    use pretrained VGG-11 Network

    (called L2 Loss)

    Idea:
        im = just does
        gen_img = generator(im)
        l2_loss = VGG-11(im) - VGG-11(generator)
'''