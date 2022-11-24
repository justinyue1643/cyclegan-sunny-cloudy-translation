import torch
import torch.nn as nn
import torch.nn.functional as F

from generator import Generator
from discriminator import Discriminator

from torchvision.models import vgg11, VGG11_Weights
'''
    Idea:
    -------
        - Define 2 Generators
        - Define 2 Discriminators

        - forward pass produces image

        - Gen A takes an image from Class B and generates something from class A
        - In Layman's terms: Take cloudy image => generate sunny image

        - Gen B takes generated image from class A and re-generates image into class B
        - in Layman's terms: Take converted sunny image => reconvert back to cloudy image

        - forward_cycle_loss = GenB(GenA(x)) - x (L1 loss)
        - backward_cycle_loss = GenA(GenB(y)) - y (L1 loss)

    Loss:
    --------
    L(G,F,D_x,D_y) = L_GAN(G,D_y,X,Y) + L_GAN(F,D_x,Y,X)
                     + lambda*L_cyc(G,F)
                     +  gamma*L_sim(G)
'''
class CycleGAN(nn.Module):
    def __init__(self):

        vgg_pretrain = vgg11(VGG11_Weights.DEFAULT)
        self.vgg_pretrain = vgg_pretrain.features[:5].eval() # take only first few layers
        self.genA = Generator()
        self.genB = Generator()
        pass
    def forward(self,x):
        pass


    
    def compute_l2loss(self,x):
        '''
        : params

        : notes         MSE loss based on VGG texture from generated image and input image.
                        attempts to preserve the texture when generating an image.
        '''
        l2 = torch.mean((self.vgg_pretrain(x) - self.vgg_pretrain(self.genA(x)))**2)
        return l2


#### CONTENT SIMILARITY CODE ####
#################################
'''
    use pretrained VGG-11 Network => outputs textures

    (called L2 Loss)

    Idea:
        im = just exists
        gen_img = generator(im)
        l2_loss = VGG-11(im) - VGG-11(generator)
'''