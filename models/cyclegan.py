import torch
import torch.nn as nn
import torch.nn.functional as F

from models.generator import Generator
from models.discriminator import Discriminator

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
        super(CycleGAN, self).__init__()

        vgg_pretrain = vgg11(VGG11_Weights.DEFAULT)
        self.vgg_pretrain = vgg_pretrain.features[:5].eval() # take only first few layers
        self.genA = Generator()
        self.genB = Generator()
        
        self.discA = Discriminator()
        self.discB = Discriminator()

    def forward(self,x):
        pass

    def train(self):
        self.genA.train()
        self.genB.train()
        self.discA.train()
        self.discB.train()

    def eval(self):
        self.genA.eval()
        self.genB.eval()
        self.discA.eval()
        self.discB.eval()

    def compute_lossA(self,xA):
        '''
        : params        xA      image from sunny dataset distribution

        : notes         MSE loss based on VGG texture from generated image and input image.
                        attempts to preserve the texture when generating an image.
        '''
        genB_xA = self.genB(xA)
        disB_genB = self.discB(genB_xA)
        genA_genB = self.genA(genB_xA)

        disA_xA = self.discA(xA)

        # GenAdv Loss
        genB_loss = F.binary_cross_entropy(F.sigmoid(disB_genB), torch.ones_like(disB_genB))

        disB_loss = F.binary_cross_entropy(F.sigmoid(disB_genB), torch.zeros_like(disB_genB))

        disA_loss = F.binary_cross_entropy(F.sigmoid(disA_xA), torch.ones_like(disA_xA))

        #cyclic loss
        cyclic_loss = F.l1_loss(genA_genB, xA)

        #identity loss
        identity_loss = F.l1_loss(genB_xA, xA)

        # l2 similarity
        sim_loss = F.mse_loss(self.vgg_pretrain(xA), self.vgg_pretrain(genB_xA))
        return genB_loss, disA_loss, disB_loss, cyclic_loss, identity_loss, sim_loss

    def compute_lossB(self,xB):
        '''
        : params        xB      image from cloudy dataset distribution

        : notes         MSE loss based on VGG texture from generated image and input image.
                        attempts to preserve the texture when generating an image.
        '''
        #xB is cloudy
        genA_xB = self.genA(xB) # converts cloudy to sunny
        disA_genA = self.discA(genA_xB)
        genB_genA = self.genB(genA_xB) # converts sunny back to cloudy
        disB_xB = self.discB(xB)

        #if disc = 1, then it's real
        #if disc = 0, then it's fake

        # GenAdv Loss
        genA_loss = F.binary_cross_entropy(F.sigmoid(disA_genA), torch.ones_like(disA_genA))

        disA_loss = F.binary_cross_entropy(F.sigmoid(disA_genA), torch.zeros_like(disA_genA))

        disB_loss = F.binary_cross_entropy(F.sigmoid(disB_xB), torch.ones_like(disB_xB))

        # Cyclic Loss
        cyclic_loss = F.l1_loss(genB_genA, xB)

        # Identity Loss
        identity_loss = F.l1_loss(genA_xB, xB)
        
        # l2_similarity loss
        sim_loss = F.mse_loss(self.vgg_pretrain(xB), self.vgg_pretrain(genA_xB))
        return genA_loss, disA_loss, disB_loss, cyclic_loss, identity_loss, sim_loss


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