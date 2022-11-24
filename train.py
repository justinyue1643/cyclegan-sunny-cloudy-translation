from data.paired_image_dataset import PairedImageDataset
from models.cyclegan import CycleGAN

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

def train():
    # Devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 1

    # Datasets
    train_dataset = PairedImageDataset("/home/jyue86/Documents/cloudy-cycle-gans/data_small/train")
    train_loader = DataLoader(train_dataset)

    # Models
    model = CycleGAN()

    # Optimizers
    optim_g = optim.Adam([{"params": model.genA.parameters()}, {"params": model.genB.parameters()}])
    optim_dA = optim.Adam(model.discA.parameters())
    optim_dB = optim.Adam(model.discB.parameters())

    for e in tqdm(range(EPOCHS)):
        model.train()

        for sunny_img, cloudy_img in train_loader:
            sunny_img, cloudy_img = sunny_img.to(device), cloudy_img.to(device)
            genB_loss, disA_loss1, disB_loss1, cyclic_loss1, identity_loss1, sim_loss1 = model.compute_lossA(sunny_img)
            genA_loss, disA_loss2, disB_loss2, cyclic_loss2, identity_loss2, sim_loss2 = model.compute_lossB(cloudy_img)

            # Update genA
            gen_loss = genA_loss + genB_loss + (identity_loss1 + identity_loss2) + (cyclic_loss1 + cyclic_loss2)
            gen_loss.backward(retain_graph=True)

            # Update discA
            lossA = disA_loss1 + disA_loss2
            lossA.backward(retain_graph=True)

            # Update discB
            lossB = disB_loss1 + disB_loss2
            lossB.backward(retain_graph=True)

            optim_g.step()
            optim_dA.step()
            optim_dB.step()

        model.eval()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()