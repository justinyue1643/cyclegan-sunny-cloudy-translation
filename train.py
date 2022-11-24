from data.paired_image_dataset import PairedImageDataset
from models.generator import Generator

import torch
from torch.utils.data import DataLoader

def train():
    # Devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 100

    # Datasets
    data_dir = "~/Documents/cloudy-cycle-gans/data_small"
    train_dataset = PairedImageDataset(f"{data_dir}/train")
    train_loader = DataLoader(train_dataset)

    # Models
    generator1 = Generator().to(device)
    generator2 = Generator().to(device)


    for e in range(EPOCHS):
        generator1.train()
        generator2.train()

        for sunny_img, cloudy_img in train_dataset:
            sunny_img, cloudy_img = sunny_img.to(device), cloudy_img.to(device)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()