from models.cyclegan import CycleGAN
from data.paired_image_dataset import PairedImageDataset

from matplotlib import pyplot as plt
import numpy as np
import torch


if __name__ == "__main__":
    checkpoint = torch.load("/content/drive/MyDrive/cyclegan/checkpoint_9.pt")
    model = CycleGAN()

    model.genA.load_state_dict(checkpoint["genA_state_dict"])
    model.genB.load_state_dict(checkpoint["genB_state_dict"])
    model.discA.load_state_dict(checkpoint["discA_state_dict"])
    model.discB.load_state_dict(checkpoint["discB_state_dict"])
    model.eval()

    train_dataset = PairedImageDataset("/content/drive/MyDrive/data_small/train")

    for i in range(10):
        sunny_img, cloudy_img = train_dataset[i]
        sunny_img, cloudy_img = sunny_img.unsqueeze(0), cloudy_img.unsqueeze(0)

        cloudy_output = model(sunny_img, False)
        sunny_output_from_cloudy = model(cloudy_output, True)

        sunny_output = model(cloudy_img, True)
        cloudy_output_from_sunny = model(sunny_output, False)

        cloudy_output = cloudy_output.squeeze().detach().numpy().clip(0, 1)
        cloudy_output = cloudy_output.transpose((1, 2, 0))

        sunny_output_from_cloudy = sunny_output_from_cloudy.squeeze().detach().numpy().clip(0, 1)
        sunny_output_from_cloudy = sunny_output_from_cloudy.transpose((1, 2, 0))

        sunny_output = sunny_output.squeeze().detach().numpy().clip(0, 1)
        sunny_output = sunny_output.transpose((1, 2, 0))

        cloudy_output_from_sunny = cloudy_output_from_sunny.squeeze().detach().numpy().clip(0, 1)
        cloudy_output_from_sunny = cloudy_output_from_sunny.transpose((1, 2, 0))

        row1 = np.hstack((sunny_img.squeeze().transpose((1,2,0)), cloudy_img.squeeze().transpose((1, 2, 0)), sunny_output))
        row2 = np.hstack((sunny_output_from_cloudy, cloudy_output, cloudy_output_from_sunny))

        final_output = np.vstack((row1, row2))
        plt.imsave(f"/content/drive/MyDrive/cyclegan/cycle_output/cycle_img{i}.jpg")

        # plt.imsave(f"/content/drive/MyDrive/cyclegan/cycle_output/fake_cloudy_img{i}.jpg", cloudy_output)
        # plt.imsave(f"/content/drive/MyDrive/cyclegan/cycle_output/fake_sunny_img{i}.jpg", sunny_output)
        # plt.imsave(f"/content/drive/MyDrive/cyclegan/cycle_output/fake_sunny_from_cloudy_img{i}.jpg", sunny_output_from_cloudy)
        # plt.imsave(f"/content/drive/MyDrive/cyclegan/cycle_output/fake_cloudy_from_sunny_img{i}.jpg", cloudy_output_from_sunny)

