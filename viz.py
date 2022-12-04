from models.cyclegan import CycleGAN
from data.paired_image_dataset import PairedImageDataset

from matplotlib import pyplot as plt
import torch


if __name__ == "__main__":
  checkpoint = torch.load("/content/drive/MyDrive/cyclegan/checkpoint_59.pt")
  model = CycleGAN()

  model.genA.load_state_dict(checkpoint["genA_state_dict"])
  model.genB.load_state_dict(checkpoint["genB_state_dict"])
  model.discA.load_state_dict(checkpoint["discA_state_dict"])
  model.discB.load_state_dict(checkpoint["discB_state_dict"])
  model.eval()

  train_dataset = PairedImageDataset("/content/drive/MyDrive/data_small/train")

  for i in range(10):
    sunny_img, cloudy_img = train_dataset[i]
    sunny_img = sunny_img.unsqueeze(0)
    cloudy_output = model(sunny_img, False).squeeze().detach().numpy().clip(0, 1)
    cloudy_output = cloudy_output.transpose((1, 2, 0))

    cloudy_img = cloudy_img.unsqueeze(0)
    sunny_output = (model(cloudy_img, True).squeeze().detach().numpy()).clip(0, 1)
    sunny_output = sunny_output.transpose((1, 2, 0))

    plt.imsave(f"/content/drive/MyDrive/cyclegan/output/fake_cloudy_img{i}.jpg", cloudy_output)
    plt.imsave(f"/content/drive/MyDrive/cyclegan/output/fake_sunny_img{i}.jpg", sunny_output)