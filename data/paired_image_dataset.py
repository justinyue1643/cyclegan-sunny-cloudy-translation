from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm


class PairedImageDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        if self.dir[-1] != "/":
            self.dir += "/"
        self.sunny_imgs = []
        self.cloudy_imgs = []
        self._load_imgs()

        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def __len__(self):
        if len(self.sunny_imgs) != len(self.cloudy_imgs):
            raise Exception
        return len(self.sunny_imgs)

    def __getitem__(self, idx):
        sunny_img = Image.open(self.sunny_imgs[idx])
        cloudy_img = Image.open(self.cloudy_imgs[idx])

        if self.sunny_imgs[idx][-3:] == "png":
            sunny_img = sunny_img.convert("RGB")
        if self.sunny_imgs[idx][-3] == "png":
            cloudy_img = cloudy_img.convert("RGB")

        return self.transform(sunny_img), self.transform(cloudy_img)

    def _load_imgs(self):
        sunny_dir = Path(self.dir + "/sunny/")
        cloudy_dir = Path(self.dir + "/cloudy/")

        for i in sunny_dir.rglob("*"):
            self.sunny_imgs.append(str(i.resolve()))

        for i in cloudy_dir.rglob("*"):
            self.cloudy_imgs.append(str(i.resolve()))

if __name__ == "__main__":
    data_dir = "~/Documents/cloudy-cycle-gans/data_small"
    train_dataset = PairedImageDataset("/home/jyue86/Documents/cloudy-cycle-gans/data_small/train")
    train_loader = DataLoader(train_dataset)

    for s, c in tqdm(train_loader):
        continue
    print("done")
