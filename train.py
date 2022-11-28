import gc
import json
from pathlib import Path
from re import T

from data.paired_image_dataset import PairedImageDataset
from models.cyclegan import CycleGAN

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm


def train():
    # Constants
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 10
    LAMBDA = 0.5

    # Datasets
    # train_dataset = PairedImageDataset("/home/jyue86/Documents/cloudy-cycle-gans/data_small/train")
    # valid_dataset = PairedImageDataset("/home/jyue86/Documents/cloudy-cycle-gans/data_small/val")
    train_dataset = PairedImageDataset("/content/drive/MyDrive/data_small/train")
    valid_dataset = PairedImageDataset("/content/drive/MyDrive/data_small/val")
    train_loader = DataLoader(train_dataset, batch_size=8)
    valid_loader = DataLoader(valid_dataset, batch_size=8)

    # Models
    model = CycleGAN().to(device)

    # Optimizers
    optim_g = optim.Adam([{"params": model.genA.parameters()}, {"params": model.genB.parameters()}])
    optim_dA = optim.Adam(model.discA.parameters())
    optim_dB = optim.Adam(model.discB.parameters())

    train_loss = {
        "total loss": [],
        "genA loss": [],
        "genB loss": [],
        "identity loss": [],
        "disc loss": [],
        "cycle loss": []
    }
    val_loss = {
        "total loss": [],
        "genA loss": [],
        "genB loss": [],
        "identity loss": [],
        "disc loss": [],
        "cycle loss": []
    }
    start_epoch = 0

    checkpoint_path_str = "/content/drive/MyDrive/cyclegan/checkpoint_29.pt"
    if Path(checkpoint_path_str).exists():
        checkpoint = torch.load(checkpoint_path_str)
        start_epoch = checkpoint["epoch"] + 1
        EPOCHS += start_epoch
        model.genA.load_state_dict(checkpoint["genA_state_dict"])
        model.genB.load_state_dict(checkpoint["genB_state_dict"])
        model.discA.load_state_dict(checkpoint["discA_state_dict"])
        model.discB.load_state_dict(checkpoint["discB_state_dict"])
        optim_g.load_state_dict(checkpoint["optim_g_state_dict"])
        optim_dA.load_state_dict(checkpoint["optim_discA_state_dict"])
        optim_dB.load_state_dict(checkpoint["optim_discB_state_dict"])
        train_loss = checkpoint["train_loss"]
        val_loss = checkpoint["val_loss"]

    for e in tqdm(range(start_epoch, EPOCHS)):
        model.train()

        # Loss data
        avg_total_loss = 0
        avg_genA_loss = 0
        avg_genB_loss = 0
        avg_identity_loss = 0
        avg_disc_loss = 0
        avg_cycle_loss = 0

        for sunny_img, cloudy_img in train_loader:
            sunny_img, cloudy_img = sunny_img.to(device), cloudy_img.to(device)
            genB_loss, disA_loss1, disB_loss1, cyclic_loss1, identity_loss1, sim_loss1 = model.compute_lossA(sunny_img)
            genA_loss, disA_loss2, disB_loss2, cyclic_loss2, identity_loss2, sim_loss2 = model.compute_lossB(cloudy_img)

            # Backpropogate genA
            gen_loss = genA_loss + genB_loss + (identity_loss1 + identity_loss2) + (
                    cyclic_loss1 + cyclic_loss2) * LAMBDA
            gen_loss.backward(retain_graph=True)

            # Backpropogate discA
            lossA = disA_loss1 + disA_loss2
            lossA.backward(retain_graph=True)

            # Backpropogate discB
            lossB = disB_loss1 + disB_loss2
            lossB.backward(retain_graph=True)

            # Optimize
            optim_g.step()
            optim_dA.step()
            optim_dB.step()

            avg_total_loss += gen_loss.item()
            avg_genA_loss += genA_loss.item()
            avg_genB_loss += genB_loss.item()
            avg_identity_loss += (identity_loss1.item() + identity_loss2.item())
            avg_disc_loss += lossA.item() + lossB.item()
            avg_cycle_loss += (cyclic_loss1.item() + cyclic_loss2.item())

            sunny_img = sunny_img.detach().cpu()
            cloudy_img = cloudy_img.detach().cpu()
            del sunny_img, cloudy_img
            gc.collect()
            torch.cuda.empty_cache()

        train_loss["total loss"].append(avg_total_loss / len(train_loader))
        train_loss["genA loss"].append(avg_genA_loss / len(train_loader))
        train_loss["genB loss"].append(avg_genB_loss / len(train_loader))
        train_loss["identity loss"].append(avg_identity_loss / len(train_loader))
        train_loss["disc loss"].append(avg_disc_loss / len(train_loader))
        train_loss["cycle loss"].append(avg_cycle_loss / len(train_loader))

        avg_total_loss = 0
        avg_genA_loss = 0
        avg_genB_loss = 0
        avg_identity_loss = 0
        avg_disc_loss = 0
        avg_cycle_loss = 0

        model.eval()
        with torch.no_grad():
            for sunny_img, cloudy_img in valid_loader:
                sunny_img, cloudy_img = sunny_img.to(device), cloudy_img.to(device)
                genB_loss, disA_loss1, disB_loss1, cyclic_loss1, identity_loss1, sim_loss1 = model.compute_lossA(sunny_img)
                genA_loss, disA_loss2, disB_loss2, cyclic_loss2, identity_loss2, sim_loss2 = model.compute_lossB(cloudy_img)

                gen_loss = genA_loss + genB_loss + (identity_loss1 + identity_loss2) + (
                        cyclic_loss1 + cyclic_loss2) * LAMBDA
                lossA = disA_loss1 + disA_loss2
                lossB = disB_loss1 + disB_loss2

                avg_total_loss += gen_loss.item()
                avg_genA_loss += genA_loss.item()
                avg_genB_loss += genB_loss.item()
                avg_identity_loss += (identity_loss1.item() + identity_loss2.item())
                avg_disc_loss += lossA.item() + lossB.item()
                avg_cycle_loss += (cyclic_loss1.item() + cyclic_loss2.item())

                sunny_img = sunny_img.detach().cpu()
                cloudy_img = cloudy_img.detach().cpu()
                del sunny_img, cloudy_img
                gc.collect()
                torch.cuda.empty_cache()
        val_loss["total loss"].append(avg_total_loss / len(valid_loader))
        val_loss["genA loss"].append(avg_genA_loss / len(valid_loader))
        val_loss["genB loss"].append(avg_genB_loss / len(valid_loader))
        val_loss["identity loss"].append(avg_identity_loss / len(valid_loader))
        val_loss["disc loss"].append(avg_disc_loss / len(valid_loader))
        val_loss["cycle loss"].append(avg_cycle_loss / len(valid_loader))
    # state_dict_dir = Path("./results/")
    # if not state_dict_dir.exists():
    #   state_dict_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": e,
        "genA_state_dict": model.genA.state_dict(),
        "genB_state_dict": model.genB.state_dict(),
        "discA_state_dict": model.discA.state_dict(),
        "discB_state_dict": model.discB.state_dict(),
        "optim_g_state_dict": optim_g.state_dict(),
        "optim_discA_state_dict": optim_dA.state_dict(),
        "optim_discB_state_dict": optim_dB.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }, f"/content/drive/MyDrive/cyclegan/checkpoint_{e}.pt")
    # torch.save(model.genA.state_dict(), "./results/genA_weights.pt")
    # torch.save(model.genB.state_dict(), "./results/genB_weights.pt")
    # torch.save(model.discA.state_dict(), "./results/discA_weights.pt")
    # torch.save(model.discB.state_dict(), "./results/discB_weights.pt")

    # train_results_path = Path("./results/train_results.json")
    # valid_results_path = Path("./results/valid_results.json")
    # train_results_path.touch()
    # valid_results_path.touch()
    # train_json = json.dumps(train_loss, indent=4)
    # valid_json = json.dumps(val_loss, indent=4)

    # with open("./results/train_results.json", "w") as f:
    #   f.write(train_json)

    # with open("./results/valid_results.json", "w") as f:
    #   f.write(valid_json)


if __name__ == '__main__':
    train()
