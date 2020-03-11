"""quick implementation of training the dropout U-Net"""

import json
import os
import sys

from tqdm import tqdm
import skimage.io
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from unet import UNet
import cli_args
import transforms
from dataset import BFNetDataset2D as Dataset
from dataset import fix_paths


device = "cuda" if torch.cuda.is_available() else "cpu"


transformer = torchvision.transforms.Compose([
    transforms.RandomHFlipPair(),
    transforms.RandomVFlipPair(),
    transforms.RandomRotate90Pair()
])


def test_output(model, dataset, epoch, n_t, output_path):
    """
    quick test of a MC dropout model on the first index of the dataset,
    saving the input, predictions and variance
    """
    model.module.eval()
    model.module.enable_dropout()
    with torch.no_grad():
        for signal, target in dataset:
            signal = signal.to(device)
            target = target.to(device)
            store = []
            # Monte Carlo dropout approximation of a Bayesian NN
            # multiple forward passes with different
            # dropout each time
            for t in range(n_t):
                output = model(signal.unsqueeze(0))
                store.append(output.squeeze())
            output_mean = torch.stack(store).mean(0).data.cpu().numpy()
            output_var = torch.stack(store).var(0).data.cpu().numpy()
            break  # not needed, but double make sure it exits after the first image
    skimage.io.imsave(
        f"{output_path}/output_signal.tif",
        signal.data.cpu().numpy(),
        check_contrast=False
    )
    skimage.io.imsave(
        f"{output_path}/output_target.tif",
        target.data.cpu().numpy(),
        check_contrast=False
    )
    skimage.io.imsave(
        f"{output_path}/output_mean_{epoch}.tif",
        output_mean,
        check_contrast=False
    )
    skimage.io.imsave(
        f"{output_path}/output_var_{epoch}.tif",
        output_var,
        check_contrast=False
    )



if __name__ == "__main__":

    if device == "cpu":
        print("==================================================")
        print("== WARNING: no GPU detected, running on the CPU ==")
        print("==================================================")


    args = cli_args.make_args()
    PAD = 0

    # train without MC dropout
    model = UNet(p_dropout=args.prop_dropout)
    train_dataframe = pd.read_csv(args.train_csv)
    valid_dataframe = pd.read_csv(args.validation_csv).sample(frac=0.2)
    if args.image_dir is not None:
        train_dataframe = fix_paths(train_dataframe, args.image_dir)
        valid_dataframe = fix_paths(valid_dataframe, args.image_dir)
    train_dataset = Dataset(train_dataframe, transform=transformer, pad=PAD)
    valid_dataset = Dataset(valid_dataframe, transform=transformer, pad=PAD)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True if device == "cuda" else False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True if device == "cuda" else False
    )
    datasets = {"train": train_dataset, "val": valid_dataset}
    dataloaders = {"train": train_dataloader, "val": valid_dataloader}
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    lowest_epoch_loss = np.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser)
    history = {"train": [], "val": []}
    os.makedirs(args.output_path, exist_ok=True)

    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch:02d}/{args.n_epochs}")
        print("="*15)
        # NOTE: have to do model.module.function() rather than model.function()
        #       due to wrapping the model with DataParallel. Not sure if this
        #       is just for {enable,disable}_dropout() or for train() and eval()
        for phase in ["train", "val"]:
            if phase == "train":
                model.module.train()
            if phase == "val":
                model.module.eval()
            running_loss = 0
            with torch.set_grad_enabled(phase == "train"):
                for signal, target in dataloaders[phase]:
                    signal = signal.to(device)
                    target = target.to(device)
                    optimiser.zero_grad()
                    if phase == "train":
                        # single forward pass
                        output = model(signal)
                    if phase == "val":
                        # evaluation, multiple forward passes and calculate
                        # loss on the mean image
                        store = []
                        for t in range(args.n_forward_passes):
                            output = model(signal)
                            store.append(output)
                        output = torch.stack(store).mean(0)
                    loss = criterion(output, target)
                    if phase == "train":
                        loss.backward()
                        optimiser.step()
                    running_loss += loss.item() * signal.size(0)
            epoch_loss = running_loss / len(datasets[phase])
            print(f"{phase} loss = {epoch_loss}")
            history[phase].append(epoch_loss)
            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_loss < lowest_epoch_loss:
                    print("lowest validation loss yet, saving model")
                    lowest_epoch_loss = epoch_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.output_path, f"{args.model_name}.pth")
                    )
        test_output(
            model, datasets["val"], epoch, args.n_forward_passes, args.output_path
        )
        history_path = os.path.join(args.output_path, f"{args.model_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)


