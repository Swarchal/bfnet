import argparse
import os
from collections import OrderedDict

import pandas as pd
import skimage.io
import torch

from dataset import BFNetDataset2D as Dataset
import model_utils
from unet import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD = 0


def is_distributed(state_dict):
    return all(k.startswith("module.") for k in state_dict.keys())


def check_state_dict(state_dict):
    """rename nn.DataParallel keys in state_dict if they're present"""
    if is_distributed(state_dict):
        # rename keys
        new_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_dict[new_key] = value
        state_dict = new_dict
    return state_dict


def load_weights(weight_path):
    state_dict = check_state_dict(
        torch.load(weight_path, map_location=DEVICE)
    )
    return state_dict


def load_dataset(dataset_path):
    dataframe = pd.read_csv(dataset_path)
    dataset = Dataset(dataframe, pad=PAD, return_filename=True)
    return dataset


def make_save_filename(input_filename, output_dir, label):
    ext = "." + input_filename.split('.')[-1]
    final_path = os.path.basename(input_filename).replace(ext, "")
    new_final_path = final_path + f"_{label}{ext}"
    return os.path.join(output_dir, new_final_path)


def main(weights, dataset, output_dir, t, dropout):
    if not os.path.isdir(output_dir):
        print(f"  ** {output_dir} not found, creating directory...")
        os.mkdir(output_dir)
    model = UNet(p_dropout=dropout)
    model = model.to(DEVICE)
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        for signal_image, _, signal_filename in dataset:
            # re-enable dropout each time as the last prediction requires
            # disabling dropout
            model.enable_dropout()
            store = []
            # add a batch dimension to single image input
            signal_image = signal_image.unsqueeze(0).to(DEVICE)
            for i in range(t):
                output = model(signal_image)
                store.append(output.squeeze())
            output_mean = torch.stack(store).mean(0).data.cpu().numpy()
            output_var = torch.stack(store).var(0).data.cpu().numpy()
            # make a prediction from the complete model without dropout
            model.disable_dropout()
            output_prediction = model(signal_image)
            output_prediction = output_prediction.squeeze().data.cpu().numpy()
            print(f"  ** saving prediction from {signal_filename}")
            skimage.io.imsave(
                make_save_filename(signal_filename, output_dir, "prediction"),
                output_prediction,
                check_contrast=False
            )
            skimage.io.imsave(
                make_save_filename(signal_filename, output_dir, "mean"),
                output_mean,
                check_contrast=False
            )
            skimage.io.imsave(
                make_save_filename(signal_filename, output_dir, "variance"),
                output_var,
                check_contrast=False
            )
            skimage.io.imsave(
                make_save_filename(signal_filename, output_dir, "signal"),
                signal_image.data.cpu().numpy(),
                check_contrast=False
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Use a trained F-Net model on a dataset in csv form to make predictions"
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        type=str,
        help="Path to the trained model weights (state_dict)."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Path to dataset csv file."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Path to directory in which to save the images."
    )
    parser.add_argument(
        "-t",
        "--forward_passes",
        type=int,
        default=50,
        help="Number of forward passes for MC dropout variance calculation."
    )
    parser.add_argument(
        "-p",
        "--p_dropout",
        type=float,
        default=0.1,
        help="Proportion of dropout in MC dropout variance calculation."
    )

    args = parser.parse_args()

    model_weights = load_weights(args.weight_path)
    dataset = load_dataset(args.dataset)
    main(
        weights=model_weights,
        dataset=dataset,
        output_dir=args.output_dir,
        t=args.forward_passes,
        dropout=args.p_dropout
    )

