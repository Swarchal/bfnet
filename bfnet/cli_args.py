import argparse
import textwrap


def make_args():
    """create command line arguments"""
    parser = argparse.ArgumentParser(
        textwrap.dedent(
            """
            BFNet, a bayesiallen approximation for uncertainty measures in
            brightfield to fluorescence image inference.
            """
        )
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        help="path to csv file containing training image paths"
    )
    parser.add_argument(
        "--validation_csv",
        type=str,
        help="path to csv file containing validation image paths"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="directory containing images, this will be appended to the paths in the data csv"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="directory in which to store the results"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="learning rate for optimiser"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size"
    )
    parser.add_argument(
        "-t",
        "--n_forward_passes",
        type=int,
        default=20,
        help="number of forward passes for stochastic MC dropout during inference"
    )
    parser.add_argument(
        "-p",
        "--prop_dropout",
        type=float,
        default=0.1,
        help="proportion of dropout to apply during inference"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="fnet_model",
        help="what to call the model weights saved in --output_path"
    )
    return parser.parse_args()
