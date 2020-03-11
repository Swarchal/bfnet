"""
Custom pytorch datsets
"""

import os
import torch
import torchvision
import skimage
import skimage.io
import skimage.transform
import torch.nn.functional as F
import transforms


class BFNetDataset2D(torch.utils.data.Dataset):
    """
    Pytorch dataset for 2D images.

    Parameters:
    ----------
    dataframe: pandas.DataFrame
    transform: pytorch transform
    signal_col: string
        name of input image column in dataframe
    target_col: string
        name of target image column in dataframe
    pad: int
        number of zero-padding pixels added to each edge, i.e if pad=3
        then 3 pixels of value zero will be added to each border
    return_filename: bool
        if true then the filename of the input image will also be returned.
        this is useful during evaluation for saving the predictions with
        a filename based on the input filename
    """

    def __init__(
        self,
        dataframe,
        transform=None,
        signal_col="path_signal",
        target_col="path_target",
        pad=0,
        resize=None,
        return_filename=False
    ):
        super(BFNetDataset2D).__init__()
        self.dataframe = dataframe
        self.transform = transform
        self.signal_col = signal_col
        self.target_col = target_col
        self.pad = pad
        if isinstance(resize, int):
            resize = (resize, resize)
        self.resize=resize
        self.return_filename = return_filename

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path_signal_image = row[self.signal_col]
        path_target_image = row[self.target_col]
        image_signal = skimage.io.imread(path_signal_image).astype("float32")
        image_signal = skimage.img_as_float(image_signal).astype("float32")
        image_signal = (image_signal - image_signal.mean()) / image_signal.std()
        image_target = (skimage.io.imread(path_target_image) / 2**16).astype("float32")
        if self.resize is not None:
            image_signal = skimage.transform.resize(image_signal, self.resize, anti_aliasing=True).astype("float32")
            image_target = skimage.transform.resize(image_target, self.resize, anti_aliasing=True).astype("float32")
        if self.pad > 0:
            image_signal = F.pad(image_signal, (self.pad, self.pad, self.pad, self.pad))
            image_target = F.pad(image_target, (self.pad, self.pad, self.pad, self.pad))
        if self.transform:
            # should the target image always be subjected to the same
            # transformation as the signal image??
            #   - yes for rotations
            #   - not for intensity normalisation
            #   - separate signal and target transform objects?
            #       - how to keep them equal for stochastic transforms such as
            #         random rotations?
            image_signal, image_target = self.transform((image_signal, image_target))
        image_signal = torchvision.transforms.functional.to_tensor(image_signal)
        image_target = torchvision.transforms.functional.to_tensor(image_target)
        if self.return_filename:
            return image_signal, image_target, path_signal_image
        else:
            return image_signal, image_target


class BFNetDataset3D(torch.utils.data.Dataset):
    """
    Pytorch dataset for 3D images.

    Parameters:
    ----------
    dataframe: pandas.DataFrame
    transform: pytorch transform
    signal_col: string
        name of input image column in dataframe
    target_col: string
        name of target image column in dataframe
    """

    def __init__(
        self,
        dataframe,
        transform=None,
        signal_col="path_signal",
        target_col="path_target",
    ):
        super(BFNetDataset3D).__init__()
        self.dataframe = dataframe
        self.transform = transform
        self.signal_col = signal_col
        self.target_col = target_col
        raise NotImplementedError("not made this yet")

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        # TODO: make this return z-stacks
        pass


def fix_paths(dataframe, image_dir):
    """
    change the root directory for the image paths.
    This is useful if mounting data in container volumes
    where the path will change from that of the host system

    Paramerers:
    -----------
    dataframe: pd.dataframe
    image_dir: str
        the new root directory which will replace everything up
        until the final image url
    Returns:
    --------
    pd.dataframe
    """
    image_dir = image_dir
    def fix(x):
        return os.path.join(image_dir, os.path.basename(x))
    dataframe["path_signal"] = dataframe["path_signal"].apply(fix)
    dataframe["path_target"] = dataframe["path_target"].apply(fix)
    return dataframe

