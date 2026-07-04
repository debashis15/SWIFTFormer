import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from natsort import natsorted
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.PNG')


def is_image_file(filename):
    return filename.endswith(IMG_EXTENSIONS)


class DataLoaderTrain(Dataset):
    """Paired low/normal-light training loader.

    Expects directory layout:
        rgb_dir/low/*.png      (low-light inputs)
        rgb_dir/high/*.png     (normal-light targets)

    Applies random cropping (patch-based training) and geometric
    augmentation: random rotation (90/180/270) + horizontal flipping.
    """

    def __init__(self, rgb_dir, img_options=None):
        super().__init__()
        inp_files = natsorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = natsorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options or {}
        self.sizex = len(self.tar_filenames)
        self.ps = self.img_options.get('patch_size', 128)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_img = Image.open(self.inp_filenames[index_]).convert('RGB')
        tar_img = Image.open(self.tar_filenames[index_]).convert('RGB')

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]
        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # geometric augmentation: rotation + horizontal flip
        aug = random.randint(0, 7)
        if aug == 1:
            inp_img, tar_img = inp_img.flip(1), tar_img.flip(1)
        elif aug == 2:
            inp_img, tar_img = inp_img.flip(2), tar_img.flip(2)
        elif aug == 3:
            inp_img, tar_img = torch.rot90(inp_img, dims=(1, 2)), torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img, tar_img = torch.rot90(inp_img, dims=(1, 2), k=2), torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img, tar_img = torch.rot90(inp_img, dims=(1, 2), k=3), torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img, tar_img = torch.rot90(inp_img.flip(1), dims=(1, 2)), torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img, tar_img = torch.rot90(inp_img.flip(2), dims=(1, 2)), torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.basename(self.tar_filenames[index_]))[0]
        return tar_img, inp_img, filename


class DataLoaderVal(Dataset):
    """Paired validation / synthetic-benchmark loader (full images)."""

    def __init__(self, rgb_dir):
        super().__init__()
        inp_files = natsorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = natsorted(os.listdir(os.path.join(rgb_dir, 'high')))
        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]
        self.sizex = len(self.tar_filenames)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_img = TF.to_tensor(Image.open(self.inp_filenames[index_]).convert('RGB'))
        tar_img = TF.to_tensor(Image.open(self.tar_filenames[index_]).convert('RGB'))
        filename = os.path.splitext(os.path.basename(self.tar_filenames[index_]))[0]
        return tar_img, inp_img, filename


class DataLoaderTestUnpaired(Dataset):
    """Unpaired real-world loader (DICM / LIME / MEF / NPE): inputs only."""

    def __init__(self, inp_dir):
        super().__init__()
        inp_files = natsorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]
        self.sizex = len(self.inp_filenames)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index % self.sizex]
        inp = TF.to_tensor(Image.open(path_inp).convert('RGB'))
        filename = os.path.splitext(os.path.basename(path_inp))[0]
        return inp, filename


def get_training_data(rgb_dir, img_options):
    return DataLoaderTrain(rgb_dir, img_options)


def get_validation_data(rgb_dir):
    return DataLoaderVal(rgb_dir)


def get_test_unpaired_data(inp_dir):
    return DataLoaderTestUnpaired(inp_dir)
