from __future__ import division

import torch
import torch.nn as nn
import logging
from scipy.io import loadmat,savemat

from PIL import Image, ImageOps
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torchvision.transforms import Compose
from torchvision import transforms
import random
import numpy as np
import h5py
import cv2
from scipy.interpolate import interp1d
import torch.nn.functional as F

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def get_patch(rgb, hsi, patch_size, ix=-1, iy=-1):
    ih, iw, _ = rgb.shape
    ip = patch_size  # input_patch_size

    # randomly crop
    if ix == -1:
        ix = random.randrange(0, ih - ip + 1)
    if iy == -1:
        iy = random.randrange(0, iw - ip + 1)

    hsi_patch = hsi[ix: ix + ip, iy: iy + ip, :]
    rgb_patch = rgb[ix: ix + ip, iy: iy + ip, :]

    return rgb_patch, hsi_patch


def augment(rgb, hsi):

    if random.random() < 0.5:
        # Random vertical Flip
        rgb = rgb[:, :, ::-1].copy()
        hsi = hsi[:, :, ::-1].copy()

    if random.random() < 0.5:
        # Random horizontal Flip
        rgb = rgb[:, ::-1, :].copy()
        hsi = hsi[:, ::-1, :].copy()

    if random.random() < 0.5:
        # Random rotation
        rgb = np.rot90(rgb.copy(), axes=(1, 2))
        hsi = np.rot90(hsi.copy(), axes=(1, 2))

    return rgb, hsi




class HyperTrain_ICVL(Dataset):
    def __init__(self, args):
        super(HyperTrain_ICVL, self).__init__()

        self.mat_path = args.mat_path
        self.mat_names = os.listdir(self.mat_path)

        self.rgb_path = args.rgb_path

        self.patch_size = args.patch_size
        self.data_augmentation = args.augmentation

    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = loadmat(mat_name)
        hsi = mat['HyperImage']

        # rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4]+"_clean.png")
        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4]+".png") # patch
        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0

        h, w = hsi.shape[:2]
        rand_h = random.randint(0, h - self.patch_size)
        rand_w = random.randint(0, w - self.patch_size)
        hsi_patch = hsi[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size, :]  # (xx, xx, 61), in range [0, 1], float64
        rgb_patch = rgb[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size, :]  # (xx, xx, 3),  in range [0, 1], float64

        hsi_patch = np.transpose(hsi_patch, [2, 0, 1])  # [3, xx, xx]
        rgb_patch = np.transpose(rgb_patch, [2, 0, 1])  # [3, xx, xx]

        if self.data_augmentation:
            rgb_patch, hsi_patch = augment(rgb_patch, hsi_patch)

        hsi_patch = torch.from_numpy(hsi_patch.astype(np.float32)).contiguous()
        rgb_patch = torch.from_numpy(rgb_patch.astype(np.float32)).contiguous()

        return rgb_patch, hsi_patch


class HyperValid_ICVL(Dataset):
    def __init__(self, args):
        super(HyperValid_ICVL, self).__init__()

        self.mat_path = args.mat_path_valid
        self.mat_names = os.listdir(self.mat_path)

        self.rgb_path = args.rgb_path_valid
        self.rgb_names = os.listdir(self.rgb_path)


    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = loadmat(mat_name)
        hsi = mat['HyperImage']

        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4] + "_clean.png")
        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0
        
        h_hr, w_hr, _ = hsi.shape
        hsi = hsi[:h_hr // 4 * 4, :w_hr // 4 * 4, :]
        rgb = rgb[:h_hr // 4 * 4, :w_hr // 4 * 4, :]


        hsi = torch.from_numpy(hsi.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)).contiguous()


        # w_crop = 128
        # h_crop = 128
        # rgb = rgb[:, 500: 500 + w_crop, 500: 500 + h_crop]
        # hsi = hsi[:, 500: 500 + w_crop, 500: 500 + h_crop]

        return rgb, hsi, self.mat_names[index][:-4]
