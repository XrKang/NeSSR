from __future__ import division

import torch
import torch.nn as nn
import logging
from scipy.io import loadmat,savemat

from PIL import Image, ImageOps
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision.transforms import Compose
from torchvision import transforms
import random
import numpy as np
import h5py
import cv2
from scipy.interpolate import interp1d
import torch.nn.functional as F



def down_scale_bic(hsi_hr, scale):
    hsi_hr = torch.from_numpy(hsi_hr.astype(np.float32)).contiguous()

    hsi_patch = torch.unsqueeze(hsi_hr, dim=0)
    hsi_patch_down = F.interpolate(hsi_patch, scale_factor=1 / scale, mode='bicubic')

    hsi_patch = torch.squeeze(hsi_patch, dim=0)


def get_patch(rgb, hsi_lr, hsi_hr, patch_size):
    _, ih, iw = hsi_lr.shape
    ip = patch_size  # input_patch_size
    ix = random.randrange(0, int(ih - ip))
    iy = random.randrange(0, int(iw - ip))

    rgb_patch    = rgb[:, ix: ix + ip, iy: iy + ip]
    hsi_lr_patch = hsi_lr[:, ix: ix + ip, iy: iy + ip]
    hsi_hr_patch = hsi_hr[:, ix: ix + ip, iy: iy + ip]

    return rgb_patch, hsi_lr_patch, hsi_hr_patch



def augment_triple(rgb, hsi_lr, hsi_hr):

    if random.random() < 0.5:
        # Random vertical Flip
        rgb = rgb[:, :, ::-1].copy()
        hsi_lr = hsi_lr[:, :, ::-1].copy()
        hsi_hr = hsi_hr[:, :, ::-1].copy()

    if random.random() < 0.5:
        # Random horizontal Flip
        rgb = rgb[:, ::-1, :].copy()
        hsi_lr = hsi_lr[:, ::-1, :].copy()
        hsi_hr = hsi_hr[:, ::-1, :].copy()

    if random.random() < 0.5:
        # Random rotation
        rgb = np.rot90(rgb.copy(), axes=(1, 2))
        hsi_lr = np.rot90(hsi_lr.copy(), axes=(1, 2))
        hsi_hr = np.rot90(hsi_hr.copy(), axes=(1, 2))

    return rgb, hsi_lr, hsi_hr


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


class HyperTrain_NTIRE22(Dataset):
    def __init__(self, args):
        super(HyperTrain_NTIRE22, self).__init__()

        self.mat_path = args.mat_path
        self.mat_names = os.listdir(self.mat_path)

        self.rgb_path = args.rgb_path

        self.scale = args.scale
        self.patch_size = args.patch_size
        self.data_augmentation = args.augmentation



    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        # full HSI
        # with h5py.File(mat_name, 'r') as mat:
        #     hsi = np.float32(np.array(mat['cube']))  # (31, 512, 482))
        # hsi = np.transpose(hsi, [2, 1, 0])  # (482,512, 31)

        # patch
        mat = loadmat(mat_name)
        hsi = mat['HyperImage']

        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4]+".jpg")
        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0        # (482, 512, 3)

        # crop edge
        h_hr, w_hr, c = hsi.shape
        hsi = hsi[:h_hr // self.scale * self.scale, :w_hr // self.scale * self.scale, :]
        rgb = rgb[:h_hr // self.scale * self.scale, :w_hr // self.scale * self.scale, :]

        # # crop patch
        # h, w = hsi.shape[:2]
        # rand_h = random.randint(0, h - self.patch_size)
        # rand_w = random.randint(0, w - self.patch_size)
        # hsi_patch = hsi[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size,
        #             :]  # (xx, xx, 31), in range [0, 1], float32
        # rgb_patch = rgb[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size,
        #             :]  # (xx, xx, 3),  in range [0, 1], float32

        hsi_patch = np.transpose(hsi, [2, 0, 1])  # [3, xx, xx]
        rgb_patch = np.transpose(rgb, [2, 0, 1])  # [3, xx, xx]

        # data augmentation
        if self.data_augmentation:
            rgb_patch, hsi_patch = augment(rgb_patch, hsi_patch)

        # to tensor
        hsi_patch = torch.from_numpy(hsi_patch.astype(np.float32)).contiguous()
        rgb_patch = torch.from_numpy(rgb_patch.astype(np.float32)).contiguous()

        # downscaling
        hsi_patch = torch.unsqueeze(hsi_patch, dim=0)
        hsi_patch_down = F.interpolate(hsi_patch, scale_factor=1/self.scale, mode='bicubic')
        hsi_patch_down = F.interpolate(hsi_patch_down, scale_factor=self.scale, mode='bicubic')
        hsi_patch = torch.squeeze(hsi_patch, dim=0)
        hsi_patch_down = torch.squeeze(hsi_patch_down, dim=0)

        # # crop patch
        rgb_patch, hsi_patch_down, hsi_patch = get_patch(rgb_patch, hsi_patch_down, hsi_patch, self.patch_size)

        return rgb_patch, hsi_patch_down, hsi_patch


class HyperValid_NTIRE22(Dataset):
    def __init__(self, args):
        super(HyperValid_NTIRE22, self).__init__()

        self.mat_path = args.mat_path_valid
        self.mat_names = os.listdir(self.mat_path)

        self.rgb_path = args.rgb_path_valid

        self.scale = args.scale

    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        with h5py.File(mat_name, 'r') as mat:
            hsi = np.float32(np.array(mat['cube']))  # (31, 512, 482))
        hsi = np.transpose(hsi, [2, 1, 0])  # (482,512, 31)

        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4] + ".jpg")
        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0  # (482,512, 3)

        # crop edge
        h_hr, w_hr, c = hsi.shape
        hsi = hsi[:h_hr // self.scale * self.scale, :w_hr // self.scale * self.scale, :]
        rgb = rgb[:h_hr // self.scale * self.scale, :w_hr // self.scale * self.scale, :]


        # to tensor
        hsi = torch.from_numpy(hsi.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)).contiguous()

        # accelerate validation, avoid oom
        w_crop = 256
        h_crop = 256
        rgb = rgb[:, 0: 0 + w_crop, 0: 0 + h_crop]
        hsi = hsi[:, 0: 0 + w_crop, 0: 0 + h_crop]


        # downscaling
        hsi = torch.unsqueeze(hsi, dim=0)
        hsi_down = F.interpolate(hsi, scale_factor=1/self.scale, mode='bicubic')
        hsi_down = F.interpolate(hsi_down, scale_factor=self.scale, mode='bicubic')

        hsi = torch.squeeze(hsi, dim=0)
        hsi_down = torch.squeeze(hsi_down, dim=0)

        return rgb, hsi_down, hsi




class HyperTrain_ICVL(Dataset):
    def __init__(self, args):
        super(HyperTrain_ICVL, self).__init__()

        self.mat_path = args.mat_path
        self.mat_names = os.listdir(self.mat_path)

        self.rgb_path = args.rgb_path

        # self.scale = args.scale
        self.scale_range  = args.scale_range
        self.patch_size = args.patch_size
        self.data_augmentation = args.augmentation


    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        self.scale = random.uniform(self.scale_range[0], self.scale_range[1])
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = loadmat(mat_name)
        hsi = mat['HyperImage']

        # rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4]+"_clean.png")
        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4]+".png") # patch
        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0

        # crop edge
        h_hr, w_hr, c = hsi.shape
        hsi = hsi[:h_hr // 16 * 16, :w_hr // 16 * 16, :]
        rgb = rgb[:h_hr // 16 * 16, :w_hr // 16 * 16, :]

        # # crop patch
        # h, w = hsi.shape[:2]
        # rand_h = random.randint(0, h - self.patch_size)
        # rand_w = random.randint(0, w - self.patch_size)
        # hsi_patch = hsi[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size, :]  # (xx, xx, 61), in range [0, 1], float64
        # rgb_patch = rgb[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size, :]  # (xx, xx, 3),  in range [0, 1], float64

        hsi_patch = np.transpose(hsi, [2, 0, 1])  # [3, xx, xx]
        rgb_patch = np.transpose(rgb, [2, 0, 1])  # [3, xx, xx]

        # data augmentation
        if self.data_augmentation:
            rgb_patch, hsi_patch = augment(rgb_patch, hsi_patch)

        # to tensor
        hsi_patch = torch.from_numpy(hsi_patch.astype(np.float32)).contiguous()
        rgb_patch = torch.from_numpy(rgb_patch.astype(np.float32)).contiguous()

        # downscaling
        hsi_patch = torch.unsqueeze(hsi_patch, dim=0)
        hsi_patch_down = F.interpolate(hsi_patch, scale_factor=1/self.scale, mode='bicubic')
        hsi_patch_down = F.interpolate(hsi_patch_down, size=(hsi_patch.shape[-2], hsi_patch.shape[-1]), mode='bicubic')
        hsi_patch = torch.squeeze(hsi_patch, dim=0)
        hsi_patch_down = torch.squeeze(hsi_patch_down, dim=0)

        # # crop patch
        rgb_patch, hsi_patch_down, hsi_patch = get_patch(rgb_patch, hsi_patch_down, hsi_patch, self.patch_size)

        return rgb_patch, hsi_patch_down, hsi_patch, self.scale


class HyperTrain_ICVL_disc(Dataset):
    def __init__(self, args):
        super(HyperTrain_ICVL_disc, self).__init__()

        self.mat_path = args.mat_path
        self.mat_names = os.listdir(self.mat_path)

        self.rgb_path = args.rgb_path

        # self.scale = args.scale
        self.scale_range  = [2, 2, 2, 2.5, 3.2, 4, 4]
        self.patch_size = args.patch_size
        self.data_augmentation = args.augmentation


    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        self.scale = random.choice(self.scale_range)
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = loadmat(mat_name)
        hsi = mat['HyperImage']

        # rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4]+"_clean.png")
        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4]+".png") # patch
        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0

        # crop edge
        h_hr, w_hr, c = hsi.shape
        hsi = hsi[:h_hr // 16 * 16, :w_hr // 16 * 16, :]
        rgb = rgb[:h_hr // 16 * 16, :w_hr // 16 * 16, :]

        # # crop patch
        # h, w = hsi.shape[:2]
        # rand_h = random.randint(0, h - self.patch_size)
        # rand_w = random.randint(0, w - self.patch_size)
        # hsi_patch = hsi[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size, :]  # (xx, xx, 61), in range [0, 1], float64
        # rgb_patch = rgb[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size, :]  # (xx, xx, 3),  in range [0, 1], float64

        hsi_patch = np.transpose(hsi, [2, 0, 1])  # [3, xx, xx]
        rgb_patch = np.transpose(rgb, [2, 0, 1])  # [3, xx, xx]

        # data augmentation
        if self.data_augmentation:
            rgb_patch, hsi_patch = augment(rgb_patch, hsi_patch)

        # to tensor
        hsi_patch = torch.from_numpy(hsi_patch.astype(np.float32)).contiguous()
        rgb_patch = torch.from_numpy(rgb_patch.astype(np.float32)).contiguous()

        # downscaling
        hsi_patch = torch.unsqueeze(hsi_patch, dim=0)
        hsi_patch_down = F.interpolate(hsi_patch, scale_factor=1/self.scale, mode='bicubic')
        hsi_patch_down = F.interpolate(hsi_patch_down, size=(hsi_patch.shape[-2], hsi_patch.shape[-1]), mode='bicubic')
        hsi_patch = torch.squeeze(hsi_patch, dim=0)
        hsi_patch_down = torch.squeeze(hsi_patch_down, dim=0)

        # # crop patch
        rgb_patch, hsi_patch_down, hsi_patch = get_patch(rgb_patch, hsi_patch_down, hsi_patch, self.patch_size)

        return rgb_patch, hsi_patch_down, hsi_patch, self.scale


class HyperValid_ICVL(Dataset):
    def __init__(self, args):
        super(HyperValid_ICVL, self).__init__()

        self.mat_path = args.mat_path_valid
        self.mat_names = os.listdir(self.mat_path)

        self.rgb_path = args.rgb_path_valid
        # self.rgb_names = os.listdir(self.rgb_path)

        self.scale = args.scale

    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = loadmat(mat_name)
        hsi = mat['HyperImage']

        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4] + "_clean.png")
        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0

        # crop edge
   
        h_hr, w_hr, c = hsi.shape
        hsi = hsi[:h_hr // 16 * 16, :w_hr // 16 * 16, :]
        rgb = rgb[:h_hr // 16 * 16, :w_hr // 16 * 16, :]

        # to tensor
        hsi = torch.from_numpy(hsi.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)).contiguous()

        # accelerate validation, avoid oom
        # w_crop = 256
        # h_crop = 256
        # rgb = rgb[:, 500: 500 + w_crop, 500: 500 + h_crop]
        # hsi = hsi[:, 500: 500 + w_crop, 500: 500 + h_crop]

        # downscaling
        hsi = torch.unsqueeze(hsi, dim=0)
        hsi_down = F.interpolate(hsi, scale_factor=1/self.scale, mode='bicubic')
        hsi_down = F.interpolate(hsi_down, size=(hsi.shape[-2], hsi.shape[-1]), mode='bicubic')

        hsi = torch.squeeze(hsi, dim=0)
        hsi_down = torch.squeeze(hsi_down, dim=0)

        return rgb, hsi_down, hsi, self.mat_names[index][:-4]






class HyperTrain_CAVE(Dataset):
    def __init__(self, args):
        super(HyperTrain_CAVE, self).__init__()

        self.mat_path = args.mat_path
        self.mat_names = os.listdir(self.mat_path)
        self.scale = args.scale

        self.rgb_path = args.rgb_path

        self.patch_size = args.patch_size
        self.data_augmentation = args.augmentation


    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = h5py.File(mat_name, 'r')
        hsi = mat['rad']
        hsi = np.transpose(hsi, [2, 1, 0])           # (512,512, 31)
        hsi = hsi / 65535.0
        hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())

        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-6] + "RGB.bmp")
        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        # crop edge
        h_hr, w_hr, c = hsi.shape
        hsi = hsi[:h_hr // self.scale * self.scale, :w_hr // self.scale * self.scale, :]
        rgb = rgb[:h_hr // self.scale * self.scale, :w_hr // self.scale * self.scale, :]

        # # crop patch
        # h, w = hsi.shape[:2]
        # rand_h = random.randint(0, h - self.patch_size)
        # rand_w = random.randint(0, w - self.patch_size)
        # hsi_patch = hsi[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size,
        #             :]  # (xx, xx, 31), in range [0, 1], float64
        # rgb_patch = rgb[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size,
        #             :]  # (xx, xx, 3),  in range [0, 1], float64

        hsi_patch = np.transpose(hsi, [2, 0, 1])  # [3, xx, xx]
        rgb_patch = np.transpose(rgb, [2, 0, 1])  # [3, xx, xx]

        # data augmentation
        # if self.data_augmentation:
        #     rgb_patch, hsi_patch = augment(rgb_patch, hsi_patch)

        # to tensor
        hsi_patch = torch.from_numpy(hsi_patch.astype(np.float32)).contiguous()
        rgb_patch = torch.from_numpy(rgb_patch.astype(np.float32)).contiguous()

        # downscaling
        hsi_patch = torch.unsqueeze(hsi_patch, dim=0)
        hsi_patch_down = F.interpolate(hsi_patch, scale_factor=1/self.scale, mode='bicubic')
        hsi_patch_down = F.interpolate(hsi_patch_down, scale_factor=self.scale, mode='bicubic')
        hsi_patch = torch.squeeze(hsi_patch, dim=0)
        hsi_patch_down = torch.squeeze(hsi_patch_down, dim=0)

        # # crop patch
        rgb_patch, hsi_patch_down, hsi_patch = get_patch(rgb_patch, hsi_patch_down, hsi_patch, self.patch_size)

        return rgb_patch, hsi_patch_down, hsi_patch


class HyperValid_CAVE(Dataset):
    def __init__(self, args):
        super(HyperValid_CAVE, self).__init__()

        self.mat_path = args.mat_path_valid
        self.mat_names = os.listdir(self.mat_path)

        self.rgb_path = args.rgb_path_valid
        self.scale = args.scale

    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = h5py.File(mat_name, 'r')
        # print(mat.keys())
        hsi = mat['rad']
        hsi = np.array(hsi)
        hsi = hsi.transpose(2, 1, 0)
        hsi = hsi / 65535.0
        hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())
        # (512, 512, 31), in range [0, 1], float32

        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-6] + "RGB.bmp")
        rgb = cv2.imread(rgb_name).astype(np.float32)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        # (512, 512, 3), in range [0, 1], float32

        # crop edge
        h_hr, w_hr, c = hsi.shape
        hsi = hsi[:h_hr // self.scale * self.scale, :w_hr // self.scale * self.scale, :]
        rgb = rgb[:h_hr // self.scale * self.scale, :w_hr // self.scale * self.scale, :]

        # to tensor
        hsi = torch.from_numpy(hsi.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)).contiguous()

        # accelerate validation, avoid oom
        w_crop = 256
        h_crop = 256
        rgb = rgb[:, 0: 0 + w_crop, 0: 0 + h_crop]
        hsi = hsi[:, 0: 0 + w_crop, 0: 0 + h_crop]

        # downscaling
        hsi = torch.unsqueeze(hsi, dim=0)
        hsi_down = F.interpolate(hsi, scale_factor=1/self.scale, mode='bicubic')
        hsi_down = F.interpolate(hsi_down, scale_factor=self.scale, mode='bicubic')

        hsi = torch.squeeze(hsi, dim=0)
        hsi_down = torch.squeeze(hsi_down, dim=0)

        return rgb, hsi_down, hsi

