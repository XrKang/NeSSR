from __future__ import division

import torch
import torch.nn as nn
import logging
from scipy.io import loadmat,savemat

from PIL import Image, ImageOps
import os

from torchvision.transforms import Compose
from torchvision import transforms
import random
import h5py
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

class Loss_train(nn.Module):
    def __init__(self):
        super(Loss_train, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / (label + 0.0001)
        # error = torch.abs(outputs - label)
        rrmse = torch.mean(error.view(-1))
        return rrmse
        
def make_coord(shape, ranges=None,):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()    # size->(H,) range([-1, 1) center(=0.00
        # r=1/H
        # seq=(((2/H * arr[0:H-1])->arr[0:2*(H-1)/H] + (-1))->arr[-1:(2*H-2)/H)-1]) + (1/H)) -> arr[-1/H:1/H].size(H,)
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    # ret->size[H,W] range([-1/H, 1/H], [-1/W, 1/W]), center(1/H,1/W)=0.0.0
    # torch.meshgrid()的功能是生成网格，可以用于生成坐标。
    return ret

# def make_coord(shape, ranges=None, flatten=False):
#     """ Make coordinates at grid centers.
#     """
#     coord_seqs = []
#     for i, n in enumerate(shape):
#         if ranges is None:
#             v0, v1 = -1, 1
#         else:
#             v0, v1 = ranges[i]
#         r = (v1 - v0) / (2 * n)
#         seq = v0 + r + (2 * r) * torch.arange(n).float()    # size->(H,) range([-1, 1) center(=0.00
#         # r=1/H
#         # seq=(((2/H * arr[0:H-1])->arr[0:2*(H-1)/H] + (-1))->arr[-1:(2*H-2)/H)-1]) + (1/H)) -> arr[-1/H:1/H].size(H,)
#         coord_seqs.append(seq)
#     ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
#     # ret->size[H,W] range([-1/H, 1/H], [-1/W, 1/W]), center(1/H,1/W)=0.0.0
#     if flatten:
#         ret = ret.view(-1, ret.shape[-1])
#     return ret

    # If not flatten: (H,W,2)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=7, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def Cal_mse(im1, im2):
    return np.mean(np.square(im1 - im2), dtype=np.float16) + 1*1e-10


def cal_psnr(im_true, im_test):
    # data range:[0,1]
    # shape: (bs, c, h, w)
    im_true = im_true.cpu()
    im_test = im_test.cpu()
    im_true = im_true.detach().numpy()
    im_test = im_test.detach().numpy()

    p = [compare_psnr(im_true[k, :, :], im_test[k, :, :], data_range=1.0) for k in range(im_test.shape[0])]
    mean_p = sum(p) / im_test.shape[0]
    return mean_p

    # channel = im_true.shape[0]
    # # im_true = 255 * im_true
    # # im_test = 255 * im_test
    # psnr_sum = 0
    # for i in range(channel):
    #     band_true = np.squeeze(im_true[i, :, :])
    #     band_test = np.squeeze(im_test[i, :, :])
    #     err = Cal_mse(band_true, band_test)
    #     max_value = 1.0
    #     psnr_sum = psnr_sum + 10 * np.log10((max_value ** 2) / (err + 1*1e-10))
    # return psnr_sum / channel

# def cal_ssim(im_true, im_test):
#     im_true = im_true.cpu()
#     im_test = im_test.cpu()
#     im_true = im_true.detach().numpy()
#     im_test = im_test.detach().numpy()
#
#     channel = im_true.shape[0]
#     s = compare_ssim(im_true, im_test, K1=0.01, K2=0.03, window_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0,
#              multichannel=True)
#     return s

def cal_ssim(img, ref):
    # data range:[0,1]
    # shape: (bs, c, h, w)
    ssim_sum = ssim(img, ref)
    return ssim_sum

def cal_psnr_np(im_true, im_test):
    # data range:[0,1]
    # shape: (c, h, w)
    channel = im_true.shape[0]
    # im_true = 255 * im_true
    # im_test = 255 * im_test
    psnr_sum = 0
    for i in range(channel):
        band_true = np.squeeze(im_true[i, :, :])
        band_test = np.squeeze(im_test[i, :, :])
        err = Cal_mse(band_true, band_test)
        max_value = 1.0
        psnr_sum = psnr_sum + 10 * np.log10((max_value ** 2) / err)
    return psnr_sum / channel


def cal_ssim_np(img, ref):
    # data range:[0,1]
    # shape: (c, h, w)
    img = torch.from_numpy(img)
    ref = torch.from_numpy(ref)
    ssim_sum = ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))
    return ssim_sum


def computeMRAE_numpy(recovered, groundTruth):
    """
    Compute MRAE between two images
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :return: Mean Realative Absolute Error between `recovered` and `groundTruth`.
    """

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

    difference = np.abs(groundTruth - recovered) / (groundTruth+0.0001)
    mrae = np.mean(difference)

    return mrae


def computeRMSE_numpy(recovered, groundTruth):
    """
    Compute RMSE between two images
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :return: RMSE between `recovered` and `groundTruth`.
    """

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

    difference = (groundTruth - recovered) ** 2
    rmse = np.sqrt(np.mean(difference))

    return rmse


def computeMRAE(recovered, groundTruth):
    """
    Compute MRAE between two images
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :return: Mean Realative Absolute Error between `recovered` and `groundTruth`.
    """
    recovered = recovered.cpu()
    groundTruth = groundTruth.cpu()
    recovered = recovered.detach().numpy()
    groundTruth = groundTruth.detach().numpy()

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

    difference = np.abs(groundTruth - recovered) / (groundTruth+0.0001)
    mrae = np.mean(difference)

    return mrae


def computeRMSE(recovered, groundTruth):
    """
    Compute RMSE between two images
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :return: RMSE between `recovered` and `groundTruth`.
    """
    recovered = recovered.cpu()
    groundTruth = groundTruth.cpu()
    recovered = recovered.detach().numpy()
    groundTruth = groundTruth.detach().numpy()

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

    difference = (groundTruth - recovered) ** 2
    rmse = np.sqrt(np.mean(difference))

    return rmse

def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=10, max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def cal_psnr(img1, img2):
#     img1 = img1.cpu()
#     img2 = img2.cpu()
#     img1_np = img1.detach().numpy()
#     img2_np = img2.detach().numpy()
#
#     p = [measure.compare_psnr(img1_np[k, :, :], img2_np[k, :, :]) for k in range(img2.shape[0])]
#     mean_p = sum(p) / (len(p) + 1)
#     return mean_p

def mrae_loss(outputs, label):
    error = torch.abs(outputs - label) / (label+0.0001)
    mrae = torch.mean(error.view(-1))
    return mrae
