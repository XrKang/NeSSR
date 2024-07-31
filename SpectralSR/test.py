# -*- coding: utf-8 -*
# !/usr/local/bin/python
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from hsi_utils import *
from hsi_dataloader import *
from NeSSR import *
import torch.nn.functional as F

from scipy.io import savemat
import os
from functools import partial
import pickle
import random
parser = argparse.ArgumentParser(description="HSI Rec")
# data loader

parser.add_argument('--patch_size', type=float, default=256)
parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
# -----------------------------------------------------
parser.add_argument("--mat_path_valid", type=str,
                    default='/data/HSI_Rec/ICVL/test_mat61',
                    help="HyperSet path")

parser.add_argument("--rgb_path_valid", type=str,
                    default='/data/HSI_Rec/ICVL/rgb_clean',
                    help="RGBSet path")
parser.add_argument('--model_path', default=r'./model_spectralRec.pth', help='model_paht')
parser.add_argument('--save_dir', type=str, default='./test_result')


def load_parallel(model_path):
    state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict



def valid_icvl(arg, model):
    H = (1392 // 4) * 4
    W = (1300 // 4) * 4


    torch.cuda.empty_cache()
    val_set = HyperValid_ICVL(arg)
    val_set_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

    MRAE_epoch_noPatch = 0
    RMSE_epoch_noPatch = 0
    model.eval()
    logtext = ""

    save_path = arg.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("----------------", save_path, "----------------")
    for iteration, (rgb, hsi, name) in enumerate(val_set_loader):
        torch.cuda.empty_cache()

        if arg.cuda:
            rgb = rgb.cuda()
            hsi = hsi.cuda()
        
        _, _, H_crop, W_crop = hsi.shape
        hsi = torch.unsqueeze(hsi, dim=0)
        hsi = F.interpolate(hsi, size=(arg.bands, H_crop, W_crop), mode='trilinear')
        hsi = torch.squeeze(hsi, dim=0)
        b, bands_num, h, w = hsi.shape

        # h_crop, w_crop = 512, 512 # A100
        # h_crop, w_crop = 256, 256
        h_crop = arg.patch_size
        w_crop = arg.patch_size
        stride = h_crop // 2  
        result = torch.zeros(b, bands_num, h, w).cuda()
        count = torch.zeros(b, bands_num, h, w).cuda()  

        for idx_H in range(0, h, stride):
            for idx_W in range(0, w, stride):
                rgb_patch = rgb[:, :, idx_H:idx_H + h_crop, idx_W:idx_W + w_crop]
                B, _, H_p, W_p = rgb_patch.shape
                coor = make_coord([bands_num, H_p, W_p]). \
                    permute(3, 0, 1, 2).unsqueeze(0).expand(B, 3, *[bands_num, H_p, W_p]).cuda()

                scale = torch.ones_like(coor)
                scale[:, -3, :, :, :] *= 1 / rgb_patch.shape[-3]
                scale[:, -2, :, :, :] *= 1 / rgb_patch.shape[-2]
                scale[:, -1, :, :, :] *= 1 / rgb_patch.shape[-1]

                pred_patch = model(rgb_patch, coor, scale)
                result[:, :, idx_H:idx_H + h_crop, idx_W:idx_W + w_crop] += pred_patch
                count[:, :, idx_H:idx_H + h_crop, idx_W:idx_W + w_crop] += 1

        result /= count 

        pred = result

        MRAE_noPatch = computeMRAE(pred, hsi)
        RMSE_noPatch = computeRMSE(pred, hsi)

        MRAE_epoch_noPatch = MRAE_epoch_noPatch + MRAE_noPatch
        RMSE_epoch_noPatch = RMSE_epoch_noPatch + RMSE_noPatch
        print("VAL_noPatch===> Val.MRAE: {:.8f} RMSE: {:.8f} ".format(MRAE_noPatch, RMSE_noPatch))

        # -----------log Writting-----------
        # logtext += "VAL_noPatch===> Val.MRAE: {:.8f} RMSE: {:.8f} ".format(MRAE_noPatch, RMSE_noPatch) + "\n"
        # with open(os.path.join(save_path, "test_log.txt"), 'a') as f:
        #     f.write(logtext + "\n")

        # -----------Save Results-----------
        gt = hsi[0].cpu().numpy()
        gt_dic = {'HyperImage': gt}
        gt_name = os.path.join(save_path, str(name[0])+'_gt.mat')
        # savemat(gt_name, gt_dic, do_compression=True)
        savemat(gt_name, gt_dic)

        pred = pred[0].cpu().numpy()
        pred_dic = {'HyperImage': pred}
        pred_name = os.path.join(save_path, str(name[0]) + '_pred.mat')
        # savemat(pred_name, pred_dic, do_compression=True)
        savemat(pred_name, pred_dic)

    MRAE_valid_noPatch = MRAE_epoch_noPatch / (iteration + 1)
    RMSE_valid_noPatch = RMSE_epoch_noPatch / (iteration + 1)

    print("VAL_noPatch===> Val_Avg. MRAE: {:.8f} RMSE: {:.8f}".format(MRAE_valid_noPatch, RMSE_valid_noPatch))
    
    # -----------log Writting-----------
    # logtext += "VAL_noPatch===> Val_Avg. MRAE: {:.8f} RMSE: {:.8f}".format(MRAE_valid_noPatch, RMSE_valid_noPatch)+ "\n"
    # with open(os.path.join(save_path, "test_log.txt"), 'a') as f:
    #     f.write(logtext + "\n")


def main(opt):
    torch.cuda.empty_cache()
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = opt.model_path

    model = NeSRP(opt)

    # model = torch.nn.DataParallel(model)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        model.load_state_dict(load_parallel(model_path))
        # model.load_state_dict(torch.load(model_path)['state_dict'])

    model.eval()

    if opt.cuda:
        model.cuda()
    # oldtime = datetime.datetime.now()
    with torch.no_grad():
        valid_icvl(opt, model)

    # newtime = datetime.datetime.now()
    # print('Time consuming: ', newtime - oldtime)



if __name__ == "__main__":
    import os
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    args = parser.parse_args()
    args.in_channels = 3
    args.n_feat = 64
    args.out_channels = 32
    args.stage = 2
    args.hidden_list = [32, 16, 128, 128, 256, 256]

    args.imnet_in_dim = 32
    args.imnet_out_dim = 1
    args.numb_MultiHead = 2
    print(args)

    bands = [11, 16, 31, 41, 51, 61]
    save_dir = args.save_dir

    for band in bands:
        args.bands = band
        args.save_dir = save_dir + '/NeSSR_icvl_' + str(args.bands)

        print('-----------------------------------------')
        print("band number: ", band)
        main(args)
        print('-----------------------------------------')


