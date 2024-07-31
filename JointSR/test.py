# -*- coding: utf-8 -*
# !/usr/local/bin/python
import argparse, os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from hsi_utils import *
from hsi_dataloader import *
from NeSSR import *
import torch.nn.functional as F
import os
from functools import partial
import pickle
import random
from scipy.io import savemat


parser = argparse.ArgumentParser(description="HSI Rec")
parser.add_argument("--mat_path_valid", type=str,
                    default='/data/HSI_Rec/ICVL/test_mat61',
                    help="HyperSet path")

parser.add_argument("--rgb_path_valid", type=str,
                    default='/data/HSI_Rec/ICVL/rgb_clean',
                    help="RGBSet path")

parser.add_argument("--model_path", type=str,
                    default=r"./model_jointRec.pth",
                    help="model path")

parser.add_argument('--patch_size', type=float, default=256) # Set to 512 if A100.
parser.add_argument('--save_dir', type=str, default='./test_result')

parser.add_argument("--cuda", action="store_false", help="use cuda?")

opt = parser.parse_args()


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
    torch.cuda.empty_cache()
    val_set = HyperValid_ICVL(arg)
    val_set_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)
    MRAE_epoch = 0
    RMSE_epoch = 0
    model.eval()
    for iteration, (rgb, hsi_lr, hsi_hr, name) in enumerate(val_set_loader):

        if arg.cuda:
            rgb = rgb.cuda()
            hsi_lr = hsi_lr.cuda()
            hsi_hr = hsi_hr.cuda()

        _, _, H_hr, W_hr = hsi_hr.shape
        hsi_hr = torch.unsqueeze(hsi_hr, dim=0)
        hsi_hr = F.interpolate(hsi_hr, size=(arg.band_number, H_hr, W_hr), mode='trilinear')
        hsi_hr = torch.squeeze(hsi_hr, dim=0)

        _, _, H_lr, W_lr = hsi_lr.shape
        hsi_lr = torch.unsqueeze(hsi_lr, dim=0)
        hsi_lr = F.interpolate(hsi_lr, size=(11, H_lr, W_lr), mode='trilinear')
        hsi_lr = torch.squeeze(hsi_lr, dim=0)
        
        B, bands_num, h, w = hsi_hr.shape
        h_crop, w_crop = arg.patch_size, arg.patch_size     
        stride = w_crop // 2  # 设置步长为块大小的一半，以实现50%的重叠
        result = torch.zeros(B, bands_num, h, w).cuda()
        count = torch.zeros(B, bands_num, h, w).cuda()  # 用于记录每个像素点的预测次数

        for idx_H in range(0, h, stride):
            for idx_W in range(0, w, stride):
                rgb_patch = rgb[:, :, idx_H:idx_H + h_crop, idx_W:idx_W + w_crop]
                lr_patch = hsi_lr[:, :, idx_H:idx_H + h_crop, idx_W:idx_W + w_crop]
                _, _, H_p, W_p = lr_patch.shape
                coor = make_coord([bands_num, H_p, W_p]). \
                    permute(3, 0, 1, 2).unsqueeze(0).expand(B, 3, *[bands_num, H_p, W_p]).cuda()

                scale = torch.ones_like(coor)
                scale[:, -3, :, :, :] *= 1 / rgb_patch.shape[-3]
                scale[:, -2, :, :, :] *= 1 / rgb_patch.shape[-2]
                scale[:, -1, :, :, :] *= 1 / rgb_patch.shape[-1]

                pred_patch = model(rgb_patch, lr_patch, coor, scale)
                result[:, :, idx_H:idx_H + h_crop, idx_W:idx_W + w_crop] += pred_patch
                count[:, :, idx_H:idx_H + h_crop, idx_W:idx_W + w_crop] += 1

        result /= count  # 对每个像素点的预测值取平均


        MRAE = computeMRAE(result, hsi_hr)
        MRAE_epoch = MRAE_epoch + MRAE

        RMSE = computeRMSE(result, hsi_hr)
        RMSE_epoch = RMSE_epoch + RMSE
        print("VAL===> Val.MRAE: {:.5f} RMSE: {:.5f}".format(MRAE, RMSE))

        
        hsi_hr = hsi_hr[0].cpu().numpy()
        gt_dic = {'HyperImage': hsi_hr}
        gt_name = os.path.join(save_path, str(name[0]) + '_gt.mat')
        # savemat(pred_name, pred_dic, do_compression=True)
        savemat(gt_name, gt_dic)

        result = result[0].cpu().numpy()
        pred_dic = {'HyperImage': result}
        pred_name = os.path.join(save_path, str(name[0]) + '_pred.mat')
        # savemat(pred_name, pred_dic, do_compression=True)
        savemat(pred_name, pred_dic)

    MRAE_valid = MRAE_epoch / (iteration + 1)
    RMSE_valid = RMSE_epoch / (iteration + 1)
    print("VAL===> Val_Avg. MRAE: {:.8f} RMSE: {:.8f}".format(MRAE_valid, RMSE_valid))
    return MRAE_valid, RMSE_valid




def main(opt, model):
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


if __name__ == '__main__':
    import os
    opt.in_channels = 3+11
    opt.n_feat = 64
    opt.out_channels = 32
    opt.stage = 2
    opt.hidden_list = [32, 16, 128, 128, 256, 256]

    opt.imnet_in_dim = 32
    opt.imnet_out_dim = 1
    opt.numb_MultiHead = 2

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # save_path = opt.save_dir
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    if opt.cuda:
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    print(opt)
    save_dir = opt.save_dir
    scale_list = [8, 6, 4, 2, 3.2, 2.5]
    for scale in scale_list:
        opt.scale = scale
        if opt.scale == 2 or opt.scale == 2.5 or opt.scale == 3.2:
            opt.band_number = 16
        elif opt.scale == 4:
            opt.band_number = 31
        elif opt.scale == 6:
            opt.band_number = 41
        elif opt.scale == 8:
            opt.band_number = 61
        model = NeSRP(opt)
        opt.save_dir = save_dir + '/NeSSR_icvl_' + str(opt.band_number) + 'bands_' + str(opt.scale) + 'x'
        save_path = opt.save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(save_path)
        print("--------Band Number:", opt.band_number, " | scale:", opt.scale, '-------------')
        main(opt, model)









