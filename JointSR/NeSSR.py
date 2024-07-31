import torch
from torch import nn
from torch.nn import functional as F
import math
from SSE import *




class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value):
        scores = torch.matmul(query.permute(1, 0), key)  # c, shape * shape, c
        p_attn = F.softmax(scores, dim=-1)  # c, c
        value = value.contiguous().permute(1, 0)
        p_val = torch.matmul(p_attn, value)
        p_val = p_val.contiguous().permute(1, 0)
        return p_val

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, args):
        super().__init__()
        hidden_list = args.hidden_list
        self.numb_MultiHead = args.numb_MultiHead

        layers_q1 = []
        lastv = in_dim+6
        for hidden in hidden_list[:2]:
            layers_q1.append(nn.Linear(lastv, hidden))
            layers_q1.append(nn.ReLU())
            lastv = hidden

        layers_q1.append(nn.Linear(lastv, lastv))
        self.layers_q1 = nn.Sequential(*layers_q1)

        layers_q2 = []
        lastv = in_dim
        for hidden in hidden_list[:2]:
            layers_q2.append(nn.Linear(lastv, hidden))
            layers_q2.append(nn.ReLU())
            lastv = hidden

        layers_q2.append(nn.Linear(lastv, lastv))
        self.layers_q2 = nn.Sequential(*layers_q2)

        layers_k = []
        lastv = in_dim
        for hidden in hidden_list[:2]:
            layers_k.append(nn.Linear(lastv, hidden))
            layers_k.append(nn.ReLU())
            lastv = hidden

        layers_k.append(nn.Linear(lastv, lastv))
        self.layers_k = nn.Sequential(*layers_k)

        layers_v = []
        lastv = in_dim
        for hidden in hidden_list[:2]:
            layers_v.append(nn.Linear(lastv, hidden))
            layers_v.append(nn.ReLU())
            lastv = hidden

        layers_v.append(nn.Linear(lastv, lastv))
        self.layers_v = nn.Sequential(*layers_v)

        self.attention = Attention()

        layers2 = []
        lastv = lastv * self.numb_MultiHead
        for hidden in hidden_list[:2]:
            layers2.append(nn.Linear(lastv, hidden))
            layers2.append(nn.ReLU())
            lastv = hidden
        layers2.append(nn.Linear(lastv, lastv))
        self.layers2 = nn.Sequential(*layers2)

        layers_coord = []
        lastv = 3
        for hidden in hidden_list[:2]:
            layers_coord.append(nn.Linear(lastv, hidden))
            layers_coord.append(nn.ReLU())
            lastv = hidden
        layers_coord.append(nn.Linear(lastv, lastv))
        self.layers_coord = nn.Sequential(*layers_coord)

        layers_out = []
        lastv = lastv
        for hidden in hidden_list[2:]:
            layers_out.append(nn.Linear(lastv, hidden))
            layers_out.append(nn.ReLU())
            lastv = hidden
        layers_out.append(nn.Linear(lastv, out_dim))
        self.layers_out = nn.Sequential(*layers_out)

    def forward(self, feature, feature_coord, coord):
        shape = feature.shape[:-1]
        query_1 = self.layers_q1(feature_coord.view(-1, feature_coord.shape[-1]))
        query_2 = self.layers_q2(feature.view(-1, feature.shape[-1]))
        key = self.layers_k(feature.view(-1, feature.shape[-1]))
        value = self.layers_v(feature.view(-1, feature.shape[-1]))

        x_att = []
        for idx in range(self.numb_MultiHead):
            x = self.attention((query_1-query_2), key, value)
            x_att.append(x)
        x_att_output = torch.cat(x_att, -1)
        x_att_output = self.layers2(x_att_output)

        coord_out = self.layers_coord(coord.view(-1, coord.shape[-1]))
        output = x_att_output + coord_out

        output = self.layers_out(output)

        return output.view(*shape, -1)


class NeSRP(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.feature_encoder = Feature_Encoder(args)
        self.conv_3d_1 = nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv_3d_2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.imnet_in_dim = args.imnet_in_dim
        self.imnet_out_dim = args.imnet_out_dim
        self.imnet = MLP(self.imnet_in_dim, self.imnet_out_dim, args)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)



    def forward(self, rgb, hsi,  coor, scale):
        # 获取LR的feature和coordinate
        x = torch.cat([rgb, hsi], dim=1)
        feat = self.feature_encoder(x)  # bs,nf, h, w
        bs, _, bands, H, W = coor.shape

        _, C0, _, _ = feat.shape

        feat_HC_pre = feat.permute(0, 3, 2, 1)  # (b,w,h,ch)
        feat_WC_pre = feat.permute(0, 2, 3, 1)  # (b,h,w,ch)

        feat_HC = F.upsample(feat_HC_pre, [H, bands], mode="nearest")
        feat_WC = F.upsample(feat_WC_pre, [W, bands], mode="nearest")

        feat_HC = feat_HC.permute(0, 3, 2, 1)   # (b,bands,h,w)
        feat_HC = feat_HC.unsqueeze(dim=1)      # (b,1, bands,h,w)
        # print("feat_HC", feat_HC.shape)
        feat_WC = feat_WC.permute(0, 3, 1, 2)   # (b,bands,h,w)
        feat_WC = feat_WC.unsqueeze(dim=1)      # (b,1,bands,h,w)
        # print("feat_WC", feat_WC.shape)

        feat_inp = torch.cat([feat_HC, feat_WC], dim=1) # (b,2,bands,h,w)
        # print("feat_inp", feat_inp.shape)
        feat_inp_3d = self.lrelu(self.conv_3d_2(self.conv_3d_1(feat_inp)))
        feat_inp_coord = torch.cat([feat_inp_3d, coor, scale], dim=1)
        # feat_inp_3d = self.lrelu((self.conv_3d_1(feat_inp)))
        # print("feat_inp_3d", feat_inp_3d.shape)     # (b,16,bands,h,w)

        feat_inp_3d = feat_inp_3d.permute(0, 2, 3, 4, 1)    # (b,bands,h,w,16)
        feat_inp_coord = feat_inp_coord.permute(0, 2, 3, 4, 1)    # (b,bands,h,w,16+3)
        coord = coor.permute(0, 2, 3, 4, 1)    # (b,bands,h,w,3)

        bs, bands, H, W, _ = feat_inp_3d.shape
        # print("feat_inp_3d", feat_inp_3d.shape)
        feat_mlp_inp = feat_inp_3d.contiguous().view(bs * bands * H * W, -1)
        feat_mlp_inp_coord = feat_inp_coord.contiguous().view(bs * bands * H * W, -1)
        coord = coord.contiguous().view(bs * bands * H * W, -1)
        pred = self.imnet(feat_mlp_inp, feat_mlp_inp_coord, coord).contiguous().view(bs, bands, H, W, -1)

        pred = pred[:, :, :, :, 0]  # (b,bands,h,w)
        # print("pred", pred.shape)

        return pred

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
    return ret
    # If not flatten: (H,W,2)

if __name__ == "__main__":
    from argparse import Namespace

    from thop.profile import profile


    args = Namespace()

    args.local_ensemble = False
    args.in_channels = 3+31
    args.n_feat = 64
    args.out_channels = 32
    args.stage = 2
    args.hidden_list = [32, 16, 128, 128, 256, 256]

    args.imnet_in_dim = 32
    args.imnet_out_dim = 1
    args.numb_MultiHead = 2

    input_tensor = torch.rand(1, 31, 128, 128).cuda()
    input_tensor2 = torch.rand(1, 3, 128, 128).cuda()

    B, ch, H_p, W_p = input_tensor.shape
    coord = make_coord([31, H_p, W_p]). \
        permute(3, 0, 1, 2).unsqueeze(0).expand(B, 3, *[31, H_p, W_p]).cuda()  # B, 3, bands_num, H, W

    scale = torch.ones_like(coord)
    scale[:, -3, :, :, :] *= 1 / coord.shape[-3]
    scale[:, -2, :, :, :] *= 1 / coord.shape[-2]
    scale[:, -1, :, :, :] *= 1 / coord.shape[-1]
    # print(coord.shape)

    # model = AWAN(3, 31, 200, 10)
    model = NeSRP(args)
    # model = nn.DataParallel(model).cuda()
    model = model.cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor, input_tensor2, coord, scale)
    print(output_tensor.size())
    name = "aaa"
    total_ops, total_params = profile(model, (input_tensor, input_tensor2, coord, scale,))
    print("%s         | %.2f(M)      | %.2f(G)         |" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))

    print('Parameters number is %.2f(M) '% (sum(param.numel() for param in model.parameters())/ (1000 ** 2)))
    print(torch.__version__)
    # aaa         | 5.26(M)      | 97.66(G)         |
