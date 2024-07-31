import sys
sys.path.insert(1, '/data/pylib')

import torch.nn as nn
import torch
import torch.nn.functional as F
from arch_utils import *


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.CA = CALayer(out_features)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        u = x.clone()
        x = self.CA(x) + u
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.v = nn.Conv2d(dim, dim, 1)

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.conv_catt = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim // 16, dim, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        v = self.v(x)

        catt = self.conv_catt(x)
        catt = self.avg_pool(catt)
        catt = self.conv_du(catt)

        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        attn = attn * catt

        output = v * attn + x

        return output


class AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.attention_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.attention_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class SLAB(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        self.attn = AttentionBlock(dim)
        self.FFN = FFN(in_features=dim, hidden_features=dim, act_layer=act_layer)
        self.layer_scale = nn.Parameter(torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.attn(x)
        x = x + self.FFN(x)
        return x



class SLAT(nn.Module):
    def __init__(self, in_dim, out_dim, dim, stage):
        super(SLAT, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                SLAB(dim=dim_stage),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = SLAB(dim=dim_stage)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                SLAB(dim=dim_stage // 2),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)   # 2*(stage)*dim

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = F.interpolate(fea, scale_factor=2, mode='bilinear')
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))  # 2*(stage-i)*dim +
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

class Feature_Encoder(nn.Module):
    def __init__(self, args):
        super(Feature_Encoder, self).__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.n_feat = args.n_feat
        self.stage = args.stage

        self.conv_in = nn.Conv2d(self.in_channels, self.n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        modules_body = [SLAT(in_dim=self.n_feat, out_dim=self.n_feat, dim=self.n_feat, stage=2)
                        for _ in range(self.stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(self.n_feat, self.n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.conv_last1 = nn.Conv2d(self.n_feat, 64, 3, 1, 1, bias=True)
        self.conv_last2 = nn.Conv2d(64, self.out_channels, 3, 1, 1, bias=True)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # x = resample_data(x, 2)

        x = self.lrelu(self.conv_in(x))
        h = self.body(x)
        h = self.conv_out(h)
        h += x
        h = self.lrelu(self.conv_last1(h))
        h = self.conv_last2(h)

        return h[:, :, :h_inp, :w_inp]



if __name__ == '__main__':
    import argparse
    from thop.profile import profile

    parser = argparse.ArgumentParser(description="SLAT")
    args = parser.parse_args()
    args.in_channels = 3
    args.out_channels = 31
    args.n_feat = 64
    args.stage = 2

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model = Feature_Encoder(args).to(device)
    model.train()
    # model.eval()
    dsize1 = (1, 3, 128, 128)
    # dsize1 = (1, 3, 482, 512)

    name = "SLAT"
    input1 = torch.randn(dsize1).to(device)
    print(model(input1).shape)
    total_ops, total_params = profile(model, (input1,))
    print("%s         | %.2f(M)      | %.2f(G)         |" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))

    # 128x 128  SLAT | 2.57(M) | 4.12(G) |

    import datetime
    oldtime = datetime.datetime.now()
    with torch.no_grad():
        for i in range(10):
            output_tensor = model(input1)
            if i==1:
                print(output_tensor.shape)
    newtime = datetime.datetime.now()
    print('Time consuming: ', newtime - oldtime)
    # 32x32 10 times -> 0.174
    # 64x64 10 times -> 0.203046






