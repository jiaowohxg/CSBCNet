import numpy as np
import torch
from torch import nn
from torch.nn import init

class Channel_Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.ch_wv = nn.Sequential(
            nn.Conv2d(channels,channels//2,kernel_size=(1,1)),  # 用 Linear 层替代 Conv2d
            nn.BatchNorm2d(channels // 2, eps=1e-3),
            nn.ReLU()
        )
        self.ch_wq = nn.Conv2d(channels,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.ch_wz = nn.Sequential(
            nn.Conv2d(channels//2,channels,kernel_size=(1,1)),  # 用 Linear 层替代 Conv2d
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.ReLU()
        )
        self.ln=nn.LayerNorm(channels)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x
        return channel_out


class Spatial_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Spatial_Attention, self).__init__()
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.GELU(),
            nn.Conv2d(in_channel // 4, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
        )
        self.ebd = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=True),
            nn.InstanceNorm2d(in_channel, eps=1e-5),
            nn.BatchNorm2d(in_channel)
        )

        self.pointcn = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel), nn.GELU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel), nn.GELU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
        )
    def forward(self, feature):
        feature = self.ebd(feature)
        res = feature
        feature = self.pointcn(feature)
        avgpool = feature.mean(dim=2).unsqueeze(dim=2)
        maxpool = feature.max(dim=2)[0].unsqueeze(dim=2)
        att = avgpool + maxpool
        att = self.channel_att(att)
        att = torch.sigmoid(att)
        feature = feature * att
        feature = feature + res
        return feature


           