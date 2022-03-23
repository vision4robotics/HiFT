import torch.nn as nn
import torch.nn.functional as F
import torch as t
import math
from pysot.models.utile.tran import Transformer

    
class HiFT(nn.Module):
    
    def __init__(self,cfg):
        super(HiFT, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, bias=False,stride=2,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, bias=False,stride=2,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=3, bias=False,stride=2,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            )
        
        
        channel=192

        self.convloc = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),                
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, 4,  kernel_size=3, stride=1,padding=1),
                )
        
        self.convcls = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),

                )

        self.row_embed = nn.Embedding(50, channel//2)
        self.col_embed = nn.Embedding(50, channel//2)
        self.reset_parameters()
        
        self.transformer = Transformer(channel, 6, 1, 2)
        
        self.cls1=nn.Conv2d(channel, 2,  kernel_size=3, stride=1,padding=1)
        self.cls2=nn.Conv2d(channel, 1,  kernel_size=3, stride=1,padding=1)
        for modules in [self.conv1]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        
    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    def forward(self,x,z):

        res1=self.conv1(self.xcorr_depthwise(x[0],z[0]))
        res2=self.conv3(self.xcorr_depthwise(x[1],z[1]))
        
        res3=self.conv2(self.xcorr_depthwise(x[2],z[2]))
        
        
        h, w = res3.shape[-2:]
        i = t.arange(w).cuda()
        j = t.arange(h).cuda()
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = t.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(res3.shape[0], 1, 1, 1)
        

        
        b,c,w,h=res3.size()
        res=self.transformer((pos+res1).view(b,c,-1).permute(2, 0, 1),\
                             (pos+res2).view(b,c,-1).permute(2, 0, 1),\
                             res3.view(b,c,-1).permute(2, 0, 1))
        
        res=res.permute(1,2,0).view(b,c,w,h)
        
        loc=self.convloc(res)
        acls=self.convcls(res)

        cls1=self.cls1(acls)
        cls2=self.cls2(acls)

        return loc,cls1,cls2





