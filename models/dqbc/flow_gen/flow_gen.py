import torch.nn as nn
import torch
from ..common import conv
from validate.bucket import Bucket
from ..corr import get_corr_dim
from ..common import Conv2

class IniNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        c = cfg.dims
        self.down0 = Conv2(6,c[0])
        self.down1 = Conv2(c[0],c[1])
        self.down2 = Conv2(c[1],c[2])

    def forward(self,im0,im1,bkt:Bucket=None):
        x = torch.cat([im0,im1],1)
        h = self.down2(self.down1(self.down0(x)))
        return h

def make_norm(name, num_channels):
    if name=='none':
        return nn.Identity()
    elif name[:5]=='group':
        n = int(name[5:])
        return nn.GroupNorm(n, num_channels)
    else:
        raise NotImplementedError

class ShiftCorrManip(nn.Module):
    def __init__(self,cfg, corr_dim, shift=True) -> None:
        super().__init__()
        out_c = cfg.out_c
        if out_c<=0:
            out_c = corr_dim
        self.shift = shift
        self.mlp = nn.Sequential(
            nn.Conv2d(corr_dim,4*out_c,1,1,0),
            nn.GELU(),
            nn.Conv2d(4*out_c,out_c,1,1,0)
        )
    
    def forward(self, corr, corr_fn):
        # corr: B,C,H,W

        if self.shift:
            corr = corr_fn.shift_corr(corr)

        corr = self.mlp(corr)
            
        return corr

def make_corr_manip(cfg, corr_dim):
    return ShiftCorrManip(cfg, corr_dim, True)

class FlowGenerator(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.inet = IniNet(cfg.inet)
        c = cfg.inet.dims[2]

        corr_dim = get_corr_dim(cfg.corr)[0]

        self.corr_manip = make_corr_manip(cfg.corr_manip, corr_dim)
        if cfg.corr_manip.out_c <= 0:
            out_c = corr_dim
        else:
            out_c = cfg.corr_manip.out_c
        
        self.fuse_block = nn.Sequential(
            conv(c+out_c,c),
            conv(c,c,7,1,3,4),
            nn.Conv2d(c,c,3,1,1)
        )
        self.act = nn.PReLU(c)

        self.norm = make_norm(cfg.norm,c)

        self.head = nn.Conv2d(c,4,3,1,1)

    def forward(self,im0,im1,corr_fn,c0,c1,bkt:Bucket=None):
        B,_, H, W =im0.shape
        H=H//8
        W=W//8
        
        h = self.inet(im0,im1)

        flow = None
        if bkt:
            bkt.push_namespace('g',scale_factor=8)

        flow_list = []

        corr = corr_fn(None, bkt)

        corr = self.corr_manip(corr,corr_fn)

        h = h + self.fuse_block(torch.cat([h,corr],1))
        h = self.act(h)
        h = self.norm(h)
        flow = self.head(h)
        flow_list.append((flow,8))

        if bkt:
            bkt.new_bi_flow(flow,f'flow1')
            bkt.pop_namespace()

        return h,flow,flow_list