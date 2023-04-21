import torch
import torch.nn as nn
from .warplayer import warp
import torch.nn.functional as F
from validate.bucket import Bucket
from .context import make_cnet
from .common import deconv, conv, make_conv_block, mlp


def make_synth(cfg):
    return SynthNet(cfg)

class UpBlock(nn.Module):
    def __init__(self, cfg, scale_factor, ctx_c, h_c, out_c, mlp_hidden_scale) -> None:
        super().__init__()
        self.mlp = mlp(ctx_c+h_c,h_c*mlp_hidden_scale,h_c)
        
        self.conv_block = make_conv_block(cfg.conv_block, h_c)

        self.up = deconv(h_c,out_c)

    def forward(self, flow, x, c0, c1, other_cats=[], bkt:Bucket=None):

        w0 = warp(c0,flow[:,0:2])
        w1 = warp(c1,flow[:,2:4])

        inp = [x,w0,w1,*other_cats]
        inp = [y for y in inp if y!=None]
        inp = torch.cat(inp, 1)

        if x == None:
            x = self.mlp(inp)
        else:
            x = self.mlp(inp) + x
        
        x = self.conv_block(x) + x

        x = self.up(x)

        return x
            
class SynthNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        c = [3] + cfg.cnet.dims
        h = cfg.dims
        in_ac = 18
        self.acnet = make_cnet(cfg.acnet,in_ac)
        a = [in_ac]+ cfg.acnet.dims
        ratio = cfg.mlp_ratio

        self.up3 = UpBlock(cfg,8,c[3]*2     ,h[3],h[2],ratio)
        self.up2 = UpBlock(cfg,4,c[2]*2+a[2],h[2],h[1],ratio)
        self.up1 = UpBlock(cfg,2,c[1]*2+a[1],h[1],h[0],ratio)

        self.net_output = nn.Sequential(
            conv(4+h[0]+2*c[0],h[0],3,1,1),
            conv(h[0],5))
    
    def forward(self,im0,im1,flow,mask,res,c0,c1,bkt:Bucket=None):
        flow_list = [flow]
        for _ in range(3):
            flow_list.append(
                0.5*F.interpolate(flow_list[-1],scale_factor=0.5,mode='bilinear',align_corners=False))
        flow_list.reverse()
        flow_list.pop()

        wp0 = warp(im0,flow[:,0:2])
        wp1 = warp(im1,flow[:,2:4])

        ac = torch.cat([x for x in [flow,mask,res,wp0,wp1,im0,im1] if x!=None],1)
        ac = self.acnet(ac)

        R = ac[-1]
        ac = [[x] for x in ac]
        R = self.up3(flow_list[0], R, c0[-1], c1[-1],
            other_cats=[], bkt=bkt)
        R = self.up2(flow_list[1], R, c0[-2], c1[-2],
            other_cats=[*ac[-2]], bkt=bkt)
        R = self.up1(flow_list[2], R, c0[-3], c1[-3],
            other_cats=[*ac[-3]], bkt=bkt)
        
        R = torch.cat([flow,R,wp0,wp1],1)
        d_mask,d_res = torch.split(self.net_output(R), [2,3], 1)
        mask = mask + d_mask
        if res != None:
            res = res + d_res
        else:
            res = d_res
        return res, mask