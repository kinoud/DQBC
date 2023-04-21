import torch
import torch.nn as nn
from ..warplayer import warp
from ..common import convex_upsample,conv,deconv,make_conv_block,mlp
from validate.bucket import Bucket, plain_up

class FlowDecoder(nn.Module):
    def __init__(self, in_c, up=True, flow_in=False, delta=True) -> None:
        super().__init__()
        self.flow_in = flow_in
        assert up==True
        if delta:
            nd = 4
        else:
            nd = 0
        self.delta = delta
        if not flow_in:
            self.decoder = nn.Conv2d(in_c,nd+4*9,3,1,1)
        else:
            self.decoder = nn.Conv2d(in_c+4,nd+4*9,3,1,1)
        
    def forward(self, flow, x):
        '''
        Args:
            flow: B,4,H,W
        '''
        if self.flow_in:
            x = torch.cat([flow,x],1)
        x = self.decoder(x)

        if self.delta:
            delta_flow, up_mask = x.split([4,36],1) # B,*,H,W
        else:
            delta_flow, up_mask = 0, x
        
        flow = flow + delta_flow
        return flow, up_mask

def make_norm(t,c):
    if t=='batch':
        return nn.BatchNorm2d(c)
    elif t=='none':
        return nn.Identity()
    else:
        raise NotImplementedError

class UpBlock(nn.Module):
    def __init__(self,cfg, scale, in_c, h_c, out_c, mlp_hidden_scale,res=False,last=False,next_up=True,delta=True) -> None:
        super().__init__()

        self.norm1 = make_norm(cfg.norm, h_c)
        self.norm2 = make_norm(cfg.norm, h_c)

        self.mlp = mlp(in_c,mlp_hidden_scale*h_c,h_c)
        self.conv_block = make_conv_block(cfg.conv_block, h_c)
        
        self.anchor_flows = False

        self.flow_decoder = FlowDecoder(h_c,up=next_up,flow_in=False,delta=delta)
        
        self.next_up = next_up
        if (not last) and next_up:
            self.up = deconv(h_c,out_c)
        self.last = last
        
    def forward(self, flow, x, c0, c1, other_cats=[], bkt:Bucket=None):

        w0 = warp(c0,flow[:,0:2])
        w1 = warp(c1,flow[:,2:4])

        skip = x if x!=None else 0

        inp = [x,flow,w0,w1,*other_cats]
        inp = [y for y in inp if y!=None]

        x = torch.cat(inp,1)
        x = self.norm1(self.mlp(x)) + skip

        x = x + self.norm2(self.conv_block(x))
        
        flow, up_mask = self.flow_decoder(flow,x)
        if self.next_up:
            flow_up = convex_upsample(flow,up_mask,upscale=2,kernel=3)
            flow_up[:,:4] = flow_up[:,:4]*2.0
        else:
            flow_up = None
        
        if bkt:
            bkt.new_bi_flow(flow,'flow_m')
            if self.next_up:
                bkt.new_bi_flow(flow_up,'flow_up',scale_factor=0.5)
        
        if (not self.last) and self.next_up:
            x = self.up(x)

        return flow, flow_up, x

def make_upblock(cfg, scale, c_ctx, c_h, c_out, res=True):
    return UpBlock(cfg, scale, 4+c_ctx*2+(c_h if res else 0),c_h,c_out,cfg.mlp_ratio)

class FlowUp(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        c = [3]+cfg.cnet.dims
        d = cfg.dims
        
        self.up3 = make_upblock(cfg,8,c[3],d[3],d[2],res=False)
        self.up2 = make_upblock(cfg,4,c[2],d[2],d[1])
        self.up1 = make_upblock(cfg,2,c[1],d[1],d[0])
        
        out_c = 2

        self.out_c = out_c

        self.conv_output = nn.Sequential(
            conv(4+d[0]+2*c[0],d[0],3,1,1),
            conv(d[0],out_c))

    def forward(self,h,im0,im1,flow,mask,c0,c1,bkt:Bucket=None):
        '''
        h: /8
        flow: /8
        c0, c1: [/2,/4,/8]
        others: /1
        '''

        if bkt:
            bkt.new_rgb((im0+im1)*0.5,'warp_')
            wp0 = warp(im0,8*plain_up(flow[:,0:2],8))
            wp1 = warp(im1,8*plain_up(flow[:,2:4],8))
            bkt.new_rgb((wp0+wp1)*0.5,'warp0')
            bkt.push_namespace('r0',8)

        flow_list = []

        M = None
        
        flow_lr, flow, M = self.up3(flow, M, c0[-1], c1[-1], bkt=bkt)

        flow_list.append((flow,4))
        
        if bkt:
            bkt.pop_namespace()
            wp0 = warp(im0,4*plain_up(flow[:,0:2],4))
            wp1 = warp(im1,4*plain_up(flow[:,2:4],4))
            bkt.new_rgb((wp0+wp1)*0.5,'warp1')
            bkt.push_namespace('r1',4)
            
        flow_lr, flow, M = self.up2(flow, M, c0[-2], c1[-2], bkt=bkt)

        flow_list.append((flow,2))
        
        if bkt:
            bkt.pop_namespace()
            wp0 = warp(im0,2*plain_up(flow[:,0:2],2))
            wp1 = warp(im1,2*plain_up(flow[:,2:4],2))
            bkt.new_rgb((wp0+wp1)*0.5,'warp2')
            bkt.push_namespace('r2',2)
            
        flow_lr, flow, M = self.up1(flow, M, c0[-3], c1[-3], bkt=bkt)

        flow_list.append((flow,1))
        
        wp0 = warp(im0,flow[:,0:2])
        wp1 = warp(im1,flow[:,2:4])

        if bkt:
            bkt.pop_namespace()
            bkt.new_rgb((wp0+wp1)*0.5,'warp3')

        M = torch.cat([flow,M,wp0,wp1],1)

        tmp = self.conv_output(M)
        if self.out_c == 2:
            mask, res = tmp, None
        elif self.out_c == 5:
            mask, res = tmp.split([2,3],1)
        
        return flow, mask, res, flow_list