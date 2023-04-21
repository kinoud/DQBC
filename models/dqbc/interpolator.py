import torch
import torch.nn as nn
import torch.nn.functional as F
from .warplayer import warp
from .vfi_model import VFIModel
from validate.bucket import Bucket

class Interpolator(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.img_size = (0,0)
        self.size_mod = cfg.size_mod
        self.vfi_model = VFIModel(cfg)

    def forward(self, im0, im1, gt=None, bkt:Bucket=None):
        '''
        Args:
            im0, im1: B,3,H,W
        '''
        b,_,h,w = im0.shape
        mod = self.size_mod
        
        if h%mod>0 or w%mod>0:
            pad = ((mod-h%mod)%mod,(mod-w%mod)%mod)
            im0 = F.pad(im0,(0,pad[1],0,pad[0]))
            im1 = F.pad(im1,(0,pad[1],0,pad[0]))
        else:
            pad = None

        flow,mask,res,output_tea = self.vfi_model(im0,im1,gt,bkt=bkt)

        wp0 = warp(im0,flow[:,0:2])
        wp1 = warp(im1,flow[:,2:4])

        raw = mask*wp0 + (1-mask)*wp1
        final = raw + res

        pred = {
            'raw':raw,
            'final':final,
            'mask':mask,
            'wp0':wp0,
            'wp1':wp1, 
            'bw_flo': flow[:,0:2], 
            'fw_flo': flow[:,2:4]
            }

        pred.update(output_tea)

        if pad!=None:
            for k,v in pred.items():
                pred[k] = v[:,:,0:h,0:w]

        if bkt:
            flow = torch.cat([pred['bw_flo'],pred['fw_flo']],1)
            bkt.new_bi_flow(flow,'flow')
        
        return pred