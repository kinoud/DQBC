import torch.nn as nn
import torch
from .extractor import BasicEncoder
from .flow_up import make_flowup

from validate.bucket import Bucket
from .flow_gen import make_flowgen
from .synth import make_synth
from .context import make_cnet
from .flow_tea import make_flowtea
from .corr import make_corr_fn
import torch.nn.functional as F

class VFIModel(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.flowgen = make_flowgen(cfg.flowgen)
        
        self.fnet = BasicEncoder(input_dim=3,output_dim=128)
        self.corr_fn = make_corr_fn(cfg.corr)

        self.cnet = make_cnet(cfg.cnet,3)
        self.flowup = make_flowup(cfg.flowup)

        self.flowtea = make_flowtea(cfg.flowtea)
        self.synth = make_synth(cfg.synth)

    def forward(self,im0,im1,gt=None,bkt:Bucket=None):
        
        down_scaled = -1 # -1: undecided; 0: no; 1: yes
        down_scale_th = self.cfg.flowgen.down_scale_th

        if self.training or down_scale_th==-1:
            down_scaled = 0

        if down_scaled == -1:
            s = 2
            down0 = F.avg_pool2d(im0,kernel_size=s,stride=s,padding=0,count_include_pad=False)
            down1 = F.avg_pool2d(im1,kernel_size=s,stride=s,padding=0,count_include_pad=False)
            _,_,Hd,Wd = down0.shape

            mod = self.cfg.size_mod

            if Hd%mod>0 or Wd%mod>0:
                pad = ((mod-Hd%mod)%mod,(mod-Wd%mod)%mod)
                down0 = F.pad(down0,(0,pad[1],0,pad[0]))
                down1 = F.pad(down1,(0,pad[1],0,pad[0]))
            else:
                pad = None

            (f0,_),(f1,_) = self.fnet([down0,down1])
            self.corr_fn.setup(f0,f1)
            c0, c1 = self.cnet([down0,down1])
            h, flow, flowgen_flow_list = self.flowgen(down0,down1,self.corr_fn,c0,c1,bkt=bkt)

            if torch.max(flow)*8*s < down_scale_th:
                down_scaled = 0
            else:
                # h,im0,im1,flow,mask,bkt:Bucket=None
                flow, mask, res, flowup_flow_list = self.flowup(h,down0,down1,flow,None,c0,c1,bkt=bkt)

                if pad!=None:
                    flow = flow[:,:,0:Hd,0:Wd]
                    mask = mask[:,:,0:Hd,0:Wd]

                flow = s*F.interpolate(flow,scale_factor=s,mode='bilinear',align_corners=False)
                mask = F.interpolate(mask,scale_factor=s,mode='bilinear',align_corners=False)

                c0, c1 = self.cnet([im0,im1])
                res, mask = self.synth(im0,im1,flow,mask,res,c0,c1)
        
        if down_scaled == 0:
            (f0,_),(f1,_) = self.fnet([im0,im1])

            self.corr_fn.setup(f0,f1)
            h, flow, flowgen_flow_list = self.flowgen(im0,im1,self.corr_fn,None,None,bkt=bkt)
            c0, c1 = self.cnet([im0,im1])
            # h,im0,im1,flow,mask,bkt:Bucket=None
            flow, mask, res, flowup_flow_list = self.flowup(h,im0,im1,flow,None,c0,c1,bkt=bkt)
            res, mask = self.synth(im0,im1,flow,mask,res,c0,c1)


        if bkt:
            bkt.push_namespace('tea')
        if gt!=None and self.flowtea!=None:
            output_tea = self.flowtea(im0,im1,flow,mask,gt,flowgen_flow_list+flowup_flow_list,bkt=bkt)
        else:
            output_tea = {}
        if bkt:
            bkt.pop_namespace()

        mask = torch.softmax(mask,1)[:,[0],:,:]

        if bkt != None:
            bkt.push_namespace('r3')
            bkt.new_gray(mask,'mask')
            bkt.pop_namespace()
                
        return flow,mask,res,output_tea