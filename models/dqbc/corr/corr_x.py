import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import make_grids, to_relative
from validate.bucket import Bucket
from ..common import conv

class CorrLookupX(nn.Module):
    def __init__(self,cfg,dev='cuda') -> None:
        super().__init__()
        self.cfg = cfg
        r_list = cfg.radius
        s_list = cfg.stride
        lvl_list = cfg.level
        assert len(r_list) == len(s_list) and len(s_list) == len(lvl_list)
        pts = {}
        max_lvl = 0
        for r,s,lvl in zip(r_list,s_list,lvl_list):
            max_lvl = lvl
            if lvl not in pts.keys():
                pts[lvl] = set()
            for x in range(-r,r+1):
                for y in range(-r,r+1):
                    pts[lvl].add((x*s,y*s))
        self.n_levels = max_lvl + 1
        # print(pts)
        self.corr_dim = 0
        for k,v in pts.items():
            # list(v): n,2 
            v = list(v)
            v.sort()
            pts[k] = torch.tensor(list(v),device=dev).float()
            self.corr_dim += 2*len(v)
        self.pts = pts
        
        c = self.corr_dim//2
        self.conv33 = nn.Sequential(
            conv(c,c),
            conv(c,c)
        )
        
        self.ln = nn.LayerNorm(self.corr_dim//2)
    
    def setup(self,f0,f1):
        B,C,H,W = f0.shape
        dev = f0.device
        
        self.shape = (B,C,H,W)
        
        self.grid = make_grids(H,W,dev).view(1,H,W,2).expand(B,-1,-1,-1) # B,H,W,2
        self.rel_grid = to_relative(self.grid,H,W) # B,H,W,2

        f0 = f0.reshape(B,C,H*W)
        f1 = f1.reshape(B,C,H*W)
        corr01 = torch.matmul(f0.transpose(1,2),f1) # B,H0*W0,H1*W1

        corr10 = corr01.transpose(1,2) # B,H1*W1,H0*W0
        
        def corr_pyramid(corr):
            corr = corr.reshape(B,H*W,H,W)
            p = [corr]
            for i in range(self.n_levels-1):
                corr = F.avg_pool2d(corr, 2, stride=2)
                p.append(corr)
            return p
        
        self.corr01_pyramid = corr_pyramid(corr01)
        self.corr10_pyramid = corr_pyramid(corr10)
        
        scale = torch.tensor([2/W,2/H],device=dev).reshape(1,2)

        nb_cfg = []

        self.neighbors = {}
        for lvl,x in self.pts.items():
            n,_ = x.shape
            d_pix = x*(1<<lvl)

            self.neighbors[lvl] = (d_pix*scale).view(1,1,n,2)

            nb_cfg.append(self.neighbors[lvl])
        
        
        nb_cfg = torch.cat(nb_cfg,2) # 1,1,L*N,2
        nb_cfg = nb_cfg.expand(-1,2,-1,-1) # 1,2,L*N,2
        nb_cfg = nb_cfg.reshape(1,-1,1,1,2) # 1,corr_dim(or 2*L*N),1,1,2

        nb_cfg = -0.5*nb_cfg + self.rel_grid.reshape(B,1,H,W,2) 
        # B,C,H,W,2
        nb_cfg = nb_cfg.reshape(-1,H,W,2) # BC,H,W,2
        self.nb_cfg = nb_cfg

    def shift_corr(self, corr):
        B,C,H,W = corr.shape
        corr = corr.reshape(B*C,1,H,W)
        nb_cfg = self.nb_cfg # BC,H,W,2
        m_corr = F.grid_sample(corr,nb_cfg,mode='bilinear',align_corners=False)
        # BC,1,H,W
        m_corr = m_corr.reshape(B,C,H,W)
        return m_corr

    @classmethod
    def corr_dim(cls,cfg):
        c = 0
        pts = {}
        for r,s,lvl in zip(cfg.radius,cfg.stride,cfg.level):
            if lvl not in pts.keys():
                pts[lvl] = set()
            for x in range(-r,r+1):
                for y in range(-r,r+1):
                    pts[lvl].add((x*s,y*s))
        for pset in pts.values():
            c += len(pset)
        return 2*c
    
    def forward(self,_flow,bkt:Bucket=None):
        B,_,H,W = self.shape
 
        rel_coords = self.rel_grid.reshape(B*H*W,1,1,2)
              
        corr01_pyr = []
        corr10_pyr = []
        for lvl, neighbors in self.neighbors.items():
            h = H//(1<<lvl)
            w = W//(1<<lvl)
            corr01 = self.corr01_pyramid[lvl].reshape(B*H*W,1,h,w) # BH0W0,1,H1,W1
            corr10 = self.corr10_pyramid[lvl].reshape(B*H*W,1,h,w) # BH1W1,1,H0,W0
            
            grid = rel_coords + neighbors
            # B*H*W, 1, n, 2

            # print(grid.shape, corr01.shape)            
            corr01 = F.grid_sample(corr01,grid,mode='bilinear',align_corners=False)
            # B*H0*W0,1,1,n
            corr10 = F.grid_sample(corr10,grid,mode='bilinear',align_corners=False)
            # B*H1*W1,1,1,n
            
            corr01 = corr01.view(B*H*W,-1)
            corr10 = corr10.view(B*H*W,-1)
                
            corr01_pyr.append(corr01)
            corr10_pyr.append(corr10)

        corr01 = torch.cat(corr01_pyr,dim=-1)
        corr10 = torch.cat(corr10_pyr,dim=-1)
        # B*H*W, L*N

        
        corr01 = corr01.reshape(B,H,W,-1).permute(0,3,1,2)
        corr01 = corr01 + self.conv33(corr01)
        corr01 = corr01.permute(0,2,3,1).reshape(B*H*W,-1)

        corr10 = corr10.reshape(B,H,W,-1).permute(0,3,1,2)
        corr10 = corr10 + self.conv33(corr10)
        corr10 = corr10.permute(0,2,3,1).reshape(B*H*W,-1)
        
        corr01 = self.ln(corr01)
        corr10 = self.ln(corr10)

        corr01 = corr01.reshape(B,H,W,-1).permute(0,3,1,2)
        corr10 = corr10.reshape(B,H,W,-1).permute(0,3,1,2)

        corr = torch.cat([corr01,corr10],dim=1) # B,2*L*N,H,W
        
        return corr

