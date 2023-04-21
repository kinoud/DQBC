
import torch.nn as nn
import torch
import torch.nn.functional as F

def bilinear_pyramid(x, n):
    p = [x]
    for _ in range(n-1):
        p.append(F.interpolate(p[-1],scale_factor=0.5,mode='bilinear',align_corners=False))
    return p

def make_grids(H,W,dev):
    '''
        don't align corners
    Returns:
        grids: [H,W,:]->(x,y)
    '''
    dx = torch.linspace(0.5,W-0.5,W,device=dev)
    dy = torch.linspace(0.5,H-0.5,H,device=dev)
    return torch.stack(torch.meshgrid(dy,dx,indexing='ij')[::-1],dim=-1)

def to_relative(grids,H,W):
    '''
    Args:
        grids: [B,R1,R2,:]->(x,y)
        H: for y
        W: for x
    Returns:
        B,R1,R2,: -> 2*x/W-1,2*y/H-1
    '''
    return 2*grids/(torch.tensor([W,H],device=grids.device).float().view(1,1,1,2))-1
