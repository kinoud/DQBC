import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}

def get_grid(flow):
    '''
    Args:
        flow: B,*,H,W
    Returns:
        grid: B,2,H,W
    '''
    device = flow.device
    k = (str(flow.device), str(flow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, flow.shape[3], device=device).view(
            1, 1, 1, flow.shape[3]).expand(flow.shape[0], -1, flow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, flow.shape[2], device=device).view(
            1, 1, flow.shape[2], 1).expand(flow.shape[0], -1, -1, flow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)
    return backwarp_tenGrid[k]
    

def warp(tenInput, tenFlow):
    grid = get_grid(tenFlow)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (grid + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

