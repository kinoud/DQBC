import datas
import numpy as np
from numpy.random import RandomState
import torch
import torch.nn.functional as F
from PIL import Image
import os.path as osp

def log_print(log_file_path, msg):
    print(msg)
    if log_file_path is not None:
        with open(log_file_path,'a') as f:
            f.write(msg+'\n')

def save_image(im, p, rgb_order='rgb'):
    '''
    Args:
        im: 0~1
    '''
    if rgb_order == 'rgb':
        pass
    elif rgb_order == 'bgr':
        im = im[:,:,::-1]
    else:
        raise NotImplementedError
    Image.fromarray(np.uint8(im*255)).save(p)

def make_validation_set(cfg):
    if cfg.data.name == 'Vimeo90K':
        from datas.Vimeo90K import Vimeo_validation as ValSet
    elif cfg.data.name == 'SNU_FILM':
        from datas.SNU_FILM import SNU_FILM as ValSet
    elif cfg.data.name == 'UCF101':
        from datas.ucf101 import UCF101_test as ValSet
    elif cfg.data.name == 'MiddleBury':
        from datas.MiddleBury_Other import MiddelBuryOther as ValSet
    else:
        raise NotImplementedError
    
    val_dataset = ValSet(cfg.data)
    num_val = cfg.num_val if hasattr(cfg,'num_val') else None
    ind = np.arange(len(val_dataset))
    if num_val is not None:
        rs = RandomState(cfg.rand_seed)
        rs.shuffle(ind)
    else:
        num_val = len(val_dataset)
    ind = ind[0:num_val]
    return val_dataset, ind

def save_deep_feature_map(fmap:torch.Tensor,dir):
    '''
    Args:
        fmap: C,H,W
    '''
    C,H,W = fmap.shape
    fmap = fmap.reshape(C,H*W)
    fmin = torch.min(fmap,dim=1).values[:,None]
    fmax = torch.max(fmap,dim=1).values[:,None]
    fmap = (fmap-fmin)/(fmax-fmin)
    fmap = fmap.reshape(C,H,W)
    # print(torch.max(fmap.view(C,-1),1).values)
    for i in range(C):
        Image.fromarray(np.uint8(fmap[i].numpy()*255)).save(osp.join(dir,f'{i}.jpg'))