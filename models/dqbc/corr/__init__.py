import torch.nn as nn
from .corr_x import CorrLookupX

def make_corr_fn(cfg):
    if cfg.arch == 'x':
        return CorrLookupX(cfg)
    else:
        raise NotImplementedError

def get_corr_dim(cfg):
    if cfg.arch == 'x':
        d = CorrLookupX.corr_dim(cfg)
        return d, d
    else:
        raise NotImplementedError