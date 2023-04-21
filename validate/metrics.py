import numpy as np
import math
from .pytorch_msssim import ssim_matlab


def calculate_psnr(gt, pred, quant=True):
    '''
    Args:
        gt, pred: [0,255] any dtype
    '''
    if quant:
        pred = np.round(pred).astype('uint8')/255.0
    else:
        pred = pred/255.0
    gt = gt/255.0
    psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
    return psnr

def calculate_ssim(gt, pred):
    return ssim_matlab(gt, pred)

def calculate_ie(gt, pred, quant=True):
    pred = np.round(pred)
    return np.abs((pred - gt * 1.0)).mean()
