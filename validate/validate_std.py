import torch
from validate.metrics import calculate_psnr, calculate_ssim, calculate_ie
from validate.bucket import Bucket
from .utils import make_validation_set
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os.path as osp
import os
from PIL import Image
import numpy as np
from utils.flow_viz import flow_to_image
from time import time
from .utils import log_print
import math
import torchvision.transforms.functional as TF


@torch.no_grad()
def validate(cfg, model, log_file_path=None, val_tag=''):

    log_print(log_file_path,f'start validating {val_tag}')

    viz_tags = []
    if hasattr(cfg,'viz_tags'):
        viz_tags = cfg.viz_tags
    
    quant = True
    if 'nq' in viz_tags:
        quant = False        

    viz_items = ['frame1','frame2','frame3','gt']
    
    if hasattr(cfg,'viz_items'):
        viz_items = cfg.viz_items
    
    if hasattr(cfg,'val_aug'):
        cfg_aug = cfg.val_aug
    else:
        cfg_aug = []
    
    augs = [['O']]
    if 'T' in cfg_aug:
        augs += [a+['T'] for a in augs]
    if 'R' in cfg_aug:
        augs += [a+['R'] for a in augs]
    if 'F' in cfg_aug:
        augs += [a+['F'] for a in augs]

    log_print(log_file_path, f'augs: {[",".join(a) for a in augs]}')

    model.eval()
    val_dataset, ind = make_validation_set(cfg)

    time_infer = 0

    psnr_list = []
    ssim_list = []
    ie_list = []


    
    for i in tqdm(ind):
        frames,gt,name = val_dataset[i]
        c,h,w = frames.shape
        # print(frames.shape)
        frames = frames.view((2,3,h,w))
        frame0 = frames[0][None].cuda()
        frame1 = frames[1][None].cuda()
        # gt = gt[None].cuda()
        
        bkt = Bucket() if cfg.viz else None

        torch.cuda.synchronize()
        t0 = time()

        pred_list = []
        for a in augs:
            if len(augs)>1 and bkt!=None:
                bkt.push_namespace(','.join(a))

            if 'T' in a:
                af0, af1 = frame1, frame0
            else:
                af0, af1 = frame0, frame1
            
            if 'R' in a:
                af0 = torch.rot90(af0,1,[2,3])
                af1 = torch.rot90(af1,1,[2,3])

            if 'F' in a:
                af0 = torch.flip(af0,[-1])
                af1 = torch.flip(af1,[-1])

            pred = model(
                af0,
                af1, # [1,3,H,W]
                bkt=bkt
                )
            
            if 'F' in a:
                if isinstance(pred, dict):
                    for k,v in pred.items():
                        pred[k] = torch.flip(v,[-1])
                else:
                    pred = torch.flip(pred,[-1])

            if 'R' in a:
                if isinstance(pred, dict):
                    for k,v in pred.items():
                        pred[k] = torch.rot90(v,3,[2,3])
                else:
                    pred = torch.rot90(pred,3,[2,3])

            pred_list.append(pred)

            if len(augs)>1 and bkt!=None:
                bkt.pop_namespace()
        
        if len(augs)==1:
            pred = pred_list[0]
        else:
            if isinstance(pred_list[0], dict):
                pred = dict()
                for k,v in pred_list[0].items():
                    pred[k] = torch.mean(torch.stack([p[k] for p in pred_list]),0)
            else:
                pred = torch.mean(torch.stack(pred_list),0)

        torch.cuda.synchronize()
        time_infer += time()-t0
        
        # gt = gt[0].cpu().numpy().transpose(1, 2, 0)
        
        if isinstance(pred,dict):
            pred = pred['final']

        ssim_list.append(
            calculate_ssim(
                torch.tensor(gt.transpose(2,0,1)/255.).to('cuda').float().unsqueeze(0),
                torch.tensor((torch.round(pred*255)/255.) if quant else pred)
            ).detach().cpu().numpy()
        )

        pred = pred[0].cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)

        pred255 = pred*255
        psnr_list.append(calculate_psnr(gt, pred255,quant))
        ie_list.append(calculate_ie(gt,pred255,quant))
        psnr = psnr_list[-1]
        ssim = ssim_list[-1]
        ie = ie_list[-1]
        

        gt = gt/255.0
        # psnr = calculate_psnr(gt, pred)
        # psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())

        # psnr_list.append(psnr)

        if cfg.viz:
            save_dir = osp.join(cfg.viz_root,name)
            if not osp.isdir(save_dir):
                os.mkdir(save_dir)
            f0 = frame0[0].cpu().numpy().transpose(1,2,0)
            f1 = frame1[0].cpu().numpy().transpose(1,2,0)

            if 'frame1' in viz_items:
                Image.fromarray(np.uint8(f0*255)).save(osp.join(save_dir,'frame1.jpg'))
            
            if 'frame3' in viz_items:
                Image.fromarray(np.uint8(f1*255)).save(osp.join(save_dir,'frame3.jpg'))
            
            if 'frame2' in viz_items:
                Image.fromarray(np.uint8(pred*255)).save(osp.join(save_dir,f'frame2.jpg'))
            
            if 'gt' in viz_items:
                Image.fromarray(np.uint8(gt*255)).save(osp.join(save_dir,'gt.jpg'))

            bkt.save_imgs(save_dir)
            bkt.save_tensors(save_dir)
            
            with open(osp.join(save_dir,'result.txt'),'w') as f:
                f.write(f'psnr={psnr} ssim={ssim} ie={ie}\n')

        if log_file_path is not None:
            with open(log_file_path,'a') as f:
                f.write(f'{name}\tpsnr={psnr} ssim={ssim} ie={ie}\n')
    
    psnr = np.mean(psnr_list)
    ssim = np.mean(ssim_list)
    ie = np.mean(ie_list)
    report = f'{val_tag} average validation psnr: {psnr} ssim={ssim} ie={ie}\taverage inference time: {time_infer/len(ind)} s/image'
    print(report)
    if log_file_path is not None:
        with open(log_file_path,'a') as f:
            f.write(report)
    model.train()
    return {'val_psnr':psnr}