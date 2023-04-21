from models import make_model, model_profile
from utils.config import make_config
import torch
import argparse
from datas.utils import imread_rgb
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import os 

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',help='.yaml config file path')
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--im0',type=str)
    parser.add_argument('--im1',type=str)
    parser.add_argument('--output_dir',type=str)
    args = parser.parse_args()
    cfg_file = args.config
    dev_id = args.gpu_id
    torch.cuda.set_device(dev_id)

    cfg = make_config(cfg_file, launch_experiment=False)
    
    print(model_profile(cfg.model))
    
    model = make_model(cfg.model)
    model.cuda()
    model.eval()

    with torch.no_grad():
        im0 = TF.to_tensor(imread_rgb(args.im0))[None].cuda()
        im1 = TF.to_tensor(imread_rgb(args.im1))[None].cuda()
        pred = model(im0,im1)['final']
        pred = pred[0].cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)*255
        Image.fromarray(np.uint8(pred)).save(os.path.join(args.output_dir,'interp.png'))
    



    


    

    
    
        
    