
from validate import make_validate_func
from models import make_model, model_profile
from utils.config import children, make_config
from functools import partial
import torch
import os.path as osp
import argparse

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',help='.yaml config file path')
    parser.add_argument('--gpu_id',type=int,default=0)
    args = parser.parse_args()
    cfg_file = args.config
    dev_id = args.gpu_id
    torch.cuda.set_device(dev_id)

    cfg = make_config(cfg_file)
    
    log_path = osp.join(cfg.exp_root,'val.log')
    profile = model_profile(cfg.model)
    
    print(profile+'\n')
    log = open(log_path, 'a')
    log.write(profile+'\n')
    log.close()
    
    model = make_model(cfg.model)
    model.cuda()
    validate_fn = make_validate_func(cfg)
    
    for tag, val_cfg in children(cfg.val):
        validate_fn(val_cfg, model, log_file_path=log_path, val_tag=tag)
    
    
        
    