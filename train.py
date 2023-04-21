from datas.data_feeder import make_feeder
from utils.config import make_config
from models import make_model, model_profile
from datas import make_dataloader
from utils.count import check_step
from validate import Validator, make_validate_func
from losses import make_loss
import torch
import torch.optim as optim
from utils.logger import Logger
import os.path as osp
from functools import partial
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import time
import numpy as np
import math
import random

debug = False

def warm_cosine_scheduler(step,total_steps,peak=3e-4,bottom=3e-5,peak_step=2000):
    # step ~ [0,total_steps)
    step = step + 1
    if step < peak_step:
        mul = step / peak_step
        return peak * mul
    elif step >= total_steps:
        return bottom
    else:
        mul = np.cos((step - peak_step) / (total_steps - peak_step) * math.pi) * 0.5 + 0.5
        return (peak - bottom) * mul + bottom

def multi_step_scheduler(step,initial,steps,decay):
    lr = initial
    for s in steps:
        if step >= s:
            lr *= decay
        else:
            break
    return lr

def make_optimizer(cfg, model):
    """ Create the optimizer and learning rate scheduler """
    
    if cfg.lr.type == 'multi_step':
        lr = cfg.lr.initial
        scheduler = partial(multi_step_scheduler,initial=cfg.lr.initial,steps=cfg.lr.steps,decay=cfg.lr.decay)
    elif cfg.lr.type == 'warm_cosine':
        lr = 1e-6
        scheduler = partial(warm_cosine_scheduler,total_steps=cfg.lr.total_steps,peak=cfg.lr.peak,bottom=cfg.lr.bottom,peak_step=cfg.lr.peak_step)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr,  weight_decay=cfg.wdecay)
    
    return optimizer, scheduler

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')

    args = parser.parse_args()
    

    dist.init_process_group(backend='nccl')
    
    local_rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    
    if local_rank==0:
        cfg = make_config(cfg_file=args.config)
    dist.barrier()

    if local_rank>0:
        cfg_file = args.config
        cfg_file = osp.join(osp.dirname(cfg_file),osp.basename(cfg_file)+'.full')
        cfg = make_config(cfg_file=cfg_file,launch_experiment=False)

    if hasattr(cfg.train, 'manual_seed'):
        seed = cfg.train.manual_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if hasattr(cfg.train,'debug'):
        debug = cfg.train.debug



    model = make_model(cfg.model)
    model.train()
    model = DDP(model)
    
    sampler, train_loader = make_dataloader(cfg.train)

    iters_per_epoch = len(train_loader)

    optimizer, scheduler = make_optimizer(cfg.train, model)

    if cfg.init_step!=0:
        opt_path = cfg.model.pretrained
        opt_path = opt_path.replace('.pth','.opt.pth')
        print('loading optimizer checkpoint from {}'.format(opt_path))
        dict1 = torch.load(opt_path, map_location='cpu')
        optimizer.load_state_dict(dict1)

    total_steps = cfg.init_step
    
    if local_rank==0:
        logger = Logger(cfg.log, model, scheduler, init_step = total_steps)
        
        validator = Validator(cfg.val, model, iters_per_epoch)

        profile = model_profile(cfg.model)
        
        logger.print(profile+'\n')
        if hasattr(model.module,'info'):
            logger.print(model.module.info())
        
    
            
    
    
    

    loss_fn = make_loss(cfg)
    validate_fn = make_validate_func(cfg)
    
    data_feeder = make_feeder(cfg)

    
    should_keep_training = True

    gpu_time = 0
    cpu_time = 0

    t0 = time.time()

    if debug:
        debug_cnt = 0

    epoch = 0
    while should_keep_training:
        sampler.set_epoch(epoch)
        epoch += 1
        for i_batch, data_blob in enumerate(train_loader):
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler(total_steps)
            
            optimizer.zero_grad()
            
            for_model, for_loss = data_feeder(data_blob)

            


            # print(f'{local_rank}: data device {f0.device}')

            cpu_time += time.time()-t0

            t0 = time.time()
            output = model(*for_model)

            loss,metrics = loss_fn(output,for_loss)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)

            optimizer.step()

            gpu_time += time.time() - t0

            # print(f'{local_rank}: ?')
            if local_rank==0:

                if logger.step(metrics,'gpu_t %.1fs, cpu_t %.1fs'%(gpu_time,cpu_time)):
                    cpu_time = 0
                    gpu_time = 0
                
                val_results = validator.val(total_steps)
                
                for res in val_results:
                    logger.write_dict(res)
                    logger.print('validate: '+repr(res))

                if hasattr(cfg.train, 'save_after_epoch'):
                    after_epoch = cfg.train.save_after_epoch
                else:
                    after_epoch = False

                if check_step(total_steps, cfg.train.save_freq, iters_per_epoch, after_epoch):
                    save_path = osp.join(cfg.ckp_root ,'%d_%s.pth' % (total_steps+1, cfg.exp_name))
                    torch.save(model.state_dict(), save_path)
                    logger.print('save model %s'%save_path)
                    
                    opt_path = osp.join(cfg.ckp_root, '%d_%s.opt.pth' % (total_steps+1, cfg.exp_name))
                    torch.save(optimizer.state_dict(), opt_path)
                    logger.print('save optimizer %s'%opt_path)
            

            t0 = time.time()

            total_steps += 1

            if total_steps > cfg.train.num_steps:
                should_keep_training = False
                break
        dist.barrier()

    if local_rank==0:
        logger.close()
        save_path = osp.join(cfg.ckp_root ,'%d_%s.pth' % (total_steps+1, cfg.name))
        torch.save(model.state_dict(), save_path)
        logger.print('save model %s'%save_path) 

