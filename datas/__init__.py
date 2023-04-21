from .Vimeo90K import Vimeo_train 
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler


def make_dataloader(cfg):
    if cfg.data.name=='Vimeo90K':
        train_dataset = Vimeo_train(cfg.data)
    else:
        raise NotImplementedError
    
    sampler = DistributedSampler(train_dataset)
    train_loader = data.DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    return sampler, train_loader