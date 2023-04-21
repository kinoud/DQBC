from utils.config import children
from utils.count import check_step

def make_validate_func(cfg):
    val_fn = cfg.model.name
    if hasattr(cfg,'val_fn'):
        val_fn = cfg.val_fn
        
    if val_fn == 'std':
        from validate.validate_std import validate as func_val
        return func_val
    else:
        raise NotImplementedError

class Validator:
    def __init__(self, cfg, model, iters_per_epoch) -> None:
        self.cfg = cfg
        self.model = model
        self.val_func = make_validate_func(cfg)
        self.ipe = iters_per_epoch
    
    def val(self, step):
        results = []
        for tag, cfg in children(self.cfg):

            if hasattr(cfg,'val_after_epoch'):
                after_epoch = cfg.val_after_epoch
            else:
                after_epoch = False
                
            if check_step(step,cfg.val_freq,self.ipe,after_epoch):
                res = self.val_func(cfg,self.model.module)
                tagged_res = dict()
                for k,v in res.items():
                    tagged_res[k+'/'+tag] = v
                results.append(tagged_res)
        return results
