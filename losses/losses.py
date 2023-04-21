import torch.nn.functional as F
from utils.config import children
from functools import partial

def _make_loss(losses):
    def f(pred,gt):
        metrics = dict()
        loss = 0
        for tag, L, w in losses:
            tag = 'loss/'+tag
            metrics[tag] = L(pred,gt)
            loss += metrics[tag]*w
        return loss, metrics
    return f

def make_vfi2_loss(cfg):
    loss_fn = {
        'l1': lambda pred, gt, opt : F.l1_loss(pred['final'],gt),
        'l1_tea': lambda pred,gt, opt: F.l1_loss(pred['merged_tea'],gt),
        'distill': lambda pred, gt, opt : pred['loss_distill'],
        }
    losses = []
    for tag, opt in children(cfg):
        losses.append((tag,partial(loss_fn[tag],opt=opt),opt.w))
    return _make_loss(losses)