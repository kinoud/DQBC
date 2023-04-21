import losses.losses as losses

def make_loss(cfg):
    return losses.make_vfi2_loss(cfg.train.loss)
        