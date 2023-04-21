def make_feeder(cfg):
    '''
    Only for train
    '''
    return vfi2_feeder

def vfi2_feeder(data_blob):
    frames,gt = data_blob
    f0 = frames[:,0:3].cuda()
    f1 = frames[:,3:6].cuda()
    gt = gt.cuda()
    return (f0,f1,gt),gt