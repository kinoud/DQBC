
def check_step(step,freq,iters_per_epoch,after_epoch):
    
    if after_epoch:
        ok = True

        if step%iters_per_epoch != iters_per_epoch - 1:
            ok = False
        if step - (((step+1)//freq)*freq-1) >= iters_per_epoch:
            ok = False
    else:
        ok = (step%freq == freq - 1)  # ok=True when step = freq-1, 2*freq-1, ...
    
    return ok