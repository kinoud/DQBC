from torch.utils.tensorboard import SummaryWriter
import time
import os.path as osp

class Logger:
    def __init__(self,cfg, model, lr_fn, init_step=0):
        self.model = model
        self.total_steps = init_step
        self.lr_fn = lr_fn
        self.running_loss = {}
        self.info_str = None
        self.writer = None
        self.sum_freq = cfg.sum_freq
        self.log_file = osp.join(cfg.exp_root,cfg.exp_name+'.log')
        if hasattr(cfg,'tb_log_dir'):
            self.log_dir = cfg.tb_log_dir
        else:
            self.log_dir = osp.join('runs',cfg.exp_name)

        if self.total_steps!=0:
            self.print('RESUME from step %d'%self.total_steps)

    def _print_training_status(self):
        
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps, self.lr_fn(self.total_steps-1))
        
        metrics_str = ''
        for k,v in self.running_loss.items():
            metrics_str += '%s %.4f, '%(k,v/self.sum_freq)
        
        # print the training status
        self.print(training_str + metrics_str + self.info_str)

        writer = self.get_writer()

        for k in self.running_loss:
            writer.add_scalar(k, self.running_loss[k]/self.sum_freq, self.total_steps)
            self.running_loss[k] = 0.0

    def print(self, msg):
        msg = time.strftime('%Y-%m-%d %X ') + msg
        print(msg)
        with open(self.log_file,'a') as f:
            f.write(msg+'\n')

    def step(self, metrics, info):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.sum_freq == 0:
            self.info_str = info
            self._print_training_status()
            self.running_loss = {}
            return True
        return False

    def get_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        return self.writer

    def write_dict(self, results):
        writer = self.get_writer()

        for key in results:
            writer.add_scalar(key, results[key], self.total_steps)


    def close(self):
        self.get_writer().close()