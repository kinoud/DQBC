import argparse
import yaml
import os.path as osp
import os
import shutil
from glob import glob

class MyLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        self._ref_cache = {}
        super().__init__(stream)


    def include(self, node):
        
        node = node.value.split('@')
        spec = node[0].split('.')
        fname = node[1]

        fname = os.path.join(self._root, fname)
        if fname in self._ref_cache.keys():
            ref = self._ref_cache[fname]
        else:
            with open(fname, 'r') as fr:
                ref = yaml.load(fr, MyLoader)
            self._ref_cache = ref
            
        for x in spec:
            ref = ref[x]
        return ref

MyLoader.add_constructor('!ref', MyLoader.include)

class Config(object):
    
    def __init__(self, parent=None):
        super().__init__()
        self._parent = parent

    def __getattr__(self, name):
        if self.__dict__.get(name) is not None:
            return self.__dict__[name]
        if self.__dict__.get('_parent') is not None:
            return self._parent.__getattr__(name)
        raise AttributeError('no attribute "%s"'%name)
    
    def __hasattr__(self, name):
        try:
            getattr(self, name)
            return True
        except:
            return False
    
    def _node_name(self):
        if self._parent is None:
            return 'root'
        else:
            parent_name = self._parent._node_name()
            for k,v in self._parent.__dict__.items():
                if v is self:
                    return parent_name+'.'+k

    def __setattr__(self, name, value):
        # print('set',name,'to',value)
        if self.__dict__.get(name) is not None:
            print('config warning: rewriting config item {%s:%s}'%(self._node_name()+'.'+name,repr(value)))
        elif hasattr(self, name):
            print('config warning: creating config item {%s:%s} with the same name as its ancestor nodes'%(self._node_name()+'.'+name,repr(value)))

        self.__dict__[name] = value
    
    def __repr__(self) -> str:
        ans = ''
        for k,v in self.__dict__.items():
            if k=='_parent':
                continue
            ans += k + ': '
            if type(v)!=Config:
                ans += repr(v) +'\n'
            else:
                ans += '\n'
                ans += '\n'.join(['  '+line for line in repr(v).split('\n')])
                ans += '\n'
        return ans[:-1]
    
def children(cfg:Config):
    for k,v in cfg.__dict__.items():
        if k[0]!='_':
            yield k,v

def get_config(cfg:Config, attr:str, default=None):
    if hasattr(cfg,attr):
        return getattr(cfg,attr)
    return default
                
def make_config(cfg_file:str=None,launch_experiment=True):
    '''
    notions: 1. "require" 2. [option] 3. (auto generated)
    
    "model"
    "reset": wether to regenerate experiment dirs and discard the old ones
    [exp_name]: experiment name, same as config file if not assigned
    (exp_name): same with config file name
    (exp_root): experiments/exp_name
    (ckp_root): exp_root/checkpoints
    (viz_root): exp_root/viz
    (init_step): 0 IF reset ELSE search ckp_root 
    '''
    if cfg_file is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config',help='.yaml config file path')
        args = parser.parse_args()
        cfg_file = args.config
    cfg = parse_yaml(cfg_file)

    if not launch_experiment:
        return cfg 

    if not hasattr(cfg,'exp_name'):
        cfg.exp_name = cfg_file.split('/')[-1].split('.')[0]
    cfg.exp_root = osp.join('experiments',cfg.exp_name)
    cfg.ckp_root = osp.join(cfg.exp_root,'checkpoints')
    cfg.viz_root = osp.join(cfg.exp_root,'viz')
    if cfg.reset and osp.isdir(cfg.exp_root):
        ans = input(f'reset experiment "{cfg.exp_name}"? (y/n):')
        if ans!='y':
            print('bye')
            exit()
        shutil.rmtree(cfg.exp_root)
    os.makedirs(cfg.ckp_root,exist_ok=True)
    os.makedirs(cfg.viz_root,exist_ok=True)


    cfg.init_step = 0
    if not cfg.reset:
        ckp_list = glob(osp.join(cfg.ckp_root,'*.pth'))
        ckp_list = [path for path in ckp_list if path[-8:-4]!='.opt']
        ckp_list.sort(key=lambda x:int(x.split('/')[-1].split('_')[0]))
        if len(ckp_list)==0:
            print('no .pth found in %s'%cfg.ckp_root)
            ans = input('is that ok? (y/n):')
            if ans[0]!='y':
                print('bye')
                quit()
        else:
            cfg.init_step = int(ckp_list[-1].split('/')[-1].split('_')[0])
            print('found ckp %s'%ckp_list[-1])
            ans = input('would you like to resume here with init_step=%d? (y/n):'%cfg.init_step)
            if ans[0]!='y':
                print('bye')
                quit()
            cfg.model.pretrained = ckp_list[-1]

    with open(osp.join(cfg.exp_root,cfg_file.split('/')[-1]),'w') as f:
        with open(cfg_file) as g:
            f.write(g.read())

    with open(osp.join(osp.dirname(cfg_file),osp.basename(cfg_file)+'.full'),'w') as f:
        f.write(cfg_to_yaml(cfg))

    return cfg

def cfg_to_yaml(cfg):
    d = dict()

    def update_attr(d, c):
        for k,v in c.__dict__.items():
            if k=='_parent':
                continue
            if type(v)!=Config:
                d[k] = v
            else:
                dv = dict()
                update_attr(dv, v)
                d[k] = dv
    
    update_attr(d,cfg)
    return str(yaml.dump(d))

def parse_yaml(cfg_path):
    cfg = Config()
    ydict = yaml.load(open(cfg_path),MyLoader)

    def update_attr(c, attr_dict):
        for k,v in attr_dict.items():
            setattr(c,k,v)
        for k,v in c.__dict__.items():
            if type(v)==dict:
                c.__dict__[k] = Config(c)
                update_attr(c.__dict__[k],v)
                

    update_attr(cfg,ydict)
    return cfg


if __name__=='__main__':
    # cfg = parse_yaml('configs/demo.yaml')
    cfg = parse_yaml('configs/train_w_sgm.yaml')