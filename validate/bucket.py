from utils.flow_viz import flow_to_image
import torch
import numpy as np
from PIL import Image
import os

def plain_up(x,scale_factor):
    '''
    Args:
        x: B,C,H,W
        scale_factor: int
    '''
    x = torch.repeat_interleave(x,int(scale_factor),dim=-2)
    x = torch.repeat_interleave(x,int(scale_factor),dim=-1)
    return x




class Bucket:
    def __init__(self) -> None:
        pass
        self.imgs = dict()
        self.tensors = dict()
        self.random_id = 0
        self._prefix = ['bkt']
        self._scale_factor = [1]
        
    def push_namespace(self,ns,scale_factor=1):
        self._prefix.append(ns)
        self._scale_factor.append(self._scale_factor[-1]*scale_factor)
    
    def pop_namespace(self):
        self._prefix.pop()
        self._scale_factor.pop()
    
    def pop_and_push_namespace(self,ns,scale_factor=1):
        self.pop_namespace()
        self.push_namespace(ns,scale_factor)
        
    def get_anonymous_name(self):
        self.random_id += 1
        return '_anonymouse_'+str(self.random_id)
    
    def full_name(self,name):
        prefix = '.'.join(self._prefix)+'.'
        if name==None:
            name = self.get_anonymous_name()
        return prefix + name

    def new_bi_flow(self,flow,name=None,scale_factor=1):
        '''
        Args:
            flow: 1,4,H,W
        '''
        assert flow.shape[1]==4
        return self.new_cat(
            [[self.new_flow(flow[:,0:2],scale_factor=scale_factor)],
             [self.new_flow(flow[:,2:4],scale_factor=scale_factor)]],
            name)

    def new_rgb(self,im,name=None,scale_factor=1):
        name = self.full_name(name)
        scale_factor = self._scale_factor[-1]*scale_factor
        im = plain_up(im, scale_factor)
        im = im[0].cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
        im = im*255
        self.imgs[name] = im
        return name

    def new_tensor(self,x,name=None):
        name = self.full_name(name)
        self.tensors[name] = x
        return name
    
    def new_flow(self,flow,name=None,scale_factor=1):
        '''
        Args:
            flow: 1,2,H,W
        '''
        assert flow.shape[1]==2
        name = self.full_name(name)
        scale_factor = self._scale_factor[-1]*scale_factor
        flow = plain_up(flow,scale_factor)
        flow = flow[0].cpu().numpy().transpose(1,2,0)
        flow = flow_to_image(flow)
        self.imgs[name] = flow
        return name
        
    def new_gray(self,gray,name=None,scale_factor=1,min_v=0,max_v=1):
        '''
        Args:
            gray: 1,1,H,W
        '''
        assert gray.shape[1]==1
        name = self.full_name(name)
        scale_factor = self._scale_factor[-1]*scale_factor
        gray = plain_up(gray,scale_factor)
        gray = (gray-min_v)/(max_v-min_v)*255
        gray = gray[0].cpu().numpy().transpose(1,2,0)
        gray = np.repeat(gray,3,axis=2)
        self.imgs[name] = gray
        return name
    
    def new_cat(self,img_matrix,name=None):
        name = self.full_name(name)
        row_list = []
        for row in img_matrix:
            row_list.append(np.concatenate([self.imgs[i] for i in row],axis=1))
            for i in row:
                self.imgs.pop(i)
        self.imgs[name] = np.concatenate(row_list,axis=0)
        return name
    
    def save_imgs(self, save_dir, rgb_order='rgb'):
        for name, img in self.imgs.items():
            if rgb_order=='rgb':
                pass
            elif rgb_order=='bgr':
                img = img[:,:,::-1]
            else:
                raise NotImplementedError
            Image.fromarray(np.uint8(img)).save(os.path.join(save_dir,name+'.jpg'))

    def save_tensors(self, save_dir):
        for name, x in self.tensors.items():
            torch.save(x, os.path.join(save_dir,name+'.pth'))
            
