import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import cv2
import random
import numpy as np
import os
from .utils import imread


class Vimeo_train(data.Dataset):
    def __init__(self, args):
        self.crop_size = args.crop_size
        self.sequence_list = []
        with open('%s/tri_trainlist.txt' % args.dataset_root, 'r') as txt:
            for line in txt:
                self.sequence_list.append('%s/sequences/%s' % (args.dataset_root, line.strip()))
        self.rgb_order = args.rgb_order
        if hasattr(args, 'aug_script'):
            self.aug_script = args.aug_script
        else:
            self.aug_script = 'abme'

        print(f'Vimeo_train: aug_script={self.aug_script}')
        
        assert self.aug_script in ['abme','rife']

        if hasattr(args,'same_with_rife') and args.same_with_rife:
            cnt = int(len(self.sequence_list)*0.95)
            self.sequence_list = self.sequence_list[:cnt]
    
    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1
    
    def aug_abme(self, frame1,frame2,frame3):
        if not random.randint(0,1):
            frame1,frame3 = frame3,frame1
        # Rotation augmentation
        if self.crop_size[0] == self.crop_size[1]:
            if random.randint(0, 1):
                frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
                frame2 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
                frame3 = cv2.rotate(frame3, cv2.ROTATE_90_CLOCKWISE)
            elif random.randint(0, 1):
                frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
                frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
                frame3 = cv2.rotate(frame3, cv2.ROTATE_180)
            elif random.randint(0, 1):
                frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame3 = cv2.rotate(frame3, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Flip augmentation
        if random.randint(0, 1):
            flip_code = random.randint(-1,1) # 0 : Top-bottom | 1: Right-left | -1: both
            frame1 = cv2.flip(frame1, flip_code)
            frame2 = cv2.flip(frame2, flip_code)
            frame3 = cv2.flip(frame3, flip_code)
        
        return frame1,frame2,frame3
    
    def aug_rife(self,img0,gt,img1):
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, :, ::-1]
            img1 = img1[:, :, ::-1]
            gt = gt[:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            img0 = img0[::-1]
            img1 = img1[::-1]
            gt = gt[::-1]
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, ::-1]
            img1 = img1[:, ::-1]
            gt = gt[:, ::-1]
        if random.uniform(0, 1) < 0.5:
            tmp = img1
            img1 = img0
            img0 = tmp
        # random rotation
        p = random.uniform(0, 1)
        if p < 0.25:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        elif p < 0.5:
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
            gt = cv2.rotate(gt, cv2.ROTATE_180)
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
        elif p < 0.75:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img0,gt,img1


    def transform(self, frame1, frame2, frame3):

        frame1,frame2, frame3 = self.crop(frame1, frame2, frame3,self.crop_size[0],self.crop_size[1])

        if self.aug_script == 'abme':
            frame1,frame2,frame3 = self.aug_abme(frame1,frame2,frame3)
        elif self.aug_script == 'rife':
            frame1,frame2,frame3 = self.aug_rife(frame1, frame2, frame3)
        else:
            raise NotImplementedError

        return map(TF.to_tensor, (frame1.copy(), frame2.copy(), frame3.copy()))

    def __getitem__(self, index):

        First_fn  = os.path.join(self.sequence_list[index], 'im1.png')
        Second_fn = os.path.join(self.sequence_list[index], 'im2.png')
        Third_fn  = os.path.join(self.sequence_list[index], 'im3.png')

        frame1 = imread(First_fn, self.rgb_order)
        frame2 = imread(Second_fn, self.rgb_order)
        frame3 = imread(Third_fn, self.rgb_order)

        frame1, frame2, frame3 = self.transform(frame1, frame2, frame3)

        Input = torch.cat((frame1, frame3), dim=0)

        return Input, frame2

    def __len__(self):
        return len(self.sequence_list)

class Vimeo_validation(data.Dataset):
    def __init__(self, args):
        self.sequence_list = []
        tri_list = 'tri_testlist.txt'
        if hasattr(args,'tri_list'):
            tri_list = args.tri_list
        with open('%s/%s'%(args.dataset_root,tri_list),'r') as txt:
            for line in txt:
                line = line.strip()
                if line=='':
                    continue
                self.sequence_list.append('%s/sequences/%s'%(args.dataset_root, line))
        if hasattr(args,'rgb_order'):
            self.rgb_order = args.rgb_order
        else:
            self.rgb_order = 'rgb'
        assert self.rgb_order in ['rgb','bgr']

    def transform(self, frame1, frame3):
        return map(TF.to_tensor, (frame1, frame3))

    def __getitem__(self, index):
        first_fn = os.path.join(self.sequence_list[index],'im1.png')
        second_fn = os.path.join(self.sequence_list[index],'im2.png')
        third_fn = os.path.join(self.sequence_list[index],'im3.png')

        frame1 = imread(first_fn, self.rgb_order)
        frame2 = imread(second_fn, self.rgb_order)
        frame3 = imread(third_fn, self.rgb_order)

        frame1, frame3 = self.transform(frame1, frame3)

        Input = torch.cat((frame1, frame3), dim=0)

        return Input, frame2, '_'.join(self.sequence_list[index].split('/')[-2:])

    def __len__(self):
        return len(self.sequence_list)
