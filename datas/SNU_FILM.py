import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import os
from hashlib import sha256
from .utils import imread

class SNU_FILM(data.Dataset):
    def __init__(self, args):
        self.triplets = []

        if args.split in ['easy','medium','hard','extreme']:
            split_file = f'{args.dataset_root}/test-{args.split}.txt'
        else:
            split_file = f'{args.dataset_root}/{args.split}'

        self.dataset_root = args.dataset_root
        self.rgb_order = args.rgb_order
        
        if hasattr(args,'hash_name') and args.hash_name:
            self.hash_name = True
        else:
            self.hash_name = False

        with open(split_file,'r') as txt:
            for line in txt:
                line = line.strip()
                if line=='':
                    continue
                tri = line.split(' ')
                tri = [f"{t.replace('data/SNU-FILM/','')}" for t in tri]
                self.triplets.append(tri)

    def transform(self, frame1, frame3):
        return map(TF.to_tensor, (frame1, frame3))

    def __getitem__(self, index):
        tri = self.triplets[index]

        frame1 = imread(os.path.join(self.dataset_root,tri[0]),self.rgb_order)
        frame2 = imread(os.path.join(self.dataset_root,tri[1]),self.rgb_order)
        frame3 = imread(os.path.join(self.dataset_root,tri[2]),self.rgb_order)

        frame1, frame3 = self.transform(frame1, frame3)

        Input = torch.cat((frame1, frame3), dim=0)

        name = ','.join([tri[0],tri[1],tri[2]])
        if self.hash_name:
            name = sha256(name.encode('utf-8')).hexdigest()

        return Input, frame2, name

    def __len__(self):
        return len(self.triplets)