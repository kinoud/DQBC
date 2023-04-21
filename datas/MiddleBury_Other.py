import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import os
from .utils import imread

class MiddelBuryOther(data.Dataset):
    def __init__(self, args):
        self.triplets = []

        self.name_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        
        for x in self.name_list:
            f0 = f'other-data/{x}/frame10.png'
            f1 = f'other-gt-interp/{x}/frame10i11.png'
            f2 = f'other-data/{x}/frame11.png'
            self.triplets.append((f0,f1,f2))
        
        self.dataset_root = args.dataset_root

        self.rgb_order = args.rgb_order


    def transform(self, frame1, frame3):
        return map(TF.to_tensor, (frame1, frame3))

    def __getitem__(self, index):
        tri = self.triplets[index]

        frame1 = imread(os.path.join(self.dataset_root,tri[0]),self.rgb_order)
        frame2 = imread(os.path.join(self.dataset_root,tri[1]),self.rgb_order)
        frame3 = imread(os.path.join(self.dataset_root,tri[2]),self.rgb_order)

        frame1, frame3 = self.transform(frame1, frame3)

        Input = torch.cat((frame1, frame3), dim=0)

        return Input, frame2, self.name_list[index]

    def __len__(self):
        return len(self.triplets)