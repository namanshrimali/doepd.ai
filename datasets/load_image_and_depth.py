from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import numpy as np
import torch

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

class LoadImageDepthAndLabels(Dataset):
    def __init__(self, train=False, transforms=None):
        self.transforms = transforms
        if train:
            root ="data/assignment13/train.txt"
        else:
            root="data/assignment13/test.txt"
        with open(root, 'r') as f:
            self.images = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
                              if os.path.splitext(x)[-1].lower() in img_formats]
            
        self.depth_images = [x.replace('images', 'depth').replace(os.path.splitext(x)[-1], '.png')
                            for x in self.images]
        print(len(self.depth_images))
        print(len(self.images))
        
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = cv2.imread(self.images[index]) #BGR
        depth = cv2.imread(self.depth_images[index]) #BGR
        print(depth.shape)
        return self.transforms({"image": image})["image"], self.transforms({"image": depth})["image"]