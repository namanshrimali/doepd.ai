from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import numpy as np
import torch

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

class LoadImageDepthAndLabels(Dataset):
    def __init__(self, train=False, transforms=None):
        if train:
            root ="data/assignment13/train.txt"
        else:
            root="data/assignment13/test.txt"
        with open(root, 'r') as f:
            self.images = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
                              if os.path.splitext(x)[-1].lower() in img_formats]
            
        self.depth_images = [x.replace('images', 'depth').replace(os.path.splitext(x)[-1], '.png')
                            for x in self.img_files]
        
    
    def __len__(self):
        return len(self.img_files)
    
    def __get_item__(self, index):
        image = cv2.imread(self.images[index]) #BGR
        depth = cv2.imread(self.depthimages[index]) #BGR
        return self.transforms({"image": image})["image"], self.transforms({"image": depth})["image"]