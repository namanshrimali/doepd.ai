from torch.utils.data import Dataset
import os
import cv2
from utils.transforms import Resize, NormalizeImage, PrepareForNet

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
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = cv2.imread(self.images[index]) #BGR
        depth = cv2.imread(self.depth_images[index], cv2.IMREAD_UNCHANGED) #BGR
        if depth is not None:
            return self.transforms({"image": image})["image"], self.transforms({"image": depth})["image"]
        else:
            return self.__getitem__(index + 1)