import torch
import os
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, img_path='data/image_sequence.txt', depth_path='data/depth_sequence.txt'):
        self.img_lst = []
        self.depth_lst = []
        with open(img_path) as file:
            for line in file.readlines():
                self.img_lst.append(line.strip())
        with open(depth_path) as file:
            for line in file.readlines():
                self.depth_lst.append(line.strip())
    
    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, index):
        image_trans = transforms.Compose([
                        transforms.Resize((375, 1424)),
                        transforms.ToTensor()
        ])
        image = Image.open(self.img_lst[index])
        image = image_trans(image)

        depth = cv2.imread(self.depth_lst[index], -1)
        depth = cv2.resize(depth, dsize=(1424, 375), interpolation=cv2.INTER_LINEAR)
        depth = depth.astype(float)
        depth /= 256.0
        mask = (depth == 0.0).astype(float) # mask zero
        depth += mask
        depth = np.reciprocal(depth)
        depth -= mask
        depth = torch.from_numpy(depth) 

        return image, depth
