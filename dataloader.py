import torch
import os
from PIL import Image, ImageOps
import numpy as np
import cv2
import torchvision.transforms as transforms
import random

class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        self.img_lst = []
        self.depth_lst = []
        self.mode = mode

        if mode == 'train':
            self.img_path='data/image_sequence.txt' 
            self.depth_path='data/depth_sequence.txt'
        else:
            self.img_path='data/image_sequence_test.txt'
            self.depth_path='data/depth_sequence_test.txt'

        with open(self.img_path) as file:
            for line in file.readlines():
                self.img_lst.append(line.strip())
        with open(self.depth_path) as file:
            for line in file.readlines():
                self.depth_lst.append(line.strip())
    
    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, index):
        if self.mode == 'train':
            image_trans = transforms.Compose([
                            transforms.Resize((375,1424)),
                            transforms.ColorJitter(brightness=(0.8, 1.2), 
                                                   contrast=(0.8, 1.2), 
                                                   saturation=(0.8, 1.2), 
                                                   hue=(-0.1, 0.1)),
            ])
        else:
            image_trans = transforms.Compose([
                            transforms.Resize((375, 1424)),
            ])

        totensor_trans = transforms.ToTensor()

        image = Image.open(self.img_lst[index])
        image = image_trans(image)

        depth = cv2.imread(self.depth_lst[index], -1)
        depth = cv2.resize(depth, dsize=(1424, 375), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(float)
        depth /= 256.0

        if self.mode == 'train':
            depth = Image.fromarray(depth) # convert numpy to PIL image
            # random horizontal flip
            if random.random() > 0.5:
                depth = ImageOps.mirror(depth)
                image = ImageOps.mirror(image)

            image = totensor_trans(depth)
            depth = totensor_trans(depth)

            image, depth = random_crop(image, depth, 1324, 275)
        else:
            image = totensor_trans(depth)
            depth = torch.from_numpy(depth)

        return image, depth


def random_crop(image, depth, height, width):
    assert image.shape[1] >= height
    assert image.shape[2] >= width
    assert image.shape[1] == depth.shape[1]
    assert image.shape[2] == depth.shape[2]

    x = random.randint(0, image.shape[2] - width)
    y = random.randint(0, image.shape[1] - height)
    image = image[:, y:y+height, x:x + width]
    depth = depth[:, y:y+height, x:x + width]
    return image, depth
