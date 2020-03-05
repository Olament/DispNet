import torch
import cv2
import numpy as np

def visualize_img_depth(img, depth):
    depth = torch.unsqueeze(depth, 0)
    depth = depth/depth.max()
    _, h, w = img.shape
    
    new_img = img.new(3, h*2, w)
    depth = depth.expand(3, -1, -1)
    
    new_img[:, h:] = depth
    new_img[:, :h] = img
    new_img = new_img.cpu().detach().permute(1, 2, 0).numpy()

    cv2.imshow('img', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_img(img, title):
    cv2.imshow(title, img.cpu().detach().permute(1, 2, 0).numpy())
    cv2.waitKey(0)

def visualize_dep(dep, title):
    cv2.imshow(title, dep.cpu().detach().numpy())
    cv2.waitKey(0)
