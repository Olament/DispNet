import torch
import torchvision.transforms as transforms
import cv2
import io
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

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


#def convert_to_colormap(tensor):
#    plt.imsave('v.png', tensor.cpu().detach().numpy(), cmap='plasma')
#    tensor = torch.unsqueeze(tensor, 0)
#    tensor = torch.clamp(tensor, 0.0001, 100.0)
#    tensor = (tensor / 100.0) * 256.0
#    img = tensor.permute(1, 2, 0).cpu().detach().numpy()
#    #img *= 256
#    img = img.astype(np.uint8)
#    img = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
#    return img


def convert_to_colormap(tensor):
    #buf = io.BytesIO()
    #plt.imshow(tensor.cpu().detach().numpy(), cmap='plasma')
    #plt.savefig(buf, format='png')
    #buf.seek(0)
    plt.imsave('depth.png', tensor.cpu().detach().numpy(), cmap='plasma')
    img = Image.open('depth.png')
    img = img.convert('RGB')
    #img = img.resize((1424, 375))
    trans = transforms.ToTensor()
    img = trans(img)

    return img 
