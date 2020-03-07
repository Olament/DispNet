import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _mask_input(input, mask=None):
    if mask is not None:
        input = input * mask
        count = torch.sum(mask).item()
    else:
        count = np.prod(input.size(), dytpe=np.float32).item()
    return input, count


class DiffLoss(nn.Module):
    def forward(self, output, target):
        mask = (target>0).double()
        output = mask * output # apply mask to get rid of undefined depth in ground truth depth
        loss = torch.abs(output-target)
        return torch.sum(loss) / torch.sum(mask) 


def BerHuLoss(input, target, mask=None):
    x = input - target
    abs_x = torch.abs(x)
    c = torch.max(abs_x).item() / 5
    leq = (abs_x <= c).float()
    l2_losses = (x**2 + c**2) / (2 * c)
    losses = leq * abs_x + (1 - leq) * l2_losses
    losses, count = _mask_input(losses, mask)
        
    return torch.sum(losses) / count


def SmoothLoss(depth, image):
    # gradient for predicted truth
    depth_x = gradient_x(depth)
    depth_y = gradient_y(depth) 
    # gradient for image 
    img_x = gradient_x(image)
    img_y = gradient_y(image)
    # weight
    weight_x = ((img_x.abs().mean(1,keepdim=True))*(-1)).exp()
    weight_y = ((img_y.abs().mean(1,keepdim=True))*(-1)).exp()
    smooth_x = weight_x * depth_x
    smooth_y = weight_y * depth_y
    loss = smooth_x.abs().mean() + smooth_y.abs().mean()
    return loss
   

def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def gradient_y(img):
    return img[:, :, ;, :-1] - img[:, :, :, 1:]

