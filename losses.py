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


def AbsLoss(input, target):
    input = input.reshape(-1)
    target = target.reshape(-1)
    mask = (target > 0) & (target < 100) 

    input = torch.masked_select(input, mask)
    target = torch.masked_select(target, mask)

    return (target - input).abs().mean()
    

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
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def SILogLoss(input, target, ratio=10, ratio2=0.85, type='disparity'):
    input = input.reshape(-1)
    target = target.reshape(-1)

    if type == 'disparity':
        mask = (target < 1) & (target > 0.01234) # 1/81
    else:
        mask = (target > 0) & (target < 80)

    masked_input = torch.masked_select(input, mask)
    masked_output = torch.masked_select(target, mask)

    log_diff = torch.log(masked_input * ratio) - torch.log(masked_output * ratio)

    silog1 = torch.mean(log_diff ** 2)
    silog2 = ratio2 * (torch.mean(log_diff) ** 2)
    silog_loss = torch.sqrt(silog1 - silog2) * ratio

    return silog_loss


def MaskLoss(mask):
    return (mask.sum(dim=1)-1).abs().mean()

def Eval(pred, target):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    #mask = (target > 0) & (target > 0.01234)
    mask = (target > 0) & (target < 80) & (pred > 0) & (pred < 80)
    pred = torch.masked_select(pred, mask)
    target = torch.masked_select(target, mask)

    thresh = torch.max(pred / target, target / pred)
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (target - pred).pow(2)
    rmse = rmse.mean().pow(0.5)
    
    rmse_log = (target.log() - pred.log()).pow(2)
    rmse_log = rmse_log.mean().pow(0.5)

    abs_rel = ((target - pred) / target).abs().mean()

    sq_rel = (((target - pred).pow(2)) / target).mean()
    
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
