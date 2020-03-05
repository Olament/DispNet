import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _mask_input(input, mask=None):
    if mask is not None:
        input = input * mask
        count = torch.sum(mask).data[0]
    else:
        count = np.prod(input.size(), dytpe=np.float32).item()
    return input, count


class DiffLoss(nn.Module):
    def forward(self, output, target):
        mask = (target>0).double()
        output = mask * output # apply mask to get rid of undefined depth in ground truth depth
        loss = torch.abs(output-target)
        return torch.sum(loss) / torch.sum(mask) 


class BerHuLoss(nn.Module):
    def forward(self, input, target, mask=None):
        x = input - target
        abs_x = torch.abs(x)
        c = torch.max(abs_x).data[0] / 5
        leq = (abs_x <= c).float()
        ls_losses = leq * abs_x + (1 - leq) * l2_losses
        losses, count = _mask_input(losses, mask)
        
        return torch.sum(losses) / count
