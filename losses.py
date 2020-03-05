import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RelLoss(nn.Module):
    def forward(self, output, target):
        mask = (target>0).double()
        output = mask * output # apply mask to get rid of undefined depth in ground truth depth
        loss = torch.abs(output-target)
        return torch.sum(loss) / torch.sum(mask) 
