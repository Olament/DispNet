import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(ConvBlock, self).__init__()
        padding = get_padding(kernel_size) # calculate the padding need for 'SAME' padding
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        return out

class DeConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(DeConvBlock, self).__init__()
        padding = get_padding(kernel_size)
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.deconv(x)
        out = self.relu(out)
        out = self.bn(out)
        return out

class predict_weight(nn.Module):
    '''
    return plane weights - b x 9 x h x w
    '''
    def __init__(self, in_channel):
        super(predict_weight, self).__init__()
        out_channel = in_channel // 2
        self.reduc = torch.nn.Sequential()

        while out_channel >= 16:
            self.reduc.add_module('inter_{}'.format(out_channel),\
                    torch.nn.Sequential(nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,\
                    bias=False, kernel_size=1, stride=2, padding=0), nn.ELU()))
            in_channel = out_channel
            out_channel = out_channel // 2

        self.reduc.add_module('plane_weights', torch.nn.Conv2d(in_channel, out_channels=9, bias=False,\
                kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.reduc(x)
        out = F.normalize(out, 2, 1)
        return out

class predict_plane(nn.Module):
    '''
    calculate the plane params given a feature map - b x c x h x w
    return plane params - b x 4 x h x w
    '''
    def __init__(self, in_channel, is_final, max_depth):
        super(predict_plane, self).__init__()
        out_channel = in_channel // 2
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()
        
        while out_channel >= 8:
            self.reduc.add_module('inter_{}'.format(out_channel),\
                    torch.nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel,\
                    bias=False, kernel_size=1, stride=1, padding=0), nn.ELU()))
            in_channel = out_channel
            out_channel = out_channel // 2

        if is_final:
            self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(in_channel, out_channels=1, bias=False,\
                    kernel_size=1, stride=1, padding=0),\
                    nn.Sigmoid()))
        else:
            self.reduc.add_module('plane_params', torch.nn.Conv2d(in_channel, out_channels=3, bias=False,\
                    kernel_size=1, stride=1, padding=0))

    def forward(self, feature):
        out = self.reduc(feature)
        if not self.is_final:
            theta = self.sigmoid(out[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(out[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(out[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            out = torch.cat([n1, n2, n3, n4], dim=1) # b x 4 x h x w
            normal = out[:,:3] # b x 3 x h x w
            normal = F.normalize(normal,2,1)
            out[:,:3]=normal
        return out

class local_planar_guidance(nn.Module):
    '''
    given upratio (integer) and plane_eq - b x 4 x h/k x w/k
    return estimated depth - b x 9 x h x w
    '''
    def __init__(self, upratio):
        super(local_planar_guidance, self).__init__()
        self.u = torch.arange(int(upratio)).reshape([1, 1, upratio]).float()
        self.v = torch.arange(int(upratio)).reshape([1, upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3))
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio))
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        estimated_depth = torch.ones((plane_eq.size(0), 9, plane_eq_expanded.size(2), plane_eq_expanded.size(3)), 
            dtype=torch.float64, requires_grad=True)

        '''
        For patch at (i, j), 
            estimated_depth[b, 0, h, w] is calculated by plane at (i-1, j-1)
            estimated_depth[b, 1, h, w] is calculated by plane at (i, j-1)
            ...
        '''
        count = 0
        scale = [-1 * int(self.upratio), 0, 1 * int(self.upratio)]
        for h_shift in scale:
            for w_shift in scale:
                new_u = u - h_shift
                new_v = v - w_shift
                new_n1 = shift_frame(n1, w_shift, h_shift)
                new_n2 = shift_frame(n2, w_shift, h_shift)
                new_n3 = shift_frame(n3, w_shift, h_shift)
                new_n4 = shift_frame(n4, w_shift, h_shift)
                estimated_depth[:, count, :, :] = new_n4 / (new_n1 * new_u + new_n2 * new_v + new_n3)
                count += 1

        return estimated_depth

def shift_frame(x, w_dir, h_dir):
    '''
    shift the "frame" of an N-dimentionsal image tensor and apply replicate 
    padding to fill non-exist entry
    x:
        N-dimentionsal tensor
    w_dir:
        direction of width shift, w_dir in [-1, 0, 1]
        -1 means left shift, 1 means right shift, and 0 means no shift
    h_dir:
        same as w_dir except it represent height shift
    '''

    # apply padding to x
    pad = []
    for shift in [w_dir, h_dir]:
        if shift < 0:
            pad += [-shift, 0]
        elif shift > 0:
            pad += [0, shift]
        else:
            pad += [0, 0]
    x = x.unsqueeze(dim=0) # extra 'batch' dimension is required for padding
    x = F.pad(x, tuple(pad), 'replicate')

    # crop part we do not need
    if w_dir < 0:
        x = x[:, :, :, :w_dir]
    elif w_dir > 0:
        x = x[:, :, :, w_dir:]
    
    if h_dir < 0:
        x = x[:, :, :h_dir, :]
    elif h_dir > 0:
        x = x[:, :, h_dir:, :]

    x = x.squeeze(dim=0)

    return x

class DispNet(nn.Module):
    def __init__(self, h, w, max_depth=100.0):
        super(DispNet, self).__init__()

        self.max_depth = max_depth

        self.conv1 = ConvBlock(3, 32, kernel_size=7, stride=2)
        self.conv1b = ConvBlock(32, 32, kernel_size=7, stride=1)

        self.conv2 = ConvBlock(32, 64, kernel_size=5, stride=2)
        self.conv2b = ConvBlock(64, 64, kernel_size=5, stride=1)

        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2)
        self.conv3b = ConvBlock(128, 128, kernel_size=3, stride=1)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=2)
        self.conv4b = ConvBlock(256, 256, kernel_size=3, stride=1)
        self.conv5 = ConvBlock(256, 512, kernel_size=3, stride=2)
        self.conv5b = ConvBlock(512, 512, kernel_size=3, stride=1)
        self.conv6 = ConvBlock(512, 512, kernel_size=3, stride=2)
        self.conv6b = ConvBlock(512, 512, kernel_size=3, stride=1)
        self.conv7 = ConvBlock(512, 512, kernel_size=3, stride=2)
        self.conv7b = ConvBlock(512, 512, kernel_size=3, stride=1)

        self.up7 = DeConvBlock(512, 512, kernel_size=3, stride=2)
        self.icnv7 = ConvBlock(1024, 512, kernel_size=3, stride=1)

        self.up6 = DeConvBlock(512, 512, kernel_size=3, stride=2)
        self.icnv6 = ConvBlock(1024, 512, kernel_size=3, stride=1)

        self.up5 = DeConvBlock(512, 256, kernel_size=3, stride=2)
        self.icnv5 = ConvBlock(512, 256, kernel_size=3, stride=1)

        self.up4 = DeConvBlock(256, 128, kernel_size=3, stride=2)
        self.icnv4 = ConvBlock(256, 128, kernel_size=3, stride=1)
        self.pp4 = predict_plane(128, False, self.max_depth)
        self.weight4 = predict_weight(128)
        self.lpg4 = local_planar_guidance(8)

        self.up3 = DeConvBlock(128, 64, kernel_size=3, stride=2)
        self.icnv3 = ConvBlock(129, 64, kernel_size=3, stride=1)
        self.pp3 = predict_plane(64, False, self.max_depth)
        self.weight3 = predict_weight(64)
        self.lpg3 = local_planar_guidance(4)

        self.up2 = DeConvBlock(64, 32, kernel_size=3, stride=2)
        self.icnv2 = ConvBlock(65, 32, kernel_size=3, stride=1)
        self.pp2 = predict_plane(32, False, self.max_depth)
        self.weight2 = predict_weight(32)
        self.lpg2 = local_planar_guidance(2)

        self.up1 = DeConvBlock(32, 16, kernel_size=3, stride=2)
        #self.icnv1 = ConvBlock(17, 16, kernel_size=3, stride=1)
        self.pp1 = predict_plane(16, True, self.max_depth)
        self.icnv1 = ConvBlock(20, 16, kernel_size=3, stride=1)
        self.get_depth  = torch.nn.Sequential(nn.Conv2d(16, 1, 3, 1, 1, bias=False),\
                nn.Sigmoid())

        self.apply(init_weights)

    def forward(self, input):
        # down
        out = self.conv1(input)
        res1 = self.conv1b(out)
        out = self.conv2(res1)
        res2 = self.conv2b(out)

        #print("cnv2b: {}".format(res2.size()))

        out = self.conv3(res2)
        res3 = self.conv3b(out)
        out = self.conv4(res3)
        res4 = self.conv4b(out)
        out = self.conv5(res4)
        res5 = self.conv5b(out)
        out = self.conv6(res5)
        res6 = self.conv6b(out)
        out = self.conv7(res6)
        res7 = self.conv7b(out)
        out = res7

        #print("conv7b: {}".format(out.size()))

        # out
        out = self.up7(out)
        out = _resize_like(out, res6)
        out = torch.cat((out, res6), dim=1) # concat channel
        out = self.icnv7(out)

        #print("icnv7: {}".format(out.size()))

        out = self.up6(out)
        out = _resize_like(out, res5)
        out = torch.cat((out, res5), dim=1)
        out = self.icnv6(out)

        #print("icnv6: {}".format(out.size()))

        out = self.up5(out)
        out = _resize_like(out, res4)
        out = torch.cat((out, res4), dim=1)
        out = self.icnv5(out)

        #print("icnv5: {}".format(out.size()))

        out = self.up4(out)
        out = _resize_like(out, res3) # extra resize to solve padding issue
        out = torch.cat((out, res3), dim=1)
        out = self.icnv4(out)

        plane4 = self.pp4(out)
        depth4 = self.lpg4(plane4)
        depth4 = depth4 / self.max_depth
        depth4 = _resize_like(depth4, input) # b x 9 x h x w
        weight4 = self.weight4(out) # b x 9 x h x w
        weight4 = _resize_like(weight4, input)
        depth4 = (depth4*weight4).sum(1,keepdim=True) # b x 1 x h x w

        out = self.up3(out)
        out = _resize_like(out, res2) # extra resize to solve padding issue
        depth4_ds = _resize_like(depth4, res2)
        out = torch.cat((out, res2, depth4_ds), dim=1)
        out = self.icnv3(out)

        plane3 = self.pp3(out)
        depth3 = self.lpg3(plane3)
        depth3 = depth3 / self.max_depth
        depth3 = _resize_like(depth3, input)
        weight3 = self.weight3(out) # b x 9 x h x w
        weight3 = _resize_like(weight3, input)
        depth3 = (depth3*weight3).sum(1,keepdim=True) # b x 1 x h x w

        out = self.up2(out)
        out = _resize_like(out, res1) # extra resize to solve padding issue
        depth3_ds = _resize_like(depth3, res1)
        out = torch.cat((out, res1, depth3_ds), dim=1)
        out = self.icnv2(out)

        plane2 = self.pp2(out)
        depth2 = self.lpg2(plane2)
        depth2 = depth2 / self.max_depth
        depth2 = _resize_like(depth2, input)
        weight2 = self.weight2(out) # b x 9 x h x w
        weight2 = _resize_like(weight2, input)
        depth2 = (depth2*weight2).sum(1,keepdim=True) # b x 1 x h x w

        out = self.up1(out)
        out = _resize_like(out, input) # extra resize to solve padding issue
        plane1 = self.pp1(out)
        out = torch.cat((out, plane1, depth2, depth3, depth4), dim=1)
        out = self.icnv1(out)
        depth1 = self.max_depth * self.get_depth(out)

        return depth1, depth2, depth3, depth4 

def get_padding(kernel_size):
    return (kernel_size - 1) // 2

def _resize_like(inputs, ref):
    _,_,i_h,i_w = inputs.shape
    _,_,r_h,r_w = ref.shape
    if i_h == r_h and i_w == r_w:
        return inputs
    else:
        resized = F.interpolate(inputs, (r_h, r_w), mode='nearest')
        return resized

def init_weights(m):
    # Initialize parameters
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
