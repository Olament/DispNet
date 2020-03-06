import torch
import torch.nn as nn

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


class DispNet(nn.Module):
    def __init__(self, h, w, disp_scaling=10, min_disp=0.01):
        super(DispNet, self).__init__()

        self.DISP_SCALING = disp_scaling
        self.MIN_DISP = min_disp

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
        self.disp4 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
                                   nn.Sigmoid()) # need add scaling
        self.scale4 = nn.Upsample((h // 4, w // 4), mode="bilinear")

        self.up3 = DeConvBlock(128, 64, kernel_size=3, stride=2)
        self.icnv3 = ConvBlock(129, 64, kernel_size=3, stride=1)
        self.disp3 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                                   nn.Sigmoid())  # need add scaling
        self.scale3 = nn.Upsample((h // 2, w // 2), mode="bilinear")

        self.up2 = DeConvBlock(64, 32, kernel_size=3, stride=2)
        self.icnv2 = ConvBlock(65, 32, kernel_size=3, stride=1)
        self.disp2 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
                                   nn.Sigmoid())
        self.scale2 = nn.Upsample((h, w), mode="bilinear")

        self.up1 = DeConvBlock(32, 16, kernel_size=3, stride=2)
        self.icnv1 = ConvBlock(17, 16, kernel_size=3, stride=1)
        self.disp1 = nn.Sequential(nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
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
        out = self.conv7b(out)

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
        disp4 = self.disp4(out)
        disp4 = disp4 * self.DISP_SCALING + self.MIN_DISP

        disp4_up = self.scale4(disp4)

        #print("disp4_up: {}".format(disp4_up.size()))

        out = self.up3(out)
        out = _resize_like(out, res2) # extra resize to solve padding issue
        disp4_up = _resize_like(disp4_up, res2)
        out = torch.cat((out, res2, disp4_up), dim=1)
        out = self.icnv3(out)
        disp3 = self.disp3(out)
        disp3 = disp3 * self.DISP_SCALING + self.MIN_DISP
        disp3_up = self.scale3(disp3)

        #print("disp3_up: {}".format(disp3_up.size()))

        out = self.up2(out)
        out = _resize_like(out, res1) # extra resize to solve padding issue
        disp3_up = _resize_like(disp3_up, res1)
        out = torch.cat((out, res1, disp3_up), dim=1)
        out = self.icnv2(out)
        disp2 = self.disp2(out)
        disp2 = disp2 * self.DISP_SCALING + self.MIN_DISP
        disp2_up = self.scale2(disp2)

        #print("disp2_up: {}".format(disp2_up.size()))

        out = self.up1(out)
        out = _resize_like(out, disp2_up) # extra resize to solve padding issue
        out = torch.cat((out, disp2_up), dim=1)
        out = self.icnv1(out)
        out = self.disp1(out) # b x 2 x h x w
        disp = out[:,0:1] * self.DISP_SCALING + self.MIN_DISP
        mask = out[:,1:] # >=0.5 for valid point, <0.5 for sky or infinity

        #return [out, disp2, disp3, disp4]
        return disp, mask


def get_padding(kernel_size):
    return (kernel_size - 1) // 2

def _resize_like(inputs, ref):
    _,_,i_h,i_w = inputs.shape
    _,_,r_h,r_w = ref.shape
    if i_h == r_h and i_w == r_w:
        return inputs
    else:
        upsample = nn.Upsample((r_h, r_w), mode='nearest')
        return upsample(inputs)

def init_weights(m):
    # Initialize parameters
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
