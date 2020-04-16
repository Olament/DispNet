import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### hyper-parameters ###

class_number = 10

### Network ###

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        import torchvision.models as models
        self.base_model = models.resnet101(pretrained=True)
        self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
        self.feat_out_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        features = [x]
        skip_feature = [x]
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(features[-1])
            features.append(feature)
            if any(x in k for x in self.feat_names):
                skip_feature.append(feature)

        return skip_feature


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(ConvBlock, self).__init__()
        padding = get_padding(kernel_size) # calculate the padding need for 'SAME' padding
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
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
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                                         padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.deconv(x)
        out = self.relu(out)
        out = self.bn(out)

        return out


class reduction_1x1(nn.Sequential):
    '''
    Return b x 4 x h x (w * class_number)
    '''
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()

        while num_out_filters >= 4:
            if num_out_filters < (3 * class_number):
                if self.is_final:
                    self.reduc.add_module('final',
                                          torch.nn.Sequential(nn.Conv2d(num_in_filters,
                                                                        out_channels=1,
                                                                        bias=False, kernel_size=1, stride=1, padding=0),
                                                              nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters,
                                                                          out_channels=(3 * class_number),
                                                                          bias=False, kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(
                                          nn.Conv2d(in_channels=num_in_filters,
                                                    out_channels=num_out_filters,
                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, net):
        net = self.reduc.forward(net) # b x (3 * class_number) x h x w
        if not self.is_final:
            net = net.view(net.size(0), 3, class_number, net.size(2), net.size(3)) # b x 3 x class_number x h x w
            net = net.permute(0, 1, 3, 4, 2).contiguous() # b x 3 x h x w x class_number
            net = net.view(net.size(0), 3, net.size(2), net.size(3) * class_number) # b x 3 x h x (w * class_number)
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)
            # normalized
            normal = net[:, :3].clone()  # b x 3 x h x w
            normal = F.normalize(normal, 2, 1)
            net[:, :3] = normal

        return net


class local_planar_guidance(nn.Module):
    '''
    given upratio (integer) and plane_eq - b x 3 x h/k x (w/k * class_number)
    return estimated depth - b x class_number x h x w
    '''
    def __init__(self, upratio):
        super(local_planar_guidance, self).__init__()
        self.u = torch.arange(int(upratio)).reshape([1, 1, upratio]).float()
        self.v = torch.arange(int(upratio)).reshape([1, upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :].to(device) # b x h x (w * class_number)
        n2 = plane_eq_expanded[:, 1, :, :].to(device)
        n3 = plane_eq_expanded[:, 2, :, :].to(device)
        n4 = plane_eq_expanded[:, 3, :, :].to(device)

        height, width = plane_eq.size(2), plane_eq.size(3) // class_number

        u = self.u.repeat(plane_eq.size(0), height * int(self.upratio), width).to(device)
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), height, width * int(self.upratio)).to(device)
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        u = u.repeat(1, 1, class_number)
        v = v.repeat(1, 1, class_number)

        result = n4 / (n1 * u + n2 * v + n3) # b x h x (w * class_number)
        result = result.view(result.size(0), result.size(1),
                             result.size(2) // class_number, class_number) # b x h x w x class_number
        result = result.permute(0, 3, 1, 2) # b x class_number x h x w

        return result


class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True,
                                                                   track_running_stats=True, eps=1.1e-5))

        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels,
                                                                              out_channels=out_channels * 2,
                                                                              bias=False,
                                                                              kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels * 2, momentum=0.01,
                                                                                   affine=True,
                                                                                   track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2,
                                                                              out_channels=out_channels, bias=False,
                                                                              kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation),
                                                                              dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)


class mask_predict(nn.Module):
    def __init__(self, skip_feature_channel, num_feature=512):
        super(mask_predict, self).__init__()

        self.upconv5 = DeConvBlock(skip_feature_channel[4], num_feature, 3, 2)
        self.conv5 = ConvBlock(num_feature + skip_feature_channel[3], num_feature, 3, 1)

        self.upconv4 = DeConvBlock(num_feature, num_feature // 2, 3, 2)
        self.conv4 = ConvBlock(num_feature // 2 + skip_feature_channel[2], num_feature // 2, 3, 1)

        self.upconv3 = DeConvBlock(num_feature // 2, num_feature // 4, 3, 2)
        self.conv3 = ConvBlock(num_feature // 4 + skip_feature_channel[1], num_feature // 4, 3, 1)

        self.upconv2 = DeConvBlock(num_feature // 4, num_feature // 8, 3, 2)
        self.conv2 = ConvBlock(num_feature // 8 + skip_feature_channel[0], num_feature // 8, 3, 1)

        self.upconv1 = DeConvBlock(num_feature // 8, class_number, 3, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(class_number, class_number, 3, 1, 1),
                                   nn.Sigmoid())

    def forward(self, features):
        skip0, skip1, skip2, skip3 = features[1], features[2], features[3], features[4]
        dense_feature = torch.nn.ReLU()(features[5])  # h // 32

        upconv5 = self.upconv5(dense_feature)
        upconv5 = _resize_like(upconv5, skip3)
        concat5 = torch.cat([upconv5, skip3], dim=1)
        iconv5 = self.conv5(concat5)

        upconv4 = self.upconv4(iconv5)
        upconv4 = _resize_like(upconv4, skip2)
        concat4 = torch.cat([upconv4, skip2], dim=1)
        iconv4 = self.conv4(concat4)

        upconv3 = self.upconv3(iconv4)
        upconv3 = _resize_like(upconv3, skip1)
        concat3 = torch.cat([upconv3, skip1], dim=1)
        iconv3 = self.conv3(concat3)

        upconv2 = self.upconv2(iconv3)
        upconv2 = _resize_like(upconv2, skip0)
        concat2 = torch.cat([upconv2, skip0], dim=1)
        iconv2 = self.conv2(concat2)

        upconv1 = self.upconv1(iconv2)
        mask = self.conv1(upconv1)
        mask = _resize_like(mask, features[0]) # resize to original size

        return mask


class MaskBTS(nn.Module):
    def __init__(self, skip_feature_channel, num_features=512):
        super(MaskBTS, self).__init__()

        # params
        self.max_depth = 80.0

        self.mask_predict = mask_predict(skip_feature_channel)

        self.upconv5 = DeConvBlock(skip_feature_channel[4], num_features, 3, 2) # h -> h * 2
        self.conv5 = ConvBlock(num_features + skip_feature_channel[3], num_features, 3, 1) # h -> h

        self.upconv4 = DeConvBlock(num_features, num_features // 2, 3, 2) # h -> h * 2
        self.conv4 = ConvBlock(num_features // 2 + skip_feature_channel[2], num_features // 2, 3, 1) # h -> h

        self.daspp_3 = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6 = atrous_conv(num_features // 2 + num_features // 4 + skip_feature_channel[2], num_features // 4, 6)
        self.daspp_12 = atrous_conv(num_features + skip_feature_channel[2], num_features // 4, 12)
        self.daspp_18 = atrous_conv(num_features + num_features // 4 + skip_feature_channel[2], num_features // 4, 18)
        self.daspp_24 = atrous_conv(num_features + num_features // 2 + skip_feature_channel[2], num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(
            nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())

        self.reduc8x8 = reduction_1x1(num_features // 4, num_features // 4, self.max_depth)
        self.lpg8x8 = local_planar_guidance(8)

        self.upconv3 = DeConvBlock(num_features // 4, num_features // 4, 3, 2)
        self.conv3 = ConvBlock(num_features // 4 + skip_feature_channel[1] + 1, num_features // 4, 3, 1)

        self.reduc4x4 = reduction_1x1(num_features // 4, num_features // 8, self.max_depth)
        self.lpg4x4 = local_planar_guidance(4)

        self.upconv2 = DeConvBlock(num_features // 4, num_features // 8, 3, 2)
        self.conv2 = ConvBlock(num_features // 8 + skip_feature_channel[0] + 1, num_features // 8, 3, 1)

        self.reduc2x2 = reduction_1x1(num_features // 8, num_features // 16, self.max_depth)
        self.lpg2x2 = local_planar_guidance(2)

        self.upconv1 = DeConvBlock(num_features // 8, num_features // 16, 3, 2)
        self.reduc1x1 = reduction_1x1(num_features // 16, num_features // 32, self.max_depth, is_final=True)
        self.conv1 = ConvBlock(num_features // 16 + 4, num_features // 16, 3, 1)
        self.get_depth = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid())

    def forward(self, features):
        skip0, skip1, skip2, skip3 = features[1], features[2], features[3], features[4]
        dense_feature = torch.nn.ReLU()(features[5]) # h // 32

        mask = self.mask_predict(features)

        upconv5 = self.upconv5(dense_feature) # h // 16
        upconv5 = _resize_like(upconv5, skip3)
        concat5 = torch.cat([upconv5, skip3], dim=1)
        iconv5 = self.conv5(concat5)

        upconv4 = self.upconv4(iconv5) # h // 8
        upconv4 = _resize_like(upconv4, skip2)
        concat4 = torch.cat([upconv4, skip2], dim=1)
        iconv4 = self.conv4(concat4)

        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)

        reduc8x8 = self.reduc8x8(daspp_feat)
        depth8x8 = self.lpg8x8(reduc8x8)
        depth8x8 = depth8x8 / self.max_depth
        depth8x8_mask = _resize_like(mask, depth8x8)
        depth8x8 = torch.sum(depth8x8 * depth8x8_mask, dim=1, keepdim=True)
        depth8x8 = _resize_like(depth8x8, features[0])
        depth8x8_ds = F.interpolate(depth8x8, scale_factor=0.25, mode='nearest')

        upconv3 = self.upconv3(daspp_feat) # h // 4
        upconv3 = _resize_like(upconv3, skip1)
        depth8x8_ds = _resize_like(depth8x8_ds, skip1)
        concat3 = torch.cat([upconv3, skip1, depth8x8_ds], dim=1)
        iconv3 = self.conv3(concat3)

        reduc4x4 = self.reduc4x4(iconv3)
        depth4x4 = self.lpg4x4(reduc4x4)
        depth4x4 = depth4x4 / self.max_depth
        depth4x4 = _resize_like(depth4x4, features[0])
        depth4x4 = torch.sum(depth4x4 * mask, dim=1, keepdim=True)
        depth4x4_ds = F.interpolate(depth4x4, scale_factor=0.5, mode='nearest')

        upconv2 = self.upconv2(iconv3) # h // 2
        upconv2 = _resize_like(upconv2, skip0)
        depth4x4_ds = _resize_like(depth4x4_ds, skip0)
        concat2 = torch.cat([upconv2, skip0, depth4x4_ds], dim=1)
        iconv2 = self.conv2(concat2)

        reduc2x2 = self.reduc2x2(iconv2) # h
        depth2x2 = self.lpg2x2(reduc2x2)
        depth2x2 = depth2x2 / self.max_depth
        depth2x2 = _resize_like(depth2x2, features[0])
        depth2x2 = torch.sum(depth2x2 * mask, dim=1, keepdim=True)

        upconv1 = self.upconv1(iconv2)
        reduc1x1 = self.reduc1x1(upconv1)
        upconv1 = _resize_like(upconv1, features[0])
        reduc1x1 = _resize_like(reduc1x1, features[0])
        concat1 = torch.cat([upconv1, reduc1x1, depth2x2, depth4x4, depth8x8], dim=1)
        iconv1 = self.conv1(concat1)

        final_depth = self.max_depth * self.get_depth(iconv1)

        return final_depth

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = encoder()
        self.decoder = MaskBTS(self.encoder.feat_out_channels)

    def forward(self, x):
        skip_feature = self.encoder(x)
        return self.decoder(skip_feature)


### Utilities ###

def get_padding(kernel_size):
    return (kernel_size - 1) // 2


def _resize_like(inputs, ref):
    _, _, i_h, i_w = inputs.shape
    _, _, r_h, r_w = ref.shape
    if i_h == r_h and i_w == r_w:
        return inputs
    else:
        resized = F.interpolate(inputs, (r_h, r_w), mode='nearest')
        return resized

### Test ###

model = Model()
x = torch.rand(1, 3, 375, 1424)
out = model(x)
print(out)