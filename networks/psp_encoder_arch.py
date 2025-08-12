import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm

import math
from .prior_arch import PixelNorm, EqualLinear
import torchvision
from torchvision.utils import save_image

def GroupNorm(in_channels):
    return torch.nn.GroupNorm(num_groups=in_channels//16, num_channels=in_channels, eps=1e-6, affine=False)

Norm = GroupNorm

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.norm1 = Norm(planes)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = conv3x3(planes, planes, stride)
        self.relu2 = nn.LeakyReLU(0.2)
        self.norm2 = Norm(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu3 = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu3(out)
        return out
        

class PSPEncoder(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 6, 3], strides=[(2,2),(1,2),(2,2),(1,2),(2,2)]):
        self.inplanes = 32
        super(PSPEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.relu = nn.LeakyReLU(0.2)

        feature_out_dim = 256
        self.layer1 = self._make_layer(block, 32, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[3])
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[4])

        self.layer512_to_outdim = nn.Sequential(
                nn.Conv2d(512, feature_out_dim, kernel_size=1, stride=1, bias=False),
                nn.LeakyReLU(0.2)
            )
        self.layer256_to_512 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
                nn.LeakyReLU(0.2)
            )


        self.down_h = 1
        for stride in strides:
            self.down_h *= stride[0]
        self.size_h = 32 // self.down_h * 2

    
        self.feature2w = nn.Sequential(
            PixelNorm(),
            EqualLinear(self.size_h*self.size_h*feature_out_dim, 512, bias=True, bias_init_val=0, lr_mul=1, activation='fused_lrelu'),
            EqualLinear(512, 512, bias=True, bias_init_val=0, lr_mul=1, activation='fused_lrelu'),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.LeakyReLU(0.2)
            )
                # GroupNorm(planes),
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def _check_outliers(self, crop_feature, target_width):
        B, C, H, W = crop_feature.size()
        if W != target_width:
            return F.interpolate(crop_feature, size=(H, target_width), mode='bilinear', align_corners=True)
        else:
            return crop_feature

    def _check_outliers_pad(self, crop_feature, start, end, max_lr_width, center_loc, extend_W):
        _, _, H, W = crop_feature.size()
        fill_value = crop_feature.mean().item()
        if start == 0 and end == max_lr_width:
            crop_feature = torchvision.transforms.Pad([extend_W//2-center_loc, 0, extend_W-W-(extend_W//2-center_loc), 0], fill=fill_value, padding_mode='constant')(crop_feature)
        else:
            if start == 0:
                crop_feature = torchvision.transforms.Pad([extend_W-W, 0, 0, 0], fill=fill_value, padding_mode='constant')(crop_feature)
            if end == max_lr_width:
                crop_feature = torchvision.transforms.Pad([0, 0, extend_W-W, 0], fill=fill_value, padding_mode='constant')(crop_feature)

        # if crop_feature.size(3) != extend_W:
        #     print([222, crop_feature.size(), extend_W])
        #     crop_feature = torchvision.transforms.Pad([(extend_W-W)//2, 0, extend_W-W-(extend_W-W)//2, 0], fill=0, padding_mode='constant')(crop_feature)

        return crop_feature
        

    def forward(self, x, locs):
        w_b = []
        extend_W = 32*4
        max_lr_width = x.size(3)
        for b in range(locs.size(0)): #locs: 0~2048
            x_for_w = []
            for c in range(locs.size(1)):
                center_loc = (locs[b][c]/4).int()
                start_x = max(0, center_loc - extend_W//2)
                end_x = min(center_loc + extend_W//2, max_lr_width)
                crop_x = x[b:b+1, :, :, start_x:end_x].detach()
                crop_x = self._check_outliers_pad(crop_x, start_x, end_x, max_lr_width, center_loc, extend_W) # 

                x_for_w.append(crop_x)
                # crop_x[...,62:66] = 1
                # save_image((crop_x+1)/2, 'trs_{}.png'.format(c))

            x_for_w = torch.cat(x_for_w, dim=0)
            
            x_c1 = self.conv1(x_for_w) #1
            x_c1 = self.relu(x_c1) 
            x_l1 = self.layer1(x_c1) #2
            x_l2 = self.layer2(x_l1) #1 [2, 64, 16, 256])
            x_l3 = self.layer3(x_l2) #2 torch.Size([2, 128, 8, 128]
            x_l4 = self.layer4(x_l3) #1 torch.Size([2, 256, 8, 128])
            x_l5 = self.layer5(x_l4) #2, torch.Size([2, 512, 4, 64])
            pyramid_x1 = _upsample_add(x_l5, self.layer256_to_512(x_l4))
            pyramid_x = self.layer512_to_outdim(pyramid_x1)
            w_each_b = self.feature2w(pyramid_x.view(pyramid_x.size(0), -1)) # 
            
            w_c = w_each_b
            w_b.append(w_c)
        w_b = torch.stack(w_b, dim=0)

        return w_b
 

 
        # w_b = []
        # for b in range(locs.size(0)): #locs: 0~2048
        #     w_c = []
        #     for c in range(locs.size(1)):
        #         if locs[b][c] < 2048:
        #             center_loc = (locs[b][c]/4).int() # 32*512
        #             start_x = center_loc - 16
        #             end_x = center_loc + 16
                    
        #             crop_x0 = x[b:b+1, :, :, start_x:end_x].clone()
        #             crop_x = self._check_outliers_pad(crop_x0, start_x, end_x) # 1, 512, 4, 4 or 1, 512, 8, 8
                    
        #             # save_image(crop_x[0], 'ss_{}.png'.format(c))
        #             x_c1 = self.conv1(crop_x) #1
        #             x_c1 = self.relu(x_c1) 
        #             x_l1 = self.layer1(x_c1) #2
        #             x_l2 = self.layer2(x_l1) #1 [2, 64, 16, 256])
        #             x_l3 = self.layer3(x_l2) #2 torch.Size([2, 128, 8, 128]
        #             x_l4 = self.layer4(x_l3) #1 torch.Size([2, 256, 8, 128])
        #             x_l5 = self.layer5(x_l4) #2, torch.Size([2, 512, 4, 64])
        #             pyramid_x1 = _upsample_add(x_l5, self.layer256_to_512(x_l4))
        #             pyramid_x = self.layer512_to_outdim(pyramid_x1)

        #             w = self.feature2w(pyramid_x.view(1, -1)) # 1*512
        #             w_c.append(w.squeeze(0))
        #         else:
        #             w_c.append(w.squeeze(0).detach()*0)
        #     w_c = torch.stack(w_c, dim=0)
        #     w_b.append(w_c)
        # w_b = torch.stack(w_b, dim=0)
        # print(w_b.size())
        # return w_b #, lr



        
        # # lr = x.clone()
        # x_c1 = self.conv1(x) #1
        # x_c1 = self.relu(x_c1) 
        # x_l1 = self.layer1(x_c1) #2
        # x_l2 = self.layer2(x_l1) #1 [2, 64, 16, 256])
        # x_l3 = self.layer3(x_l2) #2 torch.Size([2, 128, 8, 128]
        # x_l4 = self.layer4(x_l3) #1 torch.Size([2, 256, 8, 128])
        # x_l5 = self.layer5(x_l4) #2, torch.Size([2, 512, 4, 64]) B, 512, 4, 64, 17M parameters

        # pyramid_x1 = _upsample_add(x_l5, self.layer256_to_512(x_l4))
        # pyramid_x = self.layer512_to_outdim(pyramid_x1)
        # # pyramid_x2 = _upsample_add(self.layer128_to_outdim(x_l3), pyramid_x1)
        # B, C, H, W = pyramid_x.size()
        # w_b = []
        # for b in range(locs.size(0)): #locs: 0~2048
        #     w_c = []
        #     for c in range(locs.size(1)):
        #         if locs[b][c] < 2048:
        #             center_loc = (locs[b][c]/4/self.down_h).int() # from 32*512 to 4*64
        #             start_x = max(0, center_loc-self.size_h//2)
        #             end_x = min(center_loc+self.size_h//2, 512//self.down_h)
        #             # crop_feature = pyramid_x2[b:b+1, :, :, start_x:end_x].clone()
                    
        #             # if end_x - start_x != self.size_h:
        #             #     bgfill = torch.zeros((B, C, H, self.size_h), dtype=pyramid_x2.dtype, layout=pyramid_x2.layout, device=pyramid_x2.device)
        #             #     bgfill[:, :, :, self.size_h//2 - (center_loc - start_x):self.size_h//2 - (center_loc - start_x) + end_x - start_x] += pyramid_x2[b:b+1, :, :, start_x:end_x].clone() 
        #             #     crop_feature = bgfill.clone()
        #             # else:
        #             #     crop_feature = pyramid_x2[b:b+1, :, :, start_x:end_x].clone()
                    
        #             crop_feature = pyramid_x[b:b+1, :, :, start_x:end_x].clone()
        #             crop_feature = self._check_outliers(crop_feature, self.size_h) # 1, 512, 4, 4 or 1, 512, 8, 8

        #             # crop_feature = self._check_outliers(crop_feature, self.size_h, start_x, end_x) # 1, 512, 4, 4 or 1, 512, 8, 8
        #             print(crop_feature.size())
        #             w = self.feature2w(crop_feature.view(1, -1)) # 1*512
        #             w_c.append(w.squeeze(0))

        #         else:
        #             w_c.append(w.squeeze(0).detach()*0)

        #         # lr[b:b+1, :, :, center_loc-1:center_loc+1] = 255

        #     w_c = torch.stack(w_c, dim=0)
        #     w_b.append(w_c)
        # w_b = torch.stack(w_b, dim=0)
        
        # return w_b #, x #, lr
        



def GroupNorm(in_channels):
    return torch.nn.GroupNorm(num_groups=in_channels//32, num_channels=in_channels, eps=1e-6, affine=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
     
    


def _upsample_add(x, y):
    '''Upsample and add two feature maps.
    Args:
        x: (Variable) top feature map to be upsampled.
        y: (Variable) lateral feature map.
    Returns:
        (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    '''
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y


