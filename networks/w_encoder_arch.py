import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .prior_arch import PixelNorm, EqualLinear


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.gn1 = GroupNorm(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.gn2 = GroupNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
        
class WEncoder(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 6, 3], strides=[2,1,2,1,2]):
        self.inplanes = 32
        super(WEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        feature_out_dim = 512
        self.layer1 = self._make_layer(block, 32, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[3])
        self.layer5 = self._make_layer(block, feature_out_dim, layers[4], stride=strides[4])


        self.down_h = 1
        for stride in strides:
            self.down_h *= stride
        self.size_h = 32 // self.down_h

    
        self.feature2w = nn.Sequential(
            PixelNorm(),
            EqualLinear(self.size_h*self.size_h*feature_out_dim, 512, bias=True, bias_init_val=0, lr_mul=1,
                    activation='fused_lrelu'),
            EqualLinear(512, 512, bias=True, bias_init_val=0, lr_mul=1,
                    activation='fused_lrelu')
            # EqualLinear(self.size_h*self.size_h*feature_out_dim, 512, bias=True),
            # EqualLinear(512, 512, bias=True)
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
            )
                # GroupNorm(planes),

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _check_outliers(self, crop_feature, target_width):
        _, _, H, W = crop_feature.size()
        if W != target_width:
            return F.interpolate(crop_feature, size=(H, target_width), mode='bilinear', align_corners=True)
        else:
            return crop_feature


    def forward(self, x, locs):
        # lr = x.clone()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x) # B, 512, 4, 64, 17M parameters
  
        B, C, H, W = x.size()

        # lr = F.interpolate(lr, (x.size(2), x.size(3)))
        w_b = []
        for b in range(locs.size(0)): #locs: 0~2048
            w_c = []
            for c in range(locs.size(1)):
                if locs[b][c] < 2048:
                    center_loc = (locs[b][c]/4/self.down_h).int() # from 32*512 to 4*64
                    start_x = max(0, center_loc-self.size_h//2)
                    end_x = min(center_loc+self.size_h//2, 512//self.down_h)
                    
                    # crop_feature = x[b:b+1, :, :, start_x:end_x].clone()
                    # crop_feature = self._check_outliers(crop_feature, self.size_h) # 1, 512, 4, 4 or 1, 512, 8, 8
                    
                    if end_x - start_x != self.size_h:
                        bgfill = torch.zeros((B, C, H, self.size_h), dtype=x.dtype, layout=x.layout, device=x.device)
                        bgfill[:, :, :, self.size_h//2 - (center_loc - start_x):self.size_h//2 - (center_loc - start_x) + end_x - start_x] += x[b:b+1, :, :, start_x:end_x].clone() 
                        crop_feature = bgfill.clone()
                    else:
                        crop_feature = x[b:b+1, :, :, start_x:end_x].clone()
                    w = self.feature2w(crop_feature.view(1, -1)) # 1*512
                    w_c.append(w.squeeze(0))

                else:
                    w_c.append(w.squeeze(0).detach()*0)

            w_c = torch.stack(w_c, dim=0)
            w_b.append(w_c)
        w_b = torch.stack(w_b, dim=0)
        
        return w_b #, lr



def GroupNorm(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=False)


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




if __name__ == '__main__':
    from .helper_arch import network_param

    device = 'cuda'
    input = torch.randn(2, 3, 32, 512).to(device) #

    test_list = [64]
    for i in range(1, 8):
        test_list.append(64+128*i)
    
    for i in range(8, 16):
        test_list.append(2048)

    
    
    locs = torch.Tensor(test_list).unsqueeze(0)
    locs = locs.repeat(2, 1).to(device)
    net = WEncoder().to(device)
    '''
    strides=[2,1,2,1,1] output h is 8
    Encoder is 12.97M
    F2W+Encoder is 17.04 M

    strides=[2,1,2,1,2] output h is 4
    Encoder is 12.97M
    F2W is 4.46 M

    '''

    output = net(input, locs)
    print([input.size(), output.size(), locs.size(), network_param(net)])
    #[torch.Size([2, 3, 32, 512]), torch.Size([2, 16, 512]), torch.Size([2, 16]), 17.43344]

    # import numpy as np
    # import cv2
    # sr_results = lr[0].permute(1, 2, 0)
    # sr_results = sr_results.float().cpu().numpy()
    
    # cv2.imwrite('./tmp.png', sr_results)


