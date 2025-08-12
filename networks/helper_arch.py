import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm


def GroupNorm(in_channels):
    ec = 32
    assert in_channels % ec == 0
    return torch.nn.GroupNorm(num_groups=in_channels//32, num_channels=in_channels, eps=1e-6, affine=True)

def swish(x):
    return x*torch.sigmoid(x)

class ResTextBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = GroupNorm(in_channels)
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.norm2 = GroupNorm(out_channels)
        self.conv2 = SpectralNorm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        return x + x_in
    



def calc_mean_std_4D(feat, eps=1e-6):
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(prior_feat, lq_feat):
    size = prior_feat.size()
    lq_mean, lq_std = calc_mean_std_4D(lq_feat)
    prior_mean, prior_std = calc_mean_std_4D(prior_feat)

    normalized_feat = (prior_feat - prior_mean.expand(size)) / prior_std.expand(size)
    return normalized_feat * lq_std.expand(size) + lq_mean.expand(size)


def network_param(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params / 1e6