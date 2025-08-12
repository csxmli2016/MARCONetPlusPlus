import torch
from torch import nn
from torch.nn import functional as F
import math
from op.fused_act import FusedLeakyReLU, fused_leaky_relu


class TextPriorModel(nn.Module):
    def __init__(
        self,
        size=128,
        style_dim=512,
        n_mlp=8,
        class_num=6736,
        lr_mlp=0.01,
    ):
        super().__init__()
        self.TextGenerator = StyleCharacter(size=size, style_dim=style_dim, n_mlp=n_mlp, class_num=class_num, lr_mlp=lr_mlp)
        # '''
        # 下面作用是停止《生成先验》参数更新
        # '''
        # for param_g in self.TextGenerator.parameters():
        #     param_g.requires_grad = False

    def forward(self, styles, labels, noise):
        return self.TextGenerator(styles, labels, noise)
    
class StyleCharacter(nn.Module):
    def __init__(
        self,
        size=128,
        style_dim=512,
        n_mlp=8,
        class_num=6736,
        channel_multiplier=1,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()
        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        style_mlp_layers = [PixelNorm()]

        for i in range(n_mlp):
            style_mlp_layers.append(
                EqualLinear(
                    style_dim, style_dim, bias=True, bias_init_val=0, lr_mul=lr_mlp,
                    activation='fused_lrelu'))
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input_text = SelectText(class_num, self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)
        self.log_size = int(math.log(size, 2)) #7

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        self.n_latent = self.log_size * 2 - 2
    def forward(
        self,
        styles,
        labels, 
        noise=None,
    ):
        styles = self.style_mlp(styles)#
        latent = styles.unsqueeze(1).repeat(1, self.n_latent, 1) #
        out = self.input_text(labels) #4*4

        out = self.conv1(out, latent[:, 0], noise=None)
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        noise_i = 3
        for conv1, conv2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=None)
            out = conv2(out, latent[:, i + 1], noise=None) 
            skip = to_rgb(out.clone(), latent[:, i + 2], skip)
            if out.size(-1) == 64:
                prior_features64 = out.clone() # only 
                prior_rgb64 = skip.clone()
            if out.size(-1) == 32:
                prior_features32 = out.clone() # only 
                prior_rgb32 = skip.clone()
            i += 2
            noise_i += 2
        image = skip

        return image, prior_features64, prior_features32 #, prior_rgb64, prior_rgb32 #prior_features 7
    


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class EqualLinear(nn.Module):
    """Equalized Linear as StyleGAN2.
    Args:
        in_channels (int): Size of each sample.
        out_channels (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        lr_mul (float): Learning rate multiplier. Default: 1.
        activation (None | str): The activation after ``linear`` operation.
            Supported: 'fused_lrelu', None. Default: None.
    """

    def __init__(self, in_channels, out_channels, bias=True, bias_init_val=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_mul = lr_mul
        self.activation = activation
        if self.activation not in ['fused_lrelu', None]:
            raise ValueError(f'Wrong activation value in EqualLinear: {activation}'
                             "Supported ones are: ['fused_lrelu', None].")
        self.scale = (1 / math.sqrt(in_channels)) * lr_mul

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        if self.activation == 'fused_lrelu':
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(x, self.weight * self.scale, bias=bias)
        return out
    


class SelectText(nn.Module):
    def __init__(self, class_num, channel, size=4):
        super().__init__()
        self.size = size
        self.TextEmbeddings = nn.Parameter(torch.randn(class_num, channel, 1, 1))
    def forward(self, labels):
        b, c = labels.size()

        TestEmbs = []
        for i in range(b):
            EmbTmps = []
            for j in range(c):
                EmbTmps.append(self.TextEmbeddings[labels[i][j]:labels[i][j]+1,...].repeat(1,1,self.size,self.size)) #
            Seqs = torch.cat(EmbTmps, dim=3)
            TestEmbs.append(Seqs)
        OutEmbs = torch.cat(TestEmbs, dim=0)
        return OutEmbs


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.activate = FusedLeakyReLU(out_channel)
    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = out + self.bias
        out = self.activate(out)
        return out
    


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
       
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        
        self.modulation = EqualLinear(style_dim, in_channel, bias=True, bias_init_val=1, lr_mul=1, activation=None)

        self.demodulate = demodulate


    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            out = self.up(input)
            out = F.conv2d(out, weight, padding=1, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.upsample = upsample
        out_dim = 1
        self.conv = ModulatedConv2d(in_channel, out_dim, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(
                    skip, scale_factor=2, mode='bilinear', align_corners=False)
            out = out + skip
        return torch.tanh(out)
