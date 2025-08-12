import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
import random
from .helper_arch import ResTextBlockV2, adaptive_instance_normalization

class SRNet(nn.Module):
    def __init__(self, in_channel=3, dim_channel=256):
        super().__init__()
        self.conv_first_32 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, dim_channel//4, 3, 1, 1)),
            nn.LeakyReLU(0.2),
        )
        self.conv_first_16 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel//4, dim_channel//2, 3, 2, 1)),
            nn.LeakyReLU(0.2),
        )
        self.conv_first_8 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel//2, dim_channel, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_body_16 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel+dim_channel//2, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_body_32 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel+dim_channel//4, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )

        self.conv_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), #64*64*256
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            ResTextBlockV2(dim_channel, dim_channel),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )

        self.conv_final = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel//2, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear'), #128*128*256
            SpectralNorm(nn.Conv2d(dim_channel//2, dim_channel//4, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            ResTextBlockV2(dim_channel//4, dim_channel//4),
            SpectralNorm(nn.Conv2d(dim_channel//4, 3, 3, 1, 1)),
            nn.Tanh()
        )

        # self.conv_priorout = nn.Sequential(
        #     SpectralNorm(nn.Conv2d(dim_channel, dim_channel//2, 3, 1, 1)),
        #     nn.LeakyReLU(0.2),
        #     nn.Upsample(scale_factor=2, mode='bilinear'), #128*128*256
        #     SpectralNorm(nn.Conv2d(dim_channel//2, dim_channel//4, 3, 1, 1)),
        #     nn.LeakyReLU(0.2),
        #     ResTextBlockV2(dim_channel//4, dim_channel//4),
        #     SpectralNorm(nn.Conv2d(dim_channel//4, 3, 3, 1, 1)),
        #     nn.Tanh()
        # )

        self.conv_32_scale = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_32_shift = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_32_fuse = nn.Sequential(
            ResTextBlockV2(2*dim_channel, dim_channel)
        )

        self.conv_32_to256 = nn.Sequential(
            SpectralNorm(nn.Conv2d(512, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )

        self.conv_64_scale = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_64_shift = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim_channel, dim_channel, 3, 1, 1)),
        )
        self.conv_64_fuse = nn.Sequential(
            ResTextBlockV2(2*dim_channel, dim_channel)
        )

    def forward(self, lq, priors64, priors32, locs): #
        # lq_features:b*512*8*512 
        # priors: 8, 16,32,64,128
        # locs: b*32, center+width for 128*2048 0~1
        # locs: b*16, center for 128*2048, 0~2048

        
        single_sr = True

        
        lq_f_32 = self.conv_first_32(lq)
        lq_f_16 = self.conv_first_16(lq_f_32)
        lq_f_8 = self.conv_first_8(lq_f_16)

        sq_f_16 = self.conv_body_16(torch.cat([F.interpolate(lq_f_8, scale_factor=2, mode='bilinear'), lq_f_16], dim=1))
        sq_f_32 = self.conv_body_32(torch.cat([F.interpolate(sq_f_16, scale_factor=2, mode='bilinear'), lq_f_32], dim=1)) # 256*32*32


        if priors32 is not None:
            sq_f_32_ori = sq_f_32.clone()
            # sq_f_32_res = sq_f_32.clone().detach()*0
            prior_32_align = torch.zeros_like(sq_f_32_ori)
            prior_32_mask = torch.zeros((sq_f_32_ori.size(0), 1, sq_f_32_ori.size(2), sq_f_32_ori.size(3)), dtype=sq_f_32_ori.dtype, layout=sq_f_32_ori.layout, device=sq_f_32_ori.device)
            for b, p_32 in enumerate(priors32): #512*32*32, different batch
                p_32_256 = self.conv_32_to256(p_32.clone()) 
                for c in range(p_32_256.size(0)): #
                    center = (locs[b][c].detach()/4.0).int() #
                    width = 16

                    if center < width:
                        x1 = 0 #lq feature left
                        y1 = max(16 - center, 0)
                    else:
                        x1 = center - width
                        y1 = max(16 - width, 0)
                        # y1 = 16 - width
                    if center + width > sq_f_32.size(-1):
                        x2 = sq_f_32.size(-1) #lq feature right
                    else:
                        x2 = center + width
                    y2 = y1 + (x2 - x1)

                    '''
                    center align
                    '''
                    # y1 = 16 - torch.div(x2-x1, 2, rounding_mode='trunc')
                    y2 = y1 + x2 - x1

                    if single_sr:
                        char_prior_f = p_32_256[c:c+1, :, :, y1:y2].clone() #prior
                        char_lq_f = sq_f_32[b:b+1, :, :, x1:x2].clone()
                        adain_prior_f = adaptive_instance_normalization(char_prior_f, char_lq_f)
                        fuse_32_prior = self.conv_32_fuse(torch.cat((adain_prior_f, char_lq_f), dim=1))
                        scale = self.conv_32_scale(fuse_32_prior)
                        shift = self.conv_32_shift(fuse_32_prior)
                        prior_32_align[b, :, :, x1:x2] = prior_32_align[b, :, :, x1:x2] + sq_f_32[b, :, :, x1:x2].clone() * scale[0,...] + shift[0,...]
                    else:
                        prior_32_align[b, :, :, x1:x2] = prior_32_align[b, :, :, x1:x2] + p_32_256[c:c+1, :, :, y1:y2].clone()
            #         prior_32_mask[b, :, :, x1:x2] += 1.0
            
            # prior_32_mask[prior_32_mask<2]=1.0
            # prior_32_align = prior_32_align / prior_32_mask.repeat(1, prior_32_align.size(1), 1, 1)

            if single_sr:
                sq_pf_32_out = sq_f_32_ori + prior_32_align
            else:
                sq_f_32_norm = adaptive_instance_normalization(prior_32_align, sq_f_32)
                sq_f_32_fuse = self.conv_32_fuse(torch.cat((sq_f_32_norm, sq_f_32), dim=1))
                scale_32 = self.conv_32_scale(sq_f_32_fuse)
                shift_32 = self.conv_32_shift(sq_f_32_fuse)
                sq_f_32_res = sq_f_32_ori * scale_32 + shift_32
                sq_pf_32_out = sq_f_32_ori + sq_f_32_res
            
        else:
            sq_pf_32_out = sq_f_32.clone()


        sq_f_64 = self.conv_up(sq_pf_32_out) #64*1024
        
        sq_f_64_ori = sq_f_64.clone()
        prior_64_align = torch.zeros_like(sq_f_64_ori)
        prior_64_mask = torch.zeros((sq_f_64_ori.size(0), 1, sq_f_64_ori.size(2), sq_f_64_ori.size(3)), dtype=sq_f_64_ori.dtype, layout=sq_f_64_ori.layout, device=sq_f_64_ori.device)
 
        
        for b, p_64_prior in enumerate(priors64): #512*8*8, 512*16*16, 512*32*32, 256*64*64, 128*128*128 different batch
            p_64 = p_64_prior.clone() #.detach() #no backward to prior
            for c in range(p_64.size(0)): # for each character
                center = (locs[b][c].detach()/2.0).int() #+ random.randint(-4,4)### no backward 
                width = 32
                if center < width:
                    x1 = 0
                    y1 = max(32 - center, 0)
                else:
                    x1 = center -width
                    y1 = max(32 - width, 0)
                if center + width > sq_f_64.size(-1):
                    x2 = sq_f_64.size(-1)
                else:
                    x2 = center + width

                '''
                center align
                '''
                # y1 = 32 - torch.div(x2-x1, 2, rounding_mode='trunc')
                y2 = y1 + x2 - x1
                
                if single_sr:
                    char_prior_f = p_64[c:c+1, :, :, y1:y2].clone()
                    char_lq_f = sq_f_64[b:b+1, :, :, x1:x2].clone()
                    adain_prior_f = adaptive_instance_normalization(char_prior_f, char_lq_f)

                    fuse_64_prior = self.conv_64_fuse(torch.cat((adain_prior_f, char_lq_f), dim=1))
                    scale = self.conv_64_scale(fuse_64_prior)
                    shift = self.conv_64_shift(fuse_64_prior)
                    prior_64_align[b, :, :, x1:x2] = prior_64_align[b, :, :, x1:x2] + sq_f_64[b, :, :, x1:x2].clone() * scale[0,...] + shift[0,...]
                else:
                    prior_64_align[b, :, :, x1:x2] = prior_64_align[b, :, :, x1:x2] + p_64[c:c+1, :, :, y1:y2].clone()
        #         prior_64_mask[b, :, :, x1:x2] += 1.0
                

        # prior_64_mask[prior_64_mask<2]=1.0
        # prior_64_align = prior_64_align / prior_64_mask.repeat(1, prior_64_align.size(1), 1, 1)
        if single_sr:
            sq_pf_64 = sq_f_64_ori + prior_64_align
        else:
            sq_f_64_norm = adaptive_instance_normalization(prior_64_align, sq_f_64_ori)
            sq_f_64_fuse = self.conv_64_fuse(torch.cat((sq_f_64_norm, sq_f_64_ori), dim=1))
            scale_64 = self.conv_64_scale(sq_f_64_fuse)
            shift_64 = self.conv_64_shift(sq_f_64_fuse)
            sq_f_64_res = sq_f_64_ori * scale_64 + shift_64
            sq_pf_64 = sq_f_64_ori + sq_f_64_res

        f256 = self.conv_final(sq_pf_64)

        # adain_lr2prior = adaptive_instance_normalization(prior_full_64, F.interpolate(sq_f_32_ori, scale_factor=2, mode='bilinear'))
        # prior_out = self.conv_priorout(adain_lr2prior)

        return f256 #prior_out