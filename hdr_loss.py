import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from skimage.metrics import structural_similarity

def get_lum(img):
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    return luminance

def get_gaussian_kernel(size, sigma=2):
    interval = (2 * sigma + 1.) / size
    x = np.linspace(-sigma - interval / 2., sigma + interval / 2., size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()

    return nn.Parameter(torch.tensor([[kernel]])).float()

def bilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35
    if sigmaColor is None:
        sigmaColor = sigmaSpace

    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim()  # 6
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)

    # get gaussian kernel transfer into tensor
    weights_space = get_gaussian_kernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

    # final weight matrix
    weights = weights_space * weights_color
    weights_sum = weights.sum(dim=(-1, -2))
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix

class HDRLoss(nn.Module):
    def __init__(self, device, eps=1.0 / 255.0, tau=0.95, weight=0.8):
        super(HDRLoss, self).__init__()
        self.eps = eps
        self.tau = tau
        self.weight = weight
        self.gaussian_kernel = get_gaussian_kernel(size=5).to(device)

    def forward(self, inputs, outputs, targets, separate_loss=False):
        batch_size, _, width, height = outputs.shape
        total_loss = 0.0
        inputs = inputs.float()

        for b in range(batch_size):
            #set unsaturated mask
            max_c = inputs[b].max(dim=0).values - self.tau
            max_c[max_c < 0] = 0
            alpha = (max_c / (1 - self.tau)).float()

            if separate_loss:

                # target: linear HDR image
                target_luminance = get_lum(targets[b].permute(1, 2, 0))
                target_illumination = bilateralFilter(target_luminance.unsqueeze(0).unsqueeze(0), 7, 0.02, 2).squeeze()
                target_reflectance = targets[b].permute(1, 2, 0) - target_illumination.repeat(3, 1, 1).permute(1, 2, 0)

                # output: denoise the output HDR image
                output_luminance = get_lum(outputs[b].permute(1, 2, 0))
                output_illumination = bilateralFilter(output_luminance.unsqueeze(0).unsqueeze(0), 7, 0.02, 2).squeeze()
                output_reflectance = outputs[b].permute(1, 2, 0) - output_illumination.repeat(3, 1, 1).permute(1, 2, 0)

                # compute loss
                #ssim = structural_similarity(targets[b].permute(1,2,0).cpu().data.numpy(),outputs[b].permute(1,2,0).cpu().data.numpy(),data_range=1.0,multichannel=True)
                illumination_loss = self.weight * torch.mean((alpha * (target_illumination - output_illumination)) ** 2)
                reflectance_loss = (1 - self.weight) * torch.mean(alpha.repeat(3, 1, 1).permute(1, 2, 0) * (target_reflectance - output_reflectance) ** 2)

                total_loss += illumination_loss + reflectance_loss

            else:
                loss_target_output = torch.mean(alpha * (outputs[b] - targets[b]) ** 2)
                loss_input_output = torch.mean(alpha * (outputs[b] - (inputs[b] ** 2)) ** 2)
                total_loss += loss_target_output + loss_input_output

        return total_loss


