import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import os
import pdb
import torch.fft as FFT
from torch.autograd import Variable
from math import exp


# ---- save np.array into nii ----
def save_nii(nii_image, nii_path):
    new_image = nib.Nifti1Image(nii_image, np.eye(4))  # np.array
    nib.save(new_image, nii_path)


# ---- Load nii image into np.array ----
def load_nii(nii_path):
    img_nii = nib.load(nii_path)
    img_np = img_nii.get_data()
    return img_np


# ---- Print property ----
def disp_tensor(tensor_):
    print(tensor_)
    print(tensor_.size())
    print(type(tensor_))

def disp_array(array_):
    print(array_)
    print(array_.shape)
    print(type(array_))


# ---- IFFT Recon ----
def IFFT_Recon(kspace_volume):
    num_ch = int(kspace_volume.size(1) / 2)  # 16

    # Kspace data
    kspace_real = kspace_volume[:, 0:num_ch, :, :]
    kspace_imag = kspace_volume[:, num_ch: 2 * num_ch, :, :]
    kspace_complex = torch.complex(kspace_real, kspace_imag)  # torch.Size([B, 16, 384, 768])

    # ifftshift + ifftn
    recon_complex = FFT.ifftshift(FFT.ifftn(kspace_complex, dim=[2, 3]), dim=[2, 3])  # torch.Size([B, 16, 384, 768]), <class 'torch.Tensor'>
    recon_volume = torch.cat((recon_complex.real, recon_complex.imag), 1)  # torch.Size([1, 32, 384, 768])

    return recon_volume


# ---- FFT Proj ----
def FFT_Proj(recon_volume):
    num_ch = int(recon_volume.size(1) / 2)  # 16

    # Recon data
    recon_real = recon_volume[:, 0:num_ch, :, :]
    recon_imag = recon_volume[:, num_ch: 2 * num_ch, :, :]
    recon_complex = torch.complex(recon_real, recon_imag)  # torch.Size([B, 16, 384, 768])

    # fftn + fftshift
    kspace_back_complex = FFT.fftn(FFT.fftshift(recon_complex, dim=[2, 3]), dim=[2, 3])
    kspace_back_volume = torch.cat((kspace_back_complex.real, kspace_back_complex.imag), 1)

    return kspace_back_volume


# ---- Absolute of complex value (k-space and recon) ----
def ABS_Complex(complex_volume):
    num_ch = int(complex_volume.size(1) / 2)  # 16

    # Real and Imag
    complex_volume_real = complex_volume[:, 0:num_ch, :, :]
    complex_volume_imag = complex_volume[:, num_ch: 2 * num_ch, :, :]

    # ABS
    abs_complex = (complex_volume_real.pow(2) + complex_volume_imag.pow(2)).sqrt()

    return abs_complex


# ---- RSS Recon ----
def RSS_Recon(recon_origin):
    num_ch = int(recon_origin.size(1) / 2)  # 16

    # Real and Imag
    recon_origin_real = recon_origin[:, 0:num_ch, :, :]
    recon_origin_imag = recon_origin[:, num_ch: 2 * num_ch, :, :]

    # RSS recon
    rss_recon = (recon_origin_real.pow(2) + recon_origin_imag.pow(2)).sum(1).sqrt().unsqueeze(1)

    return rss_recon



# ---- Data Consistency ----
def Data_Consistency(kspace_pred, kspace_truth, mask, lambda_=0.01):
    kspace_consistency = (lambda_ * torch.mul(kspace_pred, mask) + kspace_truth)/(lambda_ + 1) + torch.mul(kspace_pred, 1 - mask)
    return kspace_consistency


# ---- SSIM Loss ----
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


# ---- xxxx ----



