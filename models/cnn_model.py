import os
from math import log10
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import time

from networks import get_generator
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, get_gan_loss, psnr, mse, get_nonlinearity, nmse, nmae
from skimage.metrics import structural_similarity as ssim_ski
from utils.data_patch_util import *
import pdb
import scipy.io as scio
from utils.Utility import *

class CNNModel(nn.Module):
    def __init__(self, opts):
        super(CNNModel, self).__init__()
        self.loss_names = []
        self.networks = []
        self.optimizers = []

        # Learning Rate
        self.lr = opts.lr_I1
        self.lr_I1 = opts.lr_I1
        self.lr_K1 = opts.lr_K1
        self.lr_I2 = opts.lr_I2
        self.lr_K2 = opts.lr_K2

        # Network
        self.net_I1 = get_generator(opts.net_G, opts)
        self.net_K1 = get_generator(opts.net_G, opts)
        self.net_I2 = get_generator(opts.net_G, opts)
        self.net_K2 = get_generator(opts.net_G, opts)
        self.networks.append(self.net_I1)
        self.networks.append(self.net_K1)
        self.networks.append(self.net_I2)
        self.networks.append(self.net_K2)

        # Loss Name
        self.loss_names += ['loss_I1']
        self.loss_names += ['loss_K1']
        self.loss_names += ['loss_I2']
        self.loss_names += ['loss_K2']

        # Optimizer
        self.optimizer_I1 = torch.optim.Adam(self.net_I1.parameters(), lr=self.lr_I1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_K1 = torch.optim.Adam(self.net_K1.parameters(), lr=self.lr_K1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_I2 = torch.optim.Adam(self.net_I2.parameters(), lr=self.lr_I2, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_K2 = torch.optim.Adam(self.net_K2.parameters(), lr=self.lr_K2, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizers.append(self.optimizer_I1)
        self.optimizers.append(self.optimizer_K1)
        self.optimizers.append(self.optimizer_I2)
        self.optimizers.append(self.optimizer_K2)

        # Loss function
        self.lossfunc = nn.MSELoss()

        # Options
        self.opts = opts


    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.kspace          = data['kspace'].to(self.device).float()
        self.kspace_sv       = data['kspace_sv'].to(self.device).float()
        self.recon_origin    = data['recon_origin'].to(self.device).float()
        self.recon_origin_sv = data['recon_origin_sv'].to(self.device).float()  # [B, 32, 384, 768]
        self.mask            = data['mask'].to(self.device).float()  # [B, 32, 384, 768]
        self.recon_rss       = data['recon_rss'].to(self.device).float()
        self.recon_rss_sv    = data['recon_rss_sv'].to(self.device).float()  # [B, 1, 384, 768]

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):  # self.ksapce_sv & self.mask
        self.inp = self.kspace_sv
        self.inp.requires_grad_(True)

        # ------ #
        # net_I1: IFFT reconstruction + Image Net Prediction
        self.img_in_c1 = IFFT_Recon(self.inp)
        self.img_pred_c1 = self.net_I1(self.img_in_c1)

        # net_K1: FFT Proj + DC + Kspace Net Predictiopn + DC
        self.kspace_in_c1 = FFT_Proj(self.img_pred_c1)
        self.kspace_in_dc_c1 = Data_Consistency(self.kspace_in_c1, self.inp, self.mask, lambda_=0)
        self.kspace_in_dc_pred_c1 = self.net_K1(self.kspace_in_dc_c1)
        self.kspace_in_dc_pred_dc_c1 = Data_Consistency(self.kspace_in_dc_pred_c1, self.inp, self.mask, lambda_=0)

        # ------ #
        # net_I2: IFFT reconstruction + Image Net Prediction
        self.img_in_c2 = IFFT_Recon(self.kspace_in_dc_pred_dc_c1)
        self.img_pred_c2 = self.net_I2(self.img_in_c2 + self.img_pred_c1)  # Concatenation

        # net_K2: FFT Proj + DC + Kspace Net Predictiopn + DC
        self.kspace_in_c2 = FFT_Proj(self.img_pred_c2)
        self.kspace_in_c2_fuse = self.kspace_in_c2 + self.kspace_in_dc_pred_dc_c1
        self.kspace_in_dc_c2 = Data_Consistency(self.kspace_in_c2_fuse, self.inp, self.mask, lambda_=0)
        self.kspace_in_dc_pred_c2 = self.net_K2(self.kspace_in_dc_c2)   # Concatenation
        self.kspace_in_dc_pred_dc_c2 = Data_Consistency(self.kspace_in_dc_pred_c2, self.inp, self.mask, lambda_=0)


    def update(self):
        # Zero gradient
        self.optimizer_I1.zero_grad()
        self.optimizer_K1.zero_grad()
        self.optimizer_I2.zero_grad()
        self.optimizer_K2.zero_grad()

        # Calculate the two loss idependently
        loss_I1 = self.lossfunc(self.img_pred_c1, self.recon_origin)
        loss_K1 = self.lossfunc(self.kspace_in_dc_pred_dc_c1, self.kspace)
        loss_I2 = self.lossfunc(self.img_pred_c2, self.recon_origin)
        loss_K2 = self.lossfunc(self.kspace_in_dc_pred_dc_c2, self.kspace)
        self.loss_I1 = loss_I1.item()
        self.loss_K1 = loss_K1.item()
        self.loss_I2 = loss_I2.item()
        self.loss_K2 = loss_K2.item()

        # Calculate the total loss
        loss_total = 0.25*loss_I1 + 0.25*loss_K1 + loss_I2 + loss_K2
        loss_total.backward()
        self.optimizer_I1.step()
        self.optimizer_K1.step()
        self.optimizer_I2.step()
        self.optimizer_K2.step()

    @property
    def loss_summary(self):
        message = ''
        message += ' loss: {:.4e}'.format(self.loss_K2)
        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        self.lr = self.optimizers[0].param_groups[0]['lr']


    def save(self, filename, epoch, total_iter):
        state = {}
        state['net_I1'] = self.net_I1.module.state_dict()
        state['net_K1'] = self.net_K1.module.state_dict()
        state['net_I2'] = self.net_I2.module.state_dict()
        state['net_K2'] = self.net_K2.module.state_dict()

        state['opt_I1'] = self.optimizer_I1.state_dict()
        state['opt_K1'] = self.optimizer_K1.state_dict()
        state['opt_I2'] = self.optimizer_I2.state_dict()
        state['opt_K2'] = self.optimizer_K2.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))


    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        self.net_I1.module.load_state_dict(checkpoint['net_I1'])
        self.net_K1.module.load_state_dict(checkpoint['net_K1'])
        self.net_I2.module.load_state_dict(checkpoint['net_I2'])
        self.net_K2.module.load_state_dict(checkpoint['net_K2'])
        if train:
            self.optimizer_I1.load_state_dict(checkpoint['opt_I1'])
            self.optimizer_K1.load_state_dict(checkpoint['opt_K1'])
            self.optimizer_I2.load_state_dict(checkpoint['opt_I2'])
            self.optimizer_K2.load_state_dict(checkpoint['opt_K2'])

        print('Loaded {}'.format(checkpoint_file))
        return checkpoint['epoch'], checkpoint['total_iter']


    # -------------- Evaluation, Calculate PSNR ---------------
    def evaluate(self, loader):
        val_bar = tqdm(loader)

        # For calculating metrics
        avg_psnr_recon = AverageMeter()
        avg_ssim_recon = AverageMeter()
        avg_nmse_recon = AverageMeter()
        avg_nmae_recon = AverageMeter()

        for data in val_bar:
            self.set_input(data)
            self.forward()

            # ---- RSS recon ----
            # self.recon_rss, self.recon_rss_sv
            self.recon_origin_pred = IFFT_Recon(self.kspace_in_dc_pred_dc_c2)
            self.recon_rss_pred = RSS_Recon(self.recon_origin_pred)

            # ---- recon quantification ----
            nmse_recon = nmse(self.recon_rss_pred, self.recon_rss)
            nmae_recon = nmae(self.recon_rss_pred, self.recon_rss)
            ssim_recon = ssim_ski(self.recon_rss_pred[0, 0, ...].cpu().numpy(), self.recon_rss[0, 0, ...].cpu().numpy())
            psnr_recon = psnr(self.recon_rss_pred, self.recon_rss)

            avg_nmse_recon.update(nmse_recon)
            avg_nmae_recon.update(nmae_recon)
            avg_ssim_recon.update(ssim_recon)
            avg_psnr_recon.update(psnr_recon)

            message = 'Recon - NMSE: {:4f}, NMAE: {:4f}, SSIM: {:4f}, PSNR: {:4f}'.format(avg_nmse_recon.avg, avg_nmae_recon.avg, avg_ssim_recon.avg, avg_psnr_recon.avg)
            val_bar.set_description(desc=message)

        # Calculate the average metrics
        self.nmse_recon = avg_nmse_recon.avg
        self.nmae_recon = avg_nmae_recon.avg
        self.ssim_recon = avg_ssim_recon.avg
        self.psnr_recon = avg_psnr_recon.avg


    # --------------- Save the images ------------------------------
    def save_images(self, loader, folder):
        val_bar = tqdm(loader)
        val_bar.set_description(desc='Saving images ...')

        # --------------- Mkdir save folder -------------------
        if not os.path.exists(os.path.join(folder, 'kspace_abs')):
            os.mkdir(os.path.join(folder, 'kspace_abs'))
        if not os.path.exists(os.path.join(folder, 'kspace_abs_sv')):
            os.mkdir(os.path.join(folder, 'kspace_abs_sv'))
        if not os.path.exists(os.path.join(folder, 'kspace_abs_pred')):
            os.mkdir(os.path.join(folder, 'kspace_abs_pred'))

        if not os.path.exists(os.path.join(folder, 'recon_rss')):
            os.mkdir(os.path.join(folder, 'recon_rss'))
        if not os.path.exists(os.path.join(folder, 'recon_rss_sv')):
            os.mkdir(os.path.join(folder, 'recon_rss_sv'))
        if not os.path.exists(os.path.join(folder, 'recon_rss_pred')):
            os.mkdir(os.path.join(folder, 'recon_rss_pred'))


        # Load data for each batch
        index = 0
        for data in val_bar:
            index += 1
            self.set_input(data)  # [batch_szie=1, 1, 64, 64, 64]
            self.forward()

            # ---- RSS recon ----
            # self.recon_rss, self.recon_rss_sv
            self.kspace_abs = ABS_Complex(self.kspace)
            self.kspace_abs_sv = ABS_Complex(self.kspace_sv)
            self.kspace_abs_pred = ABS_Complex(self.kspace_in_dc_pred_dc_c2)

            self.recon_origin_pred = IFFT_Recon(self.kspace_in_dc_pred_dc_c2)
            self.recon_rss_pred = RSS_Recon(self.recon_origin_pred)


            # save images
            save_nii(self.kspace_abs.squeeze().cpu().numpy(), os.path.join(folder, 'kspace_abs', 'kspace_abs_' + str(index) + '.nii'))
            save_nii(self.kspace_abs_sv.squeeze().cpu().numpy(), os.path.join(folder, 'kspace_abs_sv', 'kspace_abs_sv_' + str(index) + '.nii'))
            save_nii(self.kspace_abs_pred.squeeze().cpu().numpy(), os.path.join(folder, 'kspace_abs_pred', 'kspace_abs_pred_' + str(index) + '.nii'))

            save_nii(self.recon_rss.squeeze().cpu().numpy(), os.path.join(folder, 'recon_rss', 'recon_rss_' + str(index) + '.nii'))
            save_nii(self.recon_rss_sv.squeeze().cpu().numpy(), os.path.join(folder, 'recon_rss_sv', 'recon_rss_sv_' + str(index) + '.nii'))
            save_nii(self.recon_rss_pred.squeeze().cpu().numpy(), os.path.join(folder, 'recon_rss_pred', 'recon_rss_pred_' + str(index) + '.nii'))




