import random
import numpy as np
import torch
import torchvision.utils as utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_patch_util import *
import os
import h5py
from utils.Utility import *

# (1) Training dataset
class LVTrain(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.data_dir = os.path.join(self.root, 'train_480')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_kspace_all = []
        self.vol_kspace_sv_all = []
        self.vol_recon_origin_all = []
        self.vol_recon_origin_sv_all = []
        self.vol_mask_all = []
        self.vol_recon_rss_all = []
        self.vol_recon_rss_sv_all = []

        # Load all images
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_kspace = f['kspace'][...].transpose(0,2,1)
                vol_kspace_sv = f['kspace_sv'][...].transpose(0,2,1)
                vol_recon_origin = f['recon_origin'][...].transpose(0,2,1)
                vol_recon_origin_sv = f['recon_origin_sv'][...].transpose(0,2,1)  # [32, 384, 768]
                vol_mask = f['mask'][...].transpose(0,2,1)  # [32, 384, 768]
                vol_recon_rss = f['recon_rss'][...].transpose(1,0)
                vol_recon_rss_sv = f['recon_rss_sv'][...].transpose(1,0)  # [384, 768]

            # Patches
            X_patches = vol_kspace
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_kspace_all.append(X_patches)

            X_patches = vol_kspace_sv
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_kspace_sv_all.append(X_patches)

            X_patches = vol_recon_origin
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_recon_origin_all.append(X_patches)

            X_patches = vol_recon_origin_sv
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_recon_origin_sv_all.append(X_patches)

            X_patches = vol_mask
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_mask_all.append(X_patches)

            X_patches = vol_recon_rss
            X_patches = X_patches[np.newaxis, np.newaxis, :, :]  # (1, 1, 384, 768)
            self.vol_recon_rss_all.append(X_patches)

            X_patches = vol_recon_rss_sv
            X_patches = X_patches[np.newaxis, np.newaxis, :, :]  # (1, 1, 384, 768)
            self.vol_recon_rss_sv_all.append(X_patches)

        self.vol_kspace_all = np.concatenate(self.vol_kspace_all, 0)
        self.vol_kspace_sv_all = np.concatenate(self.vol_kspace_sv_all, 0)
        self.vol_recon_origin_all = np.concatenate(self.vol_recon_origin_all, 0)
        self.vol_recon_origin_sv_all = np.concatenate(self.vol_recon_origin_sv_all, 0)  # (B, 32, 384, 768)
        self.vol_mask_all = np.concatenate(self.vol_mask_all, 0)  # (B, 32, 384, 768)
        self.vol_recon_rss_all = np.concatenate(self.vol_recon_rss_all, 0)
        self.vol_recon_rss_sv_all = np.concatenate(self.vol_recon_rss_sv_all, 0)  # (B, 1, 384, 768)

    def __getitem__(self, index):
        vol_kspace = self.vol_kspace_all[index, ...]
        vol_kspace_sv = self.vol_kspace_sv_all[index, ...]
        vol_recon_origin = self.vol_recon_origin_all[index, ...]
        vol_recon_origin_sv = self.vol_recon_origin_sv_all[index, ...]
        vol_mask = self.vol_mask_all[index, ...]
        vol_recon_rss = self.vol_recon_rss_all[index, ...]
        vol_recon_rss_sv = self.vol_recon_rss_sv_all[index, ...]

        vol_kspace = torch.from_numpy(vol_kspace.copy())
        vol_kspace_sv = torch.from_numpy(vol_kspace_sv.copy())
        vol_recon_origin = torch.from_numpy(vol_recon_origin.copy())
        vol_recon_origin_sv = torch.from_numpy(vol_recon_origin_sv.copy())
        vol_mask = torch.from_numpy(vol_mask.copy())
        vol_recon_rss = torch.from_numpy(vol_recon_rss.copy())
        vol_recon_rss_sv = torch.from_numpy(vol_recon_rss_sv.copy())

        return {'kspace': vol_kspace,
                'kspace_sv': vol_kspace_sv,
                'recon_origin': vol_recon_origin,
                'recon_origin_sv': vol_recon_origin_sv,
                'mask': vol_mask,
                'recon_rss': vol_recon_rss,
                'recon_rss_sv': vol_recon_rss_sv}

    def __len__(self):
        return len(self.data_files)


# (2) Validation dataset
class LVVal(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.data_dir = os.path.join(self.root, 'valid')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_kspace_all = []
        self.vol_kspace_sv_all = []
        self.vol_recon_origin_all = []
        self.vol_recon_origin_sv_all = []
        self.vol_mask_all = []
        self.vol_recon_rss_all = []
        self.vol_recon_rss_sv_all = []

        # Load all images
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_kspace = f['kspace'][...].transpose(0, 2, 1)
                vol_kspace_sv = f['kspace_sv'][...].transpose(0, 2, 1)
                vol_recon_origin = f['recon_origin'][...].transpose(0, 2, 1)
                vol_recon_origin_sv = f['recon_origin_sv'][...].transpose(0, 2, 1)  # [32, 384, 768]
                vol_mask = f['mask'][...].transpose(0,2,1)  # [32, 384, 768]
                vol_recon_rss = f['recon_rss'][...].transpose(1, 0)
                vol_recon_rss_sv = f['recon_rss_sv'][...].transpose(1, 0)  # [384, 768]

            # Patches
            X_patches = vol_kspace
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_kspace_all.append(X_patches)

            X_patches = vol_kspace_sv
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_kspace_sv_all.append(X_patches)

            X_patches = vol_recon_origin
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_recon_origin_all.append(X_patches)

            X_patches = vol_recon_origin_sv
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_recon_origin_sv_all.append(X_patches)

            X_patches = vol_mask
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_mask_all.append(X_patches)

            X_patches = vol_recon_rss
            X_patches = X_patches[np.newaxis, np.newaxis, :, :]  # (1, 1, 384, 768)
            self.vol_recon_rss_all.append(X_patches)

            X_patches = vol_recon_rss_sv
            X_patches = X_patches[np.newaxis, np.newaxis, :, :]  # (1, 1, 384, 768)
            self.vol_recon_rss_sv_all.append(X_patches)

        self.vol_kspace_all = np.concatenate(self.vol_kspace_all, 0)
        self.vol_kspace_sv_all = np.concatenate(self.vol_kspace_sv_all, 0)
        self.vol_recon_origin_all = np.concatenate(self.vol_recon_origin_all, 0)
        self.vol_recon_origin_sv_all = np.concatenate(self.vol_recon_origin_sv_all, 0)  # (3, 32, 384, 768)
        self.vol_mask_all = np.concatenate(self.vol_mask_all, 0)  # (B, 32, 384, 768)
        self.vol_recon_rss_all = np.concatenate(self.vol_recon_rss_all, 0)
        self.vol_recon_rss_sv_all = np.concatenate(self.vol_recon_rss_sv_all, 0)  # (3, 1, 384, 768)

    def __getitem__(self, index):
        vol_kspace = self.vol_kspace_all[index, ...]
        vol_kspace_sv = self.vol_kspace_sv_all[index, ...]
        vol_recon_origin = self.vol_recon_origin_all[index, ...]
        vol_recon_origin_sv = self.vol_recon_origin_sv_all[index, ...]
        vol_mask = self.vol_mask_all[index, ...]
        vol_recon_rss = self.vol_recon_rss_all[index, ...]
        vol_recon_rss_sv = self.vol_recon_rss_sv_all[index, ...]

        vol_kspace = torch.from_numpy(vol_kspace.copy())
        vol_kspace_sv = torch.from_numpy(vol_kspace_sv.copy())
        vol_recon_origin = torch.from_numpy(vol_recon_origin.copy())
        vol_recon_origin_sv = torch.from_numpy(vol_recon_origin_sv.copy())
        vol_mask = torch.from_numpy(vol_mask.copy())
        vol_recon_rss = torch.from_numpy(vol_recon_rss.copy())
        vol_recon_rss_sv = torch.from_numpy(vol_recon_rss_sv.copy())

        return {'kspace': vol_kspace,
                'kspace_sv': vol_kspace_sv,
                'recon_origin': vol_recon_origin,
                'recon_origin_sv': vol_recon_origin_sv,
                'mask': vol_mask,
                'recon_rss': vol_recon_rss,
                'recon_rss_sv': vol_recon_rss_sv}

    def __len__(self):
        return len(self.data_files)


# (3) Testing dataset
class LVTest(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.data_dir = os.path.join(self.root, 'test')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_kspace_all = []
        self.vol_kspace_sv_all = []
        self.vol_recon_origin_all = []
        self.vol_recon_origin_sv_all = []
        self.vol_mask_all = []
        self.vol_recon_rss_all = []
        self.vol_recon_rss_sv_all = []

        # Load all images
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_kspace = f['kspace'][...].transpose(0, 2, 1)
                vol_kspace_sv = f['kspace_sv'][...].transpose(0, 2, 1)
                vol_recon_origin = f['recon_origin'][...].transpose(0, 2, 1)
                vol_recon_origin_sv = f['recon_origin_sv'][...].transpose(0, 2, 1)  # [32, 384, 768]
                vol_mask = f['mask'][...].transpose(0,2,1)  # [32, 384, 768]
                vol_recon_rss = f['recon_rss'][...].transpose(1, 0)
                vol_recon_rss_sv = f['recon_rss_sv'][...].transpose(1, 0)  # [384, 768]

            # Patches
            X_patches = vol_kspace
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_kspace_all.append(X_patches)

            X_patches = vol_kspace_sv
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_kspace_sv_all.append(X_patches)

            X_patches = vol_recon_origin
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_recon_origin_all.append(X_patches)

            X_patches = vol_recon_origin_sv
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_recon_origin_sv_all.append(X_patches)

            X_patches = vol_mask
            X_patches = X_patches[np.newaxis, :, :, :]  # (1, 32, 384, 768)
            self.vol_mask_all.append(X_patches)

            X_patches = vol_recon_rss
            X_patches = X_patches[np.newaxis, np.newaxis, :, :]  # (1, 1, 384, 768)
            self.vol_recon_rss_all.append(X_patches)

            X_patches = vol_recon_rss_sv
            X_patches = X_patches[np.newaxis, np.newaxis, :, :]  # (1, 1, 384, 768)
            self.vol_recon_rss_sv_all.append(X_patches)

        self.vol_kspace_all = np.concatenate(self.vol_kspace_all, 0)
        self.vol_kspace_sv_all = np.concatenate(self.vol_kspace_sv_all, 0)
        self.vol_recon_origin_all = np.concatenate(self.vol_recon_origin_all, 0)
        self.vol_recon_origin_sv_all = np.concatenate(self.vol_recon_origin_sv_all, 0)  # (3, 32, 384, 768)
        self.vol_mask_all = np.concatenate(self.vol_mask_all, 0)  # (B, 32, 384, 768)
        self.vol_recon_rss_all = np.concatenate(self.vol_recon_rss_all, 0)
        self.vol_recon_rss_sv_all = np.concatenate(self.vol_recon_rss_sv_all, 0)  # (3, 1, 384, 768)

    def __getitem__(self, index):
        vol_kspace = self.vol_kspace_all[index, ...]
        vol_kspace_sv = self.vol_kspace_sv_all[index, ...]
        vol_recon_origin = self.vol_recon_origin_all[index, ...]
        vol_recon_origin_sv = self.vol_recon_origin_sv_all[index, ...]
        vol_mask = self.vol_mask_all[index, ...]
        vol_recon_rss = self.vol_recon_rss_all[index, ...]
        vol_recon_rss_sv = self.vol_recon_rss_sv_all[index, ...]

        vol_kspace = torch.from_numpy(vol_kspace.copy())
        vol_kspace_sv = torch.from_numpy(vol_kspace_sv.copy())
        vol_recon_origin = torch.from_numpy(vol_recon_origin.copy())
        vol_recon_origin_sv = torch.from_numpy(vol_recon_origin_sv.copy())
        vol_mask = torch.from_numpy(vol_mask.copy())
        vol_recon_rss = torch.from_numpy(vol_recon_rss.copy())
        vol_recon_rss_sv = torch.from_numpy(vol_recon_rss_sv.copy())

        return {'kspace': vol_kspace,
                'kspace_sv': vol_kspace_sv,
                'recon_origin': vol_recon_origin,
                'recon_origin_sv': vol_recon_origin_sv,
                'mask': vol_mask,
                'recon_rss': vol_recon_rss,
                'recon_rss_sv': vol_recon_rss_sv}

    def __len__(self):
        return len(self.data_files)


if __name__ == '__main__':
    pass