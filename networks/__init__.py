import torch.nn as nn
import numpy as np
from utils import arange
from networks.networks import *


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network


def get_generator(name, opts):
    if name == 'UNet':
        network = UNet(in_channels=opts.n_channels, out_channels=opts.n_channels, residual=False, depth=opts.UNet_depth, wf=opts.UNet_filters, norm = opts.norm)
    elif name == 'ResUNet':
        network = ResUNet(in_channels=opts.n_channels, out_channels=opts.n_channels, n_filters=opts.n_filters, n_denselayer=opts.n_denselayer, growth_rate=opts.growth_rate, norm = opts.norm)
    elif name == 'DuRDN':
        network =   DuRDN(in_channels=opts.n_channels, out_channels=opts.n_channels, n_filters=opts.n_filters, n_denselayer=opts.n_denselayer, growth_rate=opts.growth_rate, norm = opts.norm)
    else:
        raise NotImplementedError

    # num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    # print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)

