# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
import utils.network_losses as net_loss
from datetime import datetime as dt


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
            type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_volume_views(volume):
    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def save_test_volumes_as_np(cfg, volume, sample_id, epoch_num):
    img_dir = os.path.join(cfg.DIR.OUT_PATH, 'images')
    test_case_path = os.path.join(img_dir, 'test')
    save_path = os.path.join(test_case_path, str(sample_id) + os.sep)

    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if not os.path.isdir(test_case_path):
        os.mkdir(test_case_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    np.save(save_path + 'epoch_' + str(epoch_num), volume.cpu().numpy())


def get_loss_function(cfg):
    if cfg.NETWORK.LOSS_FUNC.lower() == 'bceloss':
        loss_func = torch.nn.BCELoss()
    elif cfg.NETWORK.LOSS_FUNC.lower() == 'iou':
        loss_func = net_loss.IoULoss()
    elif cfg.NETWORK.LOSS_FUNC.lower() == 'focalloss':
        loss_func = net_loss.FocalLoss()
    elif cfg.NETWORK.LOSS_FUNC.lower() == 'tverskyloss':
        loss_func = net_loss.TverskyLoss()
    elif cfg.NETWORK.LOSS_FUNC.lower() == 'focaltverskyloss':
        loss_func = net_loss.FocalTverskyLoss()
    else:
        raise Exception('[FATAL] %s No matching loss function available for: %s. voxels' % (dt.now(), cfg.NETWORK.LOSS_FUNC))
    return loss_func

def model_size_importer(cfg):
    if cfg.NETWORK.MODEL_SIZE == 32:
        from models.decoder_32 import Decoder
        from models.encoder_32 import Encoder
        from models.merger_32 import Merger
        from models.refiner_32 import Refiner
    elif cfg.NETWORK.MODEL_SIZE == 64:
        from models.decoder_64 import Decoder
        from models.encoder_64 import Encoder
        from models.merger_64 import Merger
        from models.refiner_64 import Refiner
    else:
        raise Exception('[FATAL] %s No model available for size: %s. voxels' % (dt.now(), cfg.NETWORK.MODEL_SIZE))