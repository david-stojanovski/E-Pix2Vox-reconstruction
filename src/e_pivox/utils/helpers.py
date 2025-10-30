#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os
from pathlib import Path
from datetime import datetime as dt

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import utils.network_losses as net_loss


def var_or_cuda(x: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m: torch.nn.Module) -> None:
    if (
        torch.nn.Conv2d is type(m)
        or torch.nn.Conv3d is type(m)
        or torch.nn.ConvTranspose2d is type(m)
        or torch.nn.ConvTranspose3d is type(m)
    ):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif torch.nn.BatchNorm2d is type(m) or torch.nn.BatchNorm3d is type(m):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif torch.nn.Linear is type(m):
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_volume_views(volume: torch.Tensor) -> np.ndarray:
    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect("equal")
    ax.voxels(volume, edgecolor="k")

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape((*fig.canvas.get_width_height()[::-1], 3))
    return img


def save_test_volumes_as_np(cfg, volume, sample_id, epoch_num) -> None:
    img_dir = cfg.DIR.OUT_PATH / "images"
    test_case_path = img_dir / "test"
    save_path = test_case_path / (str(sample_id) + os.sep)

    if not Path(img_dir).exists():
        img_dir.mkdir(parents=True, exist_ok=True)
    if not Path(test_case_path).exists():
        test_case_path.mkdir(parents=True, exist_ok=True)
    if not Path(save_path).exists():
        save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / ("epoch_" + str(epoch_num)), volume.cpu().numpy())


def get_loss_function(cfg):
    if cfg.NETWORK.LOSS_FUNC.lower() == "bceloss":
        loss_func = torch.nn.BCELoss()
    elif cfg.NETWORK.LOSS_FUNC.lower() == "iou":
        loss_func = net_loss.IoULoss()
    elif cfg.NETWORK.LOSS_FUNC.lower() == "focalloss":
        loss_func = net_loss.FocalLoss()
    elif cfg.NETWORK.LOSS_FUNC.lower() == "tverskyloss":
        loss_func = net_loss.TverskyLoss()
    elif cfg.NETWORK.LOSS_FUNC.lower() == "focaltverskyloss":
        loss_func = net_loss.FocalTverskyLoss()
    else:
        msg = f"[FATAL] {dt.now()} No matching loss function available for: {cfg.NETWORK.LOSS_FUNC}. voxels"
        raise Exception(
            msg
        )
    return loss_func


def model_size_importer(cfg) -> None:
    if cfg.NETWORK.MODEL_SIZE in {32, 64}:
        pass
    else:
        msg = f"[FATAL] {dt.now()} No model available for size: {cfg.NETWORK.MODEL_SIZE}. voxels"
        raise Exception(msg)
