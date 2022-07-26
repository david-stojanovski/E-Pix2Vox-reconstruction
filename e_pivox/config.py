# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os

from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Dataset Config
#

__C.DATASETS = edict()
__C.DATASETS.HEARTSEG = edict()
__C.DATASETS.HEARTSEG.IMG_ROOT = '/home/e_pivox/datasets/heart_seg/heart_render'
__C.DATASETS.HEARTSEG.TAXONOMY_FILE_PATH = '/home/e_pivox/datasets/HeartSeg.json'
__C.DATASETS.HEARTSEG.RENDERING_PATH = os.path.join(__C.DATASETS.HEARTSEG.IMG_ROOT, '%s/%s/*.png')
__C.DATASETS.HEARTSEG.VOXEL_PATH = '/home/e_pivox/datasets/heart_seg/voxel_volumes/%s/%s/model.npy'


# __C.DATASETS = edict()
__C.DATASETS.SHAPENET = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH = r'/home/e_pivox/datasets_shapenet/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH = r'/home/e_pivox/datasets_shapenet/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH = r'/home/e_pivox/datasets_shapenet/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'
#
# Dataset
#
__C.DATASET = edict()
__C.DATASET.MEAN = [0.5, 0.5, 0.5]
__C.DATASET.STD = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET = 'HeartSeg'
__C.DATASET.TEST_DATASET = 'HeartSeg'
# __C.DATASET.TRAIN_DATASET = 'ShapeNet'
# __C.DATASET.TEST_DATASET = 'ShapeNet'


#
# Common
#
__C.CONST = edict()
__C.CONST.DEVICE = '0'
__C.CONST.RNG_SEED = 0
__C.CONST.IMG_W = 224  # Image width for input
__C.CONST.IMG_H = 224  # Image height for input
__C.CONST.BATCH_SIZE = 8
__C.CONST.N_VIEWS_RENDERING = 9  # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_W = 128  # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H = 128  # Dummy property for Pascal 3D
__C.CONST.NUM_WORKER = 10  # number of data workers
__C.CONST.TEST_SAVE_NUMBER = 125  # number of test cases that should be saved as np arrays each epoch


# Directories
#
__C.DIR = edict()
__C.DIR.OUT_PATH = os.path.join(os.getcwd(), 'output')
__C.DIR.IOU_SAVE_PATH = os.path.join(__C.DIR.OUT_PATH, 'iou_scores.xlsx')
__C.DIR.RANDOM_BG_PATH = '/home/hzxie/Datasets/SUN2012/JPEGImages'

#
# Network
#
__C.NETWORK = edict()
__C.NETWORK.LEAKY_VALUE = .2
__C.NETWORK.TCONV_USE_BIAS = False
__C.NETWORK.LOSS_FUNC = 'BCELoss'
__C.NETWORK.MODEL_SIZE = 64
__C.NETWORK.USE_EP2V = True
__C.NETWORK.USE_REFINER = True
__C.NETWORK.USE_MERGER = True


#
# Training
#
__C.TRAIN = edict()
__C.TRAIN.RESUME_TRAIN = False
__C.TRAIN.NUM_EPOCHS = 250
__C.TRAIN.BRIGHTNESS = .4
__C.TRAIN.CONTRAST = .4
__C.TRAIN.SATURATION = .4
__C.TRAIN.NOISE_STD = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY = 'adam'  # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER = 0
__C.TRAIN.EPOCH_START_USE_MERGER = 0
__C.TRAIN.ENCODER_LEARNING_RATE = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE = 1e-3
__C.TRAIN.MERGER_LEARNING_RATE = 1e-4
__C.TRAIN.ENCODER_LR_MILESTONES = [150]
__C.TRAIN.DECODER_LR_MILESTONES = [150]
__C.TRAIN.REFINER_LR_MILESTONES = [150]
__C.TRAIN.MERGER_LR_MILESTONES = [150]
__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.MOMENTUM = .9
__C.TRAIN.GAMMA = .5
__C.TRAIN.SAVE_FREQ = 50  # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING = False

#
# Testing options
#
__C.TEST = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH = [.2, .3, .4, .5, .6, .7, .8, .9]
__C.TEST.TEST_NETWORK = False
__C.TEST.VOL_OR_RENDER_SAVE = 'volume'


