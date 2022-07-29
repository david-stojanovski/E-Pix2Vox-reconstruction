import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATA = edict()
__C.DATA.DATA_PATH = '/path/to/your/folders/containing/heart/images'
__C.DATA.JSON_VIEW_DICT_PATH = r'/path/to/your/folders/containing/pseudo_image_generation/view_param_dict.json'
__C.DATA.SAVE_PATH = r'/path/to/your/folders/containing/pseudo_image_generation/heart_out'

__C.PARAMS = edict()
__C.PARAMS.OUT_IMG_SIZE = (224, 224)  # Image save size
__C.PARAMS.BLOOD_CONTRAST_RATIO = 0.6  # Contrast ratio of blood pool to tissue
__C.PARAMS.CONE_WIDTHS = 6 * np.pi / 16  # Width of the generated ultrasound cone
__C.PARAMS.CONE_RADII = 1.0 * __C.PARAMS.OUT_IMG_SIZE[0]  # Radius of generated ultrasound cone
__C.PARAMS.SHOW_BLOODPOOL = False  # bool flag of whether to plot a debug image of the found blood pools

__C.NOISE = edict()
__C.NOISE.DOWNSIZE_FACTOR_1 = 1  # Downsize factor of the 1st multiplicative noise addition
__C.NOISE.TYPE_1 = 'uniform'  # Statistical distribution of the 1st multiplicative noise addition
__C.NOISE.DOWNSIZE_FACTOR_2 = 2  # Downsize factor of the 2nd multiplicative noise addition
__C.NOISE.TYPE_2 = 'normal'  # Statistical distribution of the 2nd multiplicative noise addition
__C.NOISE.DOWNSIZE_FACTOR_3 = 6  # Downsize factor of the 3rd multiplicative noise addition
__C.NOISE.TYPE_3 = 'normal'  # Statistical distribution of the 3rd multiplicative noise addition

__C.BLUR = edict()
__C.BLUR.KSIZE_1 = (0, 0)  # Kernel size of the 1st Gaussian blur operation
__C.BLUR.SIGMAXY_1 = (2, 4)  # Sigma sizes of the 1st Gaussian blur operation
__C.BLUR.KSIZE_2 = (0, 0)  # Kernel size of the 2nd Gaussian blur operation
__C.BLUR.SIGMAXY_2 = (2, 6)  # Sigma sizes of the 2nd Gaussian blur operation

"""
Below are the parameters used in:
https://github.com/adgilbert/pseudo-image-extraction/tree/14cd41a9f1da7810376deda174e555f0d052a957
"""
__C.PARAMS.LOC_MIN = (100, 0)
__C.PARAMS.LOC_MEAN = (400, 600)
__C.PARAMS.LOC_STD = (100, 100)
__C.PARAMS.BRIGHTNESS_MEAN = 0.3
__C.PARAMS.BRIGHTNESS_STD = 0.05
__C.PARAMS.BRIGHTNESS_PROB = 0.5
__C.PARAMS.SIZE_MEAN = 4000
__C.PARAMS.SIZE_STD = 500
__C.PARAMS.ROW_COL_RATIO = 1
