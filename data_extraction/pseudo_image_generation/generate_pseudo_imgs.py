import glob
import json
import os
from random import uniform

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from natsort import natsorted
from skimage.transform import rotate

import image_operations as image_ops
import ultrasound_cone_creation as ucc
from config import cfg


def load_view_dictionary(load_path):
    """Loads the json file containing the augmentation parameters for each image.

    Args:
        load_path (str): Path to json file for loading.

    Returns:
        data (dict): Loaded image augmentation parameters.
    """
    with open(load_path) as f:
        data = json.load(f)
    return data


def zoom_shift_crop(view_dictionary, in_image, random_vars_in):
    """Perform zooming, shifting and cropping on an image.

    Args:
        view_dictionary (dict): Dictionary containing parameters for augmenting images.
        in_image (numpy.ndarray): Input image to apply operations to.
        random_vars_in (numpy.ndarray): Parameter variables for applying operations.

    Returns:
        shifted_img (numpy.ndarray): Returned image with operations applied.
    """
    width = int(in_image.shape[1] * (random_vars_in['zoom_factor'] * 100) / 100)
    height = int(in_image.shape[0] * (random_vars_in['zoom_factor'] * 100) / 100)
    dim = (width, height)
    image_zoom = np.array(Image.fromarray(in_image * 255.).resize(dim)) / 255.
    shifted_img = shift_2d_replace(image_zoom, random_vars_in['shift_x'], random_vars_in['shift_y'], constant=0)
    if view_dictionary['flip_x']:
        shifted_img = np.fliplr(shifted_img)
    if view_dictionary['flip_y']:
        shifted_img = np.flipud(shifted_img)

    return shifted_img


def save_img2file(img4saving, save_folder_path, save_folder_name, save_view_name, cone_mask=None):
    """Function that saves images to the correct paths."""
    save_folder_dir = os.sep.join(list(save_folder_path.split(os.sep)[0:-1]))
    case_name = save_folder_path.split(os.sep)[-1]

    if not os.path.isdir(os.path.join(save_folder_dir, save_folder_name, case_name)):
        os.makedirs(os.path.join(save_folder_dir, save_folder_name, case_name))
    save_path = os.path.join(save_folder_dir, save_folder_name, case_name, save_view_name)
    if cone_mask is not None:
        out_img = Image.fromarray(apply_cone(img4saving, cone_mask) * 255).convert("L")
        out_img.save(save_path)
    else:
        out_img = Image.fromarray(normalize_data(img4saving) * 255).convert("L")
        out_img.save(save_path)
    return


def shift_2d_replace(data, dx, dy, constant=0):
    """Given binary image shift it by dx, dy pixels. It fills the newly created area with zeros.

    Args:
        data (numpy.ndarray): Input image to be shifted.
        dx (int): Number of pixels to shift in x direction.
        dy (int): Number of pixels to shift in y direction.
        constant (int): Value to fill newly created pixels (on the lagging side).

    Returns:
        shifted_data (numpy.ndarray): Shifted image.
    """
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


def get_cropped_from_zoomed_img(in_image, in_cone):
    if type(in_image) == Image.Image:
        in_image = np.array(in_image)
    cone_center = [0, in_image.shape[1] / 2]
    crop4cone_x = [0, in_cone.shape[0]]
    crop4cone_y = [cone_center[1] - in_cone.shape[1] / 2, cone_center[1] + in_cone.shape[1] / 2]

    return in_image[crop4cone_x[0]:crop4cone_x[1], int(crop4cone_y[0]): int(crop4cone_y[1])]


def apply_cone(in_image, in_cone):
    cropped_img = get_cropped_from_zoomed_img(in_image, in_cone)
    coned_img = cropped_img * in_cone
    return coned_img


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def padding(array, xx, yy):
    """Pads an input array to a desired dimension (xx, yy).

    Args:
        array (numpy.ndarray): Input array to be padded.
        xx (int): Desired x dimension.
        yy (int): Desired y dimension.

    Returns:
        numpy.ndarray: Output padded image with dimensions (xx, yy)
    """
    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def find_bloodpools(cfg, in_img, rotation_angle):
    """Given a binary 2D image slice of a heart, find the blood pools. This function effectively finds enclosed contours
     and selects these as blood pools. The contours must be closed to function properly.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        in_img (numpy.ndarray): Input image containing 2D binary slice of the heart.
        rotation_angle (numpy.float): Rotation angle of the original image, so that the image is orientated correctly.

    Returns:
        out_bloodpool_mask (numpy.ndarray): Binary array representing the found bloodpools.
    """
    in_img_copy = in_img.copy().astype('uint8')
    cnt = cv2.findContours(in_img_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    all_blood_pools = []
    for ii in range(len(cnt[0]) - 1):
        single_blood_pool = cv2.fillPoly(in_img_copy, pts=[cnt[0][ii]], color=(255, 255, 255))
        all_blood_pools.append(single_blood_pool)
        in_img_copy = in_img.copy().astype('uint8')

    out_bloodpool_mask = sum(all_blood_pools) / 255
    out_bloodpool_mask = rotate(out_bloodpool_mask, rotation_angle).astype(int)

    if cfg.PARAMS.SHOW_BLOODPOOL:
        plt.figure()
        plt.imshow(out_bloodpool_mask, cmap='gray')

    return out_bloodpool_mask


def get_randomised_variables(cfg, view_augmentation_dict, view_name, dims):
    """Given the ranges for each variable in the configuration file, this function returns randomly selected parameters.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        view_augmentation_dict (dict): Dictionary containing parameters for augmenting images.
        view_name (str): Name of the image to extract variables for.
        dims (tuple): Dimensions of the image that is being processed.

    Returns:
        random_vars_out (dict), op (obj): Dictionary containing randomly assigned variables and an op representing the
        randomly selected choice of brightening or darkening.
    """

    random_rot_angle = uniform(view_augmentation_dict[view_name]['rot_angle'][0],
                               view_augmentation_dict[view_name]['rot_angle'][1])
    random_zoom_factor = uniform(view_augmentation_dict[view_name]['zoom_factor'][0],
                                 view_augmentation_dict[view_name]['zoom_factor'][1])
    random_shift_x = int(uniform(view_augmentation_dict[view_name]['shift_x'][0],
                                 view_augmentation_dict[view_name]['shift_x'][1]))
    random_shift_y = int(uniform(view_augmentation_dict[view_name]['shift_y'][0],
                                 view_augmentation_dict[view_name]['shift_y'][1]))

    op = np.random.choice([image_ops.brightening, image_ops.shadowing],
                          p=[cfg.PARAMS.BRIGHTNESS_PROB, 1 - cfg.PARAMS.BRIGHTNESS_PROB])

    loc_row = min(dims[0], max(cfg.PARAMS.LOC_MIN[0], np.random.normal(cfg.PARAMS.LOC_MIN[0], cfg.PARAMS.LOC_STD[0])))
    loc_col = min(dims[1], max(cfg.PARAMS.LOC_MIN[1], np.random.normal(cfg.PARAMS.LOC_MIN[1], cfg.PARAMS.LOC_STD[1])))
    brightness = min(1, max(0, np.random.normal(cfg.PARAMS.BRIGHTNESS_MEAN, cfg.PARAMS.BRIGHTNESS_STD)))
    sigma = max(1, 4 * dims[0] + np.random.normal(cfg.PARAMS.SIZE_MEAN, cfg.PARAMS.SIZE_STD))

    random_vars_out = dict()
    random_vars_out['rot_angle'] = random_rot_angle
    random_vars_out['zoom_factor'] = random_zoom_factor
    random_vars_out['shift_x'] = random_shift_x
    random_vars_out['shift_y'] = random_shift_y
    random_vars_out['loc_row'] = loc_row
    random_vars_out['loc_col'] = loc_col
    random_vars_out['brightness'] = brightness
    random_vars_out['sigma'] = sigma
    return random_vars_out, op


def get_img_data(in_img_path):
    """Loads image from path, converts it to be binary and also gets the image name.

    Args:
        in_img_path (str): Path for image  to be loaded.

    Returns:
        out_image (numpy.ndarray), out_view_name (str): Loaded image and the corresponding name of the image.
    """
    out_image = cv2.imread(in_img_path, 0) / 255.
    out_image = ((out_image > 0).astype(bool))
    out_view_name = in_img_path.split(os.sep)[-1]
    return out_image, out_view_name


def convert_seg2pseudo(cfg, patient_folder_path, save_folder_path, in_cone=None):
    """
    Function to call relevant steps in converting a segmentation like (binary) image of a slice of the heart to a
    pseudo image, in preparation for training/testing a CycleGAN network.
    """

    view_aug_dict = load_view_dictionary(cfg.DATA.JSON_VIEW_DICT_PATH)
    img_paths = natsorted(glob.glob(os.path.join(patient_folder_path, '*.png')))

    for img_path in img_paths:
        image, view_name = get_img_data(img_path)
        random_vars, operation = get_randomised_variables(cfg, view_aug_dict, view_name, image.shape)

        bloodpool_mask = find_bloodpools(cfg, image, random_vars['rot_angle'])
        rotated_img = rotate(image, random_vars['rot_angle']).astype(int)
        pseudo = rotated_img.astype(int)

        pseudo = zoom_shift_crop(view_aug_dict[view_name], pseudo, random_vars)
        rotated_img = zoom_shift_crop(view_aug_dict[view_name], rotated_img, random_vars)
        bloodpool_mask = zoom_shift_crop(view_aug_dict[view_name], bloodpool_mask, random_vars)

        pseudo = image_ops.add_multiplicative_noise(pseudo)
        pseudo = np.array(operation(pseudo,
                                    (random_vars['loc_row'], random_vars['loc_col']),
                                    random_vars['brightness'],
                                    random_vars['sigma'],
                                    cfg.PARAMS.ROW_COL_RATIO))

        pseudo = image_ops.add_additive_noise(pseudo, cfg.NOISE.DOWNSIZE_FACTOR_1, cfg.NOISE.TYPE_1)
        pseudo[np.where(bloodpool_mask == 1)] *= cfg.PARAMS.BLOOD_CONTRAST_RATIO
        pseudo = cv2.GaussianBlur(pseudo, cfg.BLUR.KSIZE_1, cfg.BLUR.SIGMAXY_1[0], cfg.BLUR.SIGMAXY_1[1])
        pseudo = cv2.GaussianBlur(pseudo, cfg.BLUR.KSIZE_2, cfg.BLUR.SIGMAXY_2[0], cfg.BLUR.SIGMAXY_2[1])
        pseudo = image_ops.add_additive_noise(pseudo, cfg.NOISE.DOWNSIZE_FACTOR_2, cfg.NOISE.TYPE_2)
        pseudo = image_ops.add_additive_noise(pseudo, cfg.NOISE.DOWNSIZE_FACTOR_3, cfg.NOISE.TYPE_3)
        pseudo = image_ops.set_random_max(pseudo)

        if in_cone is None:
            in_cone, params = ucc.make_us_cone(cfg)

        pseudo = apply_cone(pseudo, in_cone)
        pseudo = image_ops.resize(pseudo, cfg.PARAMS.OUT_IMG_SIZE, Image.Resampling.BILINEAR)

        cropped_pseudo = padding(np.array(pseudo), cfg.PARAMS.OUT_IMG_SIZE[0], cfg.PARAMS.OUT_IMG_SIZE[1])

        save_img2file(cropped_pseudo, save_folder_path, 'pseudo', view_name, cone_mask=None)
        save_img2file(rotated_img, save_folder_path, 'segmentation', view_name, cone_mask=in_cone)


def main(cfg):
    data_folder = cfg.DATA.DATA_PATH
    save_folder = cfg.DATA.SAVE_PATH
    data_folders = natsorted(glob.glob(os.path.join(data_folder, '*')))
    if len(data_folders) == 0:
        print('no images found')
    cone, __ = ucc.make_us_cone(cfg)

    for folder in data_folders:
        case_folder = folder.split(os.sep)[-1]
        convert_seg2pseudo(cfg, folder, os.path.join(save_folder, case_folder), in_cone=cone)
        print(folder)
    return


if __name__ == '__main__':
    main(cfg)
