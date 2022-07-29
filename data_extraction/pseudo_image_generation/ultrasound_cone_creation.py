# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:11:15 2022

@author: david
"""
import numpy as np
from PIL import Image


def resize(im, size, resample=Image.NEAREST):
    """ resize the image to the given number of pixels"""
    if type(im) == np.ndarray:
        im = Image.fromarray(im.astype(np.float32))
    height, width = size
    # pil resize uses opposite convention
    im = im.resize((width, height), resample=resample)
    return im


def crop(im, crop_amount, locs=('top', 'bottom', 'left', 'right')):
    """ crop function
    """

    if crop_amount <= 0:
        return im
    if 'top' in locs:
        im = im[crop_amount:, :]  # remove this padding
    if 'bottom' in locs:
        im = im[:-crop_amount, :]  # remove this padding
    if 'left' in locs:
        im = im[:, crop_amount:]  # remove this padding
    if 'right' in locs:
        im = im[:, :-crop_amount]  # remove this padding
    return im


def crop_to_mask(image, mask):
    """ crop to the boundaries of the mask. mask should be bool (will be cast to bool) """
    mask = mask.astype(bool)
    min_r = np.where(mask == 1)[0].min()
    image = crop(image, min_r, locs=("top",))
    max_r = np.where(mask == 1)[0].max() + 1
    image = crop(image, image.shape[0] - max_r, locs=("bottom",))
    min_c = np.where(mask == 1)[1].min()
    image = crop(image, min_c, locs=("left",))
    max_c = np.where(mask == 1)[1].max() + 1
    image = crop(image, image.shape[1] - max_c, locs=("right",))
    return image


def get_circle_mask(xx, yy, origin, radius):
    """ Gets a circular mask at the specified origin and radius.
     xx and yy define the coordinate grid for the mask so origin should be in reference to these two.
     """
    mask = np.zeros(xx.shape)
    mask[np.sqrt((yy - origin[0]) ** 2 + (xx - origin[1]) ** 2) < radius] = 1
    return mask


def get_angle_mask(xx, yy, origin, width, tilt):
    """ Gets a triangular mask of the specified origin, width, and tilt.
    xx and yy provide the grid for the mask and the parameters should be listed in reference to this.
    """
    mask = np.zeros(xx.shape)
    half_width = width / 2
    lower_bound = -np.pi / 2 - half_width + tilt
    upper_bound = -np.pi / 2 + half_width + tilt
    mask[(np.arctan2(origin[0] - yy, xx - origin[1]) < upper_bound) &
         (np.arctan2(origin[0] - yy, xx - origin[1]) > lower_bound)] = 1
    return mask


def get_full_mask(inp_size, origin, radius, width, tilt, ax=None):
    """ calls both circle mask and angle mask to get a mask of an ultrasound cone.
    if ax is not None than the mask will be plotted.
    returns a mask with 0 being the region outside the cone and 1 the region inside it
    """
    assert radius <= inp_size, "radius should be smaller than image"
    max_side_pt = max([radius * np.sin(width / 2 + tilt), radius * np.sin(-width / 2 - tilt)])
    diff = max_side_pt - inp_size / 2
    if diff > 0:
        inp_size = int(np.ceil(max_side_pt * 2))
        origin[1] += int(diff)
    xx, yy = np.meshgrid(range(inp_size), range(inp_size))
    circle_mask = get_circle_mask(xx, yy, origin, radius)
    angle_mask = get_angle_mask(xx, yy, origin, width, tilt)
    full_mask = circle_mask + angle_mask
    if ax is not None:
        ax.imshow(full_mask)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
    full_mask /= 2
    full_mask = full_mask.astype(int)
    return full_mask


def make_us_cone(cfg):
    """Main function to call sub fuctions for creating a cone, as specified in the config.py file.

    Args:
        cfg (easydict.EasyDict): Configuration file.

    Returns:
         numpy.ndarray, params (dict): Binary mask array and a dictionary containing the creation parameters.
    """
    inp_size = cfg.PARAMS.OUT_IMG_SIZE[0]
    radii = [cfg.PARAMS.CONE_RADII]
    widths = [cfg.PARAMS.CONE_WIDTHS]
    tilts = np.zeros(1)
    origin_ys = np.ones(shape=(1,))
    origin_xs = inp_size / 2 * np.ones(shape=(1,))

    for radius, width, tilt, origin_y, origin_x in zip(radii, widths, tilts, origin_ys, origin_xs):
        params = dict(radius=radius, width=width, tilt=tilt, origin_x=origin_x, origin_y=origin_y)
        mask = get_full_mask(inp_size, [origin_y, origin_x], radius, width, tilt)
        mask = crop_to_mask(mask, mask)
        mask = resize(mask, (inp_size, inp_size))

    return np.array(mask), params
