"""Geometry functions that calculate geometries of planes and volumes."""
import math
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy.stats import multivariate_normal


def rotation_matrix(axis, theta):
    """Returns rotation matrix (Euler-Rodrigues formula).

    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    :param axis: numpy array or tuple in R3, specifying the axis of rotation
    :param theta: floating point angle in radians
    :return: rotation matrix as described above
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def apply_single_rotation(bbox, rot):
    """Return rotation of 'plane' by 'theta' radians, with respect to 'axis' rotated around 'center.

    :param bbox: dictionary specifying the reference geometry that will be rotated (2D or 3D).
                 'origin' (required key) is both translated and rotated (handling the center of rotation)
                 All other keys are only rotated.
    :param rot: dictionary specifying the rotation. Keys: 'axis', 'theta', and 'center')
    :return: a new plane on the same format as input, rotated as described.
    """
    rot_matrix = rotation_matrix(rot['axis'], rot['theta'])
    # Extract origin:
    origin = bbox.pop('origin')
    # Translate origin wrt center of rotation and apply rotation:
    origin_relative = (origin - rot['center'])
    origin_rotated = np.dot(rot_matrix, origin_relative)
    # Translating back:
    new_origin = origin_rotated + rot['center']
    # Apply the rotation to all direction vectors:
    plane_rot = {name: np.dot(rot_matrix, bbox[name]) for name, vector in bbox.items()}
    plane_rot['origin'] = new_origin
    return plane_rot


def apply_rotations(bbox, rotations, gen_type='product'):
    """Generator yielding the specified rotations.

    :param bbox: dictionary specifying the reference geometry that will be rotated (2D or 3D).
                 'origin' (required key) is both translated and rotated (handling the center of rotation)
                 All other keys are only rotated. NB: See 'README.md' for further details.
    :param rotations:
    :param gen_type: 'product' or 'zip'.
        'product' results in the cartesian product of the included rotations,
        'zip' combines the rotations pairwise.
    :return: 'new_geometry' and 'angle_dict' (see above)
    """
    rotation_gen = rotation_generator(rotations, gen_type)

    for rotation_list in rotation_gen:
        # Iteratively apply rotations:
        new_geometry = bbox.copy()
        for rotation in rotation_list:
            new_geometry = apply_single_rotation(new_geometry, rotation)

        angles = [angle['theta'] for angle in rotation_list]
        angle_dict = {rotation['name']: angle for rotation, angle in zip(rotations, angles)}

        # Yield rotated geometry:
        yield new_geometry, angle_dict


def rotation_generator(rotations, gen_type='product'):
    """Generator function yields list of rotations.

    :param rotations: see function 'apply_rotations' for argument syntax
    :param gen_type: 'product' or 'zip', former generates cartesian product of rotations, latter generates the zip
    :return: each iteration yields one set of rotations,
        which when combined results in a single plane
    """
    axes = [rotation['axis'] for rotation in rotations]
    centers = [rotation.get('center', (0, 0, 0)) for rotation in rotations]
    angles_per_axis = [rotation['angles'] for rotation in rotations]

    if gen_type == 'product':
        iterator = product(*angles_per_axis)
    elif gen_type == 'zip':
        assert min([len(a) for a in angles_per_axis]) == max(
            [len(a) for a in angles_per_axis]), "sizes of angle arrays not consistent"
        # print("Warning: currently not checking that size is consistent. Will be implemented.")
        iterator = zip(*angles_per_axis)
    else:
        raise ValueError("Invalid argument to 'gent_type': {}".format(gen_type))

    for angles in iterator:
        # Iterate over the combinations of angles:
        rotation_list = []
        for axis, center, angle in zip(axes, centers, angles):
            # Return the current set of rotations:
            rotation_list.append({'axis': axis,
                                  'center': center,
                                  'theta': angle})

        yield (rotation_list)


def add_multiplicative_noise(img, downsize_factor=8):
    """ add multiplicative uniform noise to an image """
    img = norm_img(img, new_max=255.)
    r, c = img.shape
    noise = np.random.uniform(0, 1, size=(int(r / downsize_factor), int(c / downsize_factor)))
    noise = resize(noise, (r, c))
    img *= noise
    img = norm_img(img, new_max=255.)
    return img


def add_additive_noise(img, downsize_factor=8, noise_type="uniform"):
    """ Add Additive random noise to an image """
    if type(img) == Image.Image:
        img = np.array(img)
    if img.max != 255.:
        img = norm_img(img, new_max=255.)
    r, c = img.shape
    if noise_type == "uniform":
        add_noise = np.random.uniform(0, 100, size=(int(r / downsize_factor), int(c / downsize_factor)))
        sub_noise = np.random.uniform(0, 100, size=(int(r / downsize_factor), int(c / downsize_factor)))
    elif noise_type == "normal":
        add_noise = 10 * np.random.normal(0, 1, size=(int(r / downsize_factor), int(c / downsize_factor)))
        sub_noise = 10 * np.random.normal(0, 1, size=(int(r / downsize_factor), int(c / downsize_factor)))
    else:
        raise ValueError(f"noise type {noise_type} not recognized")
    add_noise = resize(add_noise, (r, c))
    img += add_noise
    sub_noise = resize(sub_noise, (r, c))
    img -= sub_noise
    img = norm_img(img, new_max=255.)
    return img


def set_random_max(img, low=230, high=250):
    """ Give an image a random max value """
    im_max = np.random.uniform(low, high)
    img = norm_img(img, im_max)
    return img


def set_random_min(img, low=5, high=30):
    """ Give an image a random minimum value"""
    im_min = np.random.uniform(low, high)
    factor = (img.max() - im_min) / img.max()
    img *= factor
    img += im_min
    return img


def rotate_image(image, degrees):
    im = Image.fromarray(image)
    return np.array(im.rotate(angle=degrees, expand=False))


def random_pad(image: np.ndarray, pad_pix: int, locs: tuple = ('top', 'bottom', 'left', 'right')) -> np.ndarray:
    """
    pad the image on the given sides.
     By default locs is applied to all sides.
     Pad can be positive or negative
    """
    current_size = image.shape
    if pad_pix < 0:
        image = crop(image, pad_pix, locs)
    elif pad_pix > 0:
        image = pad(image, pad_pix, locs)  # location is handled inside padding function
    image = resize(image, current_size)
    return image


def pad(im, pad_amount=64, locs=('top', 'bottom', 'left', 'right')):
    if type(im) == np.ndarray:
        im = Image.fromarray(im)
    im = np.array(ImageOps.expand(im, pad_amount))
    if 'top' not in locs:
        im = im[pad_amount:, :]  # remove this padding
    if 'bottom' not in locs:
        im = im[:-pad_amount, :]  # remove this padding
    if 'left' not in locs:
        im = im[:, pad_amount:]  # remove this padding
    if 'right' not in locs:
        im = im[:, :-pad_amount]  # remove this padding
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


def crop(im, crop_amount, locs=('top', 'bottom', 'left', 'right')):
    """ crop function
    """
    if type(im) == Image.Image:
        im = np.array(im)
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


def resize(im, size, resample=Image.NEAREST):
    """ resize the image to the given number of pixels"""
    if type(im) == np.ndarray:
        im = Image.fromarray(im.astype(np.float32))
    height, width = size
    # pil resize uses opposite convention
    im = im.resize((width, height), resample=resample)
    return im


def save_img(im, savename):
    if type(im) is np.ndarray:
        im = Image.fromarray(im)
    im = im.convert('L')
    im.save(savename)


def norm_img(img, new_max=255.):
    assert img.max() - img.min() >= 0
    img = new_max * (img - img.min()) / (img.max() - img.min())
    return img


def gaussian_blur_img(img, blur_kernel_size=5):
    if type(img) == np.ndarray:
        img = Image.fromarray(img)
    img = img.convert("L")
    img = img.filter(ImageFilter.GaussianBlur(blur_kernel_size))
    return np.array(img)


def generate_heatmap(shape, mu, sigma=None, ratio=1.0):
    """ generates a heatmap with a gaussian centered on the coordinate passed in."""
    x, y = np.mgrid[0: shape[0]: 1, 0: shape[1]: 1]
    xy = np.column_stack([x.flat, y.flat])
    small_axis = 1 / ratio
    cov = np.array([[small_axis, 0], [0, 1]])
    if sigma is None:
        sigma = max(shape) / 5
    return multivariate_normal.pdf(xy, mean=mu, cov=sigma * cov).reshape(shape)


def shadowing(img, loc, brightness, sigma, row_col_ratio):
    hmap = generate_heatmap(img.shape, loc, sigma=sigma, ratio=row_col_ratio)
    hmap *= brightness / hmap.max()
    img = norm_img(255. * (np.ones(img.shape) - hmap) * np.array(img), 255.)
    return img


def brightening(img, loc, brightness, sigma, row_col_ratio):
    hmap = generate_heatmap(img.shape, loc, sigma=sigma, ratio=row_col_ratio)
    hmap *= brightness / hmap.max()
    img = norm_img(255. * (np.ones(img.shape) + hmap) * np.array(img), 255.)
    return img


def circle_mask(img, r, c, radius, intensity):
    r, c = masking.check_mask_bounds(img, r, c, radius)
    xx, yy = np.meshgrid(range(2 * int(np.ceil(radius))), range(2 * int(np.ceil(radius))))
    mask = intensity * masking.get_circle_mask(xx, yy, (radius, radius), radius)
    low_r, low_c = int(np.round(r - radius)), int(np.round(c - radius))
    img[low_r:low_r + mask.shape[0], low_c:low_c + mask.shape[1]] += mask.astype(np.uint8)


def show_img(img, title=None):
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


from mesh_utils import masking
