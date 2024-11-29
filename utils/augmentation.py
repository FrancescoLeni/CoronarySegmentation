import numpy as np
import random


def square_crop(image: np.ndarray, crop_size: int, center: tuple = None):
    """
    Extracts a square crop from the given image (numpy array). Pads the image if necessary.

    Args:
        image (np.ndarray): Input image with shape (H, W) or (H, W, C).
        crop_size (int): Desired size of the square crop.
        center (tuple): (i, j) coordinates for the crop center. If None, random cropping is performed.

    Returns:
        np.ndarray: Cropped square region of the input image.
    """
    h, w = image.shape[:2]
    dx = crop_size // 2

    if center is None:
        # Randomly select the center of the crop
        x = random.randint(dx, w - dx)
        y = random.randint(dx, h - dx)
    else:
        y, x = center  # row, col are provided

    # Check bounds and calculate required padding
    top_padding = max(0, dx - y)
    bottom_padding = max(0, (y + dx) - h)
    left_padding = max(0, dx - x)
    right_padding = max(0, (x + dx) - w)

    # Pad the image if necessary
    if any([top_padding, bottom_padding, left_padding, right_padding]):
        padding = ((top_padding, bottom_padding), (left_padding, right_padding)) + ((0, 0),) * (image.ndim - 2)

        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
        x += left_padding
        y += top_padding


    crop = image[y - dx:y + dx, x - dx:x + dx]
    return crop


def mask_square(image: np.ndarray, mask_size: int, center: tuple = None):
    """

    Masks a square centered in given center with 0

    Args:
        image (np.ndarray): Input image with shape (H, W) or (H, W, C).
        mask_size (int): Desired size of the mask.
        center (tuple): (i, j) coordinates for the crop center. If None, random masking is performed.

    Returns:
        np.ndarray: masked input image.
    """
    h, w = image.shape[:2]
    dx = mask_size // 2

    if center is None:
        # Randomly select the center of the crop
        x = random.randint(dx, w - dx)
        y = random.randint(dx, h - dx)
    else:
        y, x = center  # row, col are provided

    y_start = max(0, y - dx)
    y_end = min(image.shape[0], y + dx)  # Clip to the height of the image
    x_start = max(0, x - dx)
    x_end = min(image.shape[1], x + dx)  # Clip to the width of the image

    # Modify the image
    if len(image.shape) == 3:  # RGB image
        image[y_start:y_end, x_start:x_end] = [0, 0, 0]
    else:  # Single-channel (grayscale) image
        image[y_start:y_end, x_start:x_end] = 0

    return image


def get_grid_patches(patch_size, image):
    h, w = image.shape
    crops = []

    assert not w % patch_size, 'patch size is not suited for image dims'
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):

            crop = image[y:y + patch_size, x:x + patch_size]
            crops.append(crop)

    return crops

