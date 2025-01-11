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


def adjust_idxs(vol, idxs, crop_depth):
    """
    Adjust both start_id and last_id to make the depth divisible by the crop size,
    distributing the required padding between the two ends.

    Args:
        vol: full CT vol (np.array)
        idxs: indexes of slices with centerline
        crop_depth (int): crop depth

    Returns:
        tuple: Adjusted (start_id, last_id).
    """

    start_id, last_id = idxs[0], idxs[-1]

    max_depth = vol.shape[0]
    d = vol[start_id:last_id+1].shape[0]

    pad = crop_depth - (d % crop_depth)

    # adds padding only if pad != 0
    if pad != 0:
        extra_start = max(0, start_id)  # Max slices we can take from the start
        extra_end = max(0, max_depth - 1 - last_id)  # Max slices we can add at the end

        pad_start = min(pad // 2, extra_start)  # Take half the padding from the start
        pad_end = min(pad - pad_start, extra_end)  # Take the remaining from the end

        start_id -= pad_start
        last_id += pad_end

        # Ensure padding was fully distributed
        if (pad_start + pad_end) < pad:
            raise ValueError("Not enough space to adjust start_id and last_id for the required padding.")

    return start_id, last_id


def get_grid_patches_3d(crop_size: tuple, image, idxs):

    patch_size, crop_depth = crop_size

    # check if depth is feasible and adjust if possible
    start_id, last_id = adjust_idxs(image, idxs, crop_depth)
    crops = []

    d, h, w = image[start_id:last_id+1].shape
    assert not w % patch_size, 'patch size is not suited for image dims'

    sub_vol = image[start_id:last_id+1]
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            for z in range(0, d - crop_depth + 1, crop_depth):
                crop = sub_vol[z:z + crop_depth, y:y + patch_size, x:x + patch_size]
                crops.append(crop)

    return crops


def square_crop_3d(volume: np.ndarray, crop_size: int, start_id: int, crop_depth: int, center: tuple):
    """
    Extracts a square crop from the given image (numpy array). Pads the image if necessary.

    Args:
        - volume (np.ndarray): Input volume with shape (D, H, W).
        - crop_size (int): Desired size of the square crop.
        - start_id: index to start the depth cropping from
        - crop_depth: slices to include in cropped vol
        - center (tuple): (i, j) coordinates for the crop center.

    Returns:
        np.ndarray: Cropped square region of the input image.
    """

    assert len(volume.shape) < 4, 'Intended only for gray scale volumes'

    d, h, w = volume.shape
    dx = crop_size // 2

    y, x = center  # row, col are provided

    # Check bounds and calculate required padding
    top_padding = max(0, dx - y)
    bottom_padding = max(0, (y + dx) - h)
    left_padding = max(0, dx - x)
    right_padding = max(0, (x + dx) - w)

    # Pad the image if necessary
    if any([top_padding, bottom_padding, left_padding, right_padding]):
        padding = ((0, 0), (top_padding, bottom_padding), (left_padding, right_padding))

        volume = np.pad(volume, pad_width=padding, mode='constant', constant_values=0)
        x += left_padding
        y += top_padding

    crop = volume[start_id:start_id + crop_depth, y - dx:y + dx, x - dx:x + dx]

    return crop


def reconstruct_from_grid3d(crops, patch_size, crop_depth, original_shape):
    """
    Reconstructs the original image from the given crops.

    Args:
        - crops (list): List of 3D crops as np.array (D,H,W)
        - patch_size (int): Desired size of the square crop
        - crop_depth (int): Desired depth of the square crop
        - original_shape (tuple): Original shape of the image (d,h,w).

    returns:
        - reconstructed_vol (np.array): Reconstructed volume.
    """

    d, h, w = original_shape
    # Initialize the reconstructed volume and a counter array for normalization
    reconstructed_vol = np.zeros((d, h, w))

    # Iterate over the same coordinates as in the cropping process
    crop_idx = 0  # Index for the crops list

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            for z in range(0, d - crop_depth + 1, crop_depth):
                crop = crops[crop_idx]
                crop_idx += 1

                # Add the crop to the reconstructed volume
                reconstructed_vol[z:z + crop_depth, y:y + patch_size, x:x + patch_size] += crop

    return reconstructed_vol