import numpy as np
import random


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
        tuple: (x, y) coordinates of the crop center.
    """
    h, w = image.shape[:2]
    dx = crop_size // 2

    # Determine the crop center
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
        # Adjust center coordinates for the padded image
        x += left_padding
        y += top_padding

    # Extract the crop
    crop = image[y - dx:y + dx, x - dx:x + dx]
    return crop

