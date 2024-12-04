import os
import numpy as np
import random
import warnings
import torch
from pathlib import Path
import json

import logging

import torch.nn as nn


def init_logger():

    logger = logging.getLogger("my_new_logger")

    # Set the level
    logger.setLevel(logging.INFO)

    # Create a handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("")
    handler.setFormatter(formatter)

    # Add the handler to your logger
    logger.addHandler(handler)

    # Disable propagation to the root logger
    logger.propagate = False

    # if not logger.handlers:
    #     logger.addHandler(handler)

    return logger


# GLOBAL CALL
my_logger = init_logger()


# fixes random states to same seed and silenced warnings
def random_state(seed=36):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'

    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    np.random.seed(seed)

    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def integer_division(a: int, b: int, floor=False):
    """

    args:
        -a: first term
        -b: second term
        -floor: if true returns division "per difetto", False returns "per eccesso"
    :return:
    """

    if floor:
        return a // b
    else:
        return a // b + 1


def increment_path(folder="runs", name="exp", exist_ok=False, sep=''):
    """

    Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    :param
        folder: addressed folder
        name: name of the new directory inside folder
        exist_ok: if True keeps the existing dir, else actually increments it
        sep: how to connect the increment with name (see above)
    """

    path = Path(folder) / name  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if not os.path.isdir(path):
        os.mkdir(path)

    print(f"saving folder is {path}")
    return path


def json_from_parser(parser_args, save_path, name="arguments.json"):
    args_dict = vars(parser_args)

    save_path = save_path / name
    with open(save_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=2)


def conv2d_output_dim(input_dim, kernel_size, padding, stride):

    # Formula for output dimensions
    output = ((input_dim + 2 * padding - kernel_size) // stride) + 1

    return output


def compute_padding(input_size, target_size, kernel_size, stride):
    # Ensure the kernel size and stride are integers
    k = kernel_size
    s = stride

    # Calculate the necessary padding for both height and width
    padding = ((target_size - 1) * s - input_size + k) / 2

    # Check if padding is an integer and non-negative
    if padding.is_integer() and padding >= 0:
        return int(padding)
    else:
        return "No valid padding found for the given input and target size with specified kernel and stride."


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.var = 0.0

    def __call__(self, *args, **kwargs):
        self.update(args[0])

    def update(self, vol):

        vol = vol.reshape(-1).tolist()
        for x in vol:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.var += delta * delta2

    def get_stats(self):
        if self.n < 2:
            return self.mean, float('nan')  # Variance is undefined for n < 2
        return self.mean, self.var / (self.n - 1)  # Return mean and unbiased variance


def calculate_receptive_field(model, input_size=(1, 1, 512, 512)):
    """
    Calculates the receptive field of a CNN model.

    Parameters:
    - model: PyTorch model
    - input_size: Tuple representing the input dimensions (N, C, H, W)

    Returns:
    - Receptive field size (in pixels).
    """
    # Receptive field tracker
    receptive_field = 1
    current_stride = 1
    current_size = input_size[-1]

    def register_hook(module, input, output):
        nonlocal receptive_field, current_stride, current_size

        # Only process convolutional, pooling, or transpose layers
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.ConvTranspose2d)):
            kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size,) * 2
            stride = module.stride if isinstance(module.stride, tuple) else (module.stride,) * 2
            padding = module.padding if isinstance(module.padding, tuple) else (module.padding,) * 2
            dilation = module.dilation if isinstance(module.dilation, tuple) else (module.dilation,) * 2

            # Update receptive field
            receptive_field += (kernel_size[0] - 1) * current_stride
            current_stride *= stride[0]

            # Update current spatial size
            current_size = (current_size - kernel_size[0] + 2 * padding[0]) // stride[0] + 1

    # Register hooks for all layers
    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(register_hook))

    # Forward pass to trigger hooks
    dummy_input = torch.zeros(input_size)
    model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return receptive_field

