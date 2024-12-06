import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.common import UNet

from utils.loaders import LoaderFromPath


# x = torch.rand((1,2,128,128), dtype=torch.float32)
# y = torch.rand((1,128,128)).to(torch.long)
#
# w = torch.tensor([0.0056, 0.9944], dtype=torch.float32)
# alpha = torch.where(y == 1, w[1], w[0])
# ce_loss = F.cross_entropy(x, y, reduction='none')  # shape BxHxW
# pt = torch.exp(-ce_loss)
# loss1 = alpha * ce_loss  # shape BxHxW
#
# ce_loss = F.cross_entropy(x, y, reduction='mean', weight=w)  # shape BxHxW
#
# print(loss1.mean(), ce_loss)
#
# # Class weights
# w = torch.tensor([0.0056, 0.9944], dtype=torch.float32)
#
# # Compute alpha weights
# alpha = torch.where(y == 1, w[1], w[0])  # Class-based weights
#
# # Unweighted cross-entropy loss
# ce_loss = F.cross_entropy(x, y, reduction='none')  # Per-pixel loss
#
# # Normalize the alpha weights
# alpha_normalized = alpha / alpha.mean()
#
# # Weighted loss
# loss1 = alpha_normalized * (1 - pt) ** 0.5 * ce_loss  # Weighted loss
# loss1_mean = loss1.mean()  # Average across pixels
#
# # Built-in PyTorch weighted cross-entropy
# loss2 = F.cross_entropy(x, y, reduction='mean', weight=w)
#
# print("loss1_mean (manual, normalized):", loss1_mean.item())
# print("loss2 (PyTorch):", loss2.item())

from utils.ASOCA_handler.general import load_vol_lab_graph_and_align, get_slices_with_centerline, min_max
from utils.ASOCA_handler.clustering import get_slice_centroids
from utils.augmentation import square_crop
from models.Rep_ViT import RepViTUnet
from models import check_load_model
from copy import deepcopy
from utils import my_logger

vol, masks, g = load_vol_lab_graph_and_align('ASOCA_DATASET/test/Diseased/CTCA/Diseased_16.nrrd', 'ijk')


idxs = get_slices_with_centerline(g)

vol = np.transpose(vol, (2, 0, 1))
masks = np.transpose(masks, (2, 0, 1))
vol = vol[idxs]
masks = masks[idxs]

out = []

for i, n_slice in enumerate(idxs):
    centroids, ij = get_slice_centroids(n_slice, g)
    for c in centroids:
        g_ch = np.zeros(vol[i].shape).astype(np.uint8)
        g_ch[ij[:, 0], ij[:, 1]] = 1
        g_ch = square_crop(g_ch, 128, (c[0], c[1]))
        v = square_crop(vol[i].squeeze(), 128, (c[0], c[1]))
        m = square_crop(masks[i].squeeze(), 128, (c[0], c[1]))
        v_in = np.stack((v, g_ch), axis=0)
        out.append((v, v_in, m, g_ch))

im, input_im, mask, gg = out[24]

def scal(x):
    mean, var = (-440.90224, 269087.00438)
    std = np.sqrt(var)
    return (x - mean) / std

mod = UNet(2)

mod = check_load_model(mod, 'runs/best_0.pt', my_logger)

input_im = torch.tensor(scal(input_im), dtype=torch.float32).unsqueeze(0)


mod.eval()
#
# for m in mod.modules():
#     if isinstance(m, torch.nn.BatchNorm2d):
#         m.running_mean.fill_(0)
#         m.running_var.fill_(1)

with torch.no_grad():
    p = mod(input_im)

p = p[0, :, :, :].to('cpu').unsqueeze(0)

class_1_mask = torch.argmax(p, dim=1)  # Shape BSx128x128, with values 0 or 1, where 1 is for class 1
class_1prob_t = torch.nn.functional.softmax(p, dim=1)
# class_1prob = class_1prob_t.squeeze().numpy()
# class_1prob = class_1prob[1, :, :].squeeze()
# If you want the mask for class 1 (mask where class 1 is predicted)
class_1_mask = (class_1_mask == 1).float().squeeze()  # Convert to a binary mask
class_1prob = class_1prob_t.squeeze()[1, :, :].squeeze()
# Similarly, you can extract the mask for class 0
# class_0_mask = (class_1_mask == 0).float().squeeze()

im = min_max(im)

i1 = deepcopy(im)
i1[mask == 1] = 1

i2 = deepcopy(im)
i2[class_1_mask == 1] = 1

f, a = plt.subplots(1, 3)
a = a.flatten()

# # Print stats for BatchNorm layers
# for name, module in mod.named_modules():
#     print(name)
#     if isinstance(module, nn.BatchNorm2d):
#         print(f"BatchNorm Layer: {name}")
#         print(f"  Running Mean: {module.running_mean}")
#         print(f"  Running Var: {module.running_var}")
#         print(f"  Weight (gamma): {module.weight.data}")
#         print(f"  Bias (beta): {module.bias.data}")


a[0].imshow(i1, cmap='gray')
a[0].set_title('true_mask')
a[1].imshow(i2, cmap='gray')
a[1].set_title('pred_mask')
a[2].imshow(class_1prob)
a[2].set_title('probability 1 ')
plt.show()






