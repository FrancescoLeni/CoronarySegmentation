from utils.loaders import load_all
import argparse
import torch
from pathlib import Path
import torchmetrics
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import torch.nn.functional as F

from models.common import Dummy, UNet, UNet3D, UNetBig, ConvNeXtUnet
from models.Rep_ViT import RepViTUnet, RepViTUnet3D
from models import check_load_model

from utils import my_logger


def dice(x, y, smooth=1e-6, C=2):

    x = F.softmax(x, dim=1).view(x.shape[0], C, -1)  # [N, C, *]

    y = y.view(x.shape[0], -1)  # [N, *]

    y_onehot = F.one_hot(y, num_classes=C).permute(0, 2, 1)  # [N, C, *]

    # Compute intersection and union
    intersection = torch.sum(x * y_onehot, dim=2)  # [N, C]
    union = torch.sum(x.pow(2), dim=2) + torch.sum(y_onehot, dim=2)  # [N, C]

    # Compute Dice coefficient
    dice = (2 * intersection + smooth) / (union + smooth)  # [N, C]

    # setting to 0 small values
    dice[dice <= 1e-4] = 0

    return dice[:, 1:].squeeze(-1).numpy().tolist()  # all classes except bkg


def compute_all_metrics(data, device, metrics_dict, num_classes=2):

    x, y = data

    P = torchmetrics.classification.Precision(task="multiclass", num_classes=num_classes, top_k=1, average=None).to(device)
    R = torchmetrics.classification.Recall(task="multiclass", num_classes=num_classes, top_k=1, average=None).to(device)
    A = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average=None).to(device)

    # in-place
    metrics_dict['Precision'].append(P(x, y)[1].item())
    metrics_dict['Recall'].append(R(x, y)[1].item())
    metrics_dict['Accuracy'].append(A(x, y)[1].item())


def main(args):

    dst = Path(str(Path(args.weights).parent).replace('train', 'test').replace('weights', '') + f'{Path(args.weights).stem}' + f'_{args.reshape_mode}')
    os.makedirs(dst, exist_ok=True)
    print(f'save path is "{dst}')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        batch_size = 8
    else:
        device = torch.device('cpu')
        batch_size = 16

    print(f'used device: {device}')

    out_classes = args.n_classes

    if args.model == "Dummy":
        model = Dummy()
    elif args.model == 'Unet':
        model = UNet(out_classes)
    elif args.model == 'UnetBig':
        model = UNetBig(n_classes=out_classes)
        # loading model = bla bla bla
    elif args.model == 'RepViT':
        model = RepViTUnet('m2', img_size=args.crop_size, n_classes=out_classes, fuse=True)
    elif args.model == 'ConvNeXt':
        model = ConvNeXtUnet(n_classes=out_classes)
    elif args.model == 'RepViT3D':
        model = RepViTUnet3D(n_classes=out_classes)
    elif args.model == 'Unet3D':
        model = UNet3D(out_classes)
    else:
        raise TypeError("Model name not recognised")

    model = check_load_model(model, args.weights, my_logger)
    model.to(device)

    flag3D = any(isinstance(module, torch.nn.Conv3d) for module in model.modules())

    test_loader = load_all(args.data_path, reshape_mode=args.reshape_mode, crop_size=args.crop_size, scaler='standard',
                           batch_size=batch_size, test_flag=True, flag_3D=flag3D)

    out = []
    model.eval()
    metrics_dict = {'Dice': [], 'Precision': [], 'Recall': [], 'Accuracy': []}

    # dice = torchmetrics.classification.Dice(num_classes=out_classes, top_k=1, ignore_index=0).to('cpu')

    np_batches = []

    print('evaluating on loaded testset...')
    with torch.no_grad():
        for n, batch in enumerate(test_loader):
            print(f' batch {n+1}/{len(test_loader)}')
            x, y = batch

            p = model(x.to(device))

            pred = torch.argmax(p, dim=1)
            pred = (pred == 1).float().squeeze()
            prob = torch.nn.functional.softmax(p, dim=1)

            # i[0] to only get channel img, p[1] to only select class 1 probs
            np_batches += [(i[0].cpu().numpy().squeeze(), j.cpu().numpy().squeeze(),
                            pb[1].cpu().numpy().squeeze(), pr.cpu().numpy().squeeze())
                            for i, j, pb, pr in zip(x, y, prob, pred)]
            if flag3D:
                depth_size = 32
                ones_indxs = (y.view(y.shape[0]*depth_size, -1).sum(dim=1) > 0)
                FP_idxs = (prob[:, 1, :, :, :].squeeze().reshape(prob.shape[0]*depth_size, -1).max(dim=1)[0] > 0.5)
                indexs = ones_indxs.cpu() | FP_idxs.cpu()
                if indexs.any():
                    pp = p.permute(0, 2, 1, 3, 4).reshape(p.shape[0]*depth_size, p.shape[1], p.shape[3], p.shape[4])
                    yy = y.reshape(y.shape[0]*depth_size, y.shape[2], y.shape[3])
                    metrics_dict['Dice'].extend(dice(pp[indexs].to('cpu'), yy[indexs]))
            else:
                ones_indxs = (y.view(y.shape[0], -1).sum(dim=1) > 0)
                FP_idxs = (prob[:, 1, :, :].view(prob.shape[0], -1).max(dim=1)[0] > 0.5)
                indexs = ones_indxs.cpu() | FP_idxs.cpu()
                if indexs.any():
                    metrics_dict['Dice'].extend(dice(p[indexs].to('cpu'), y[indexs]))
            # dice.reset()

            out.append((p, y.to(device)))

    preds = torch.cat([x for x, _ in out], dim=0)
    masks = torch.cat([y for _, y in out], dim=0)

    print('computing metrics...')
    compute_all_metrics((preds.to('cpu'), masks.to('cpu')), 'cpu', metrics_dict)
    cm = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=out_classes).to('cpu')

    np.save(dst / 'dice.npy', np.array(metrics_dict['Dice']))
    print('dice array saved!')

    f, a = plt.subplots(1, 1, figsize=(19.2, 10.8))
    cm_val = cm(preds.to('cpu'), masks.to('cpu'))
    cm.plot(val=cm_val, ax=a)
    plt.savefig(dst / f'ConfusionMatrix.png', dpi=100)
    plt.close()
    print('Confusion Matrix saved!')

    stats_text = []
    q1 = np.percentile(metrics_dict['Dice'], 25)
    median = np.median(metrics_dict['Dice'])
    q3 = np.percentile(metrics_dict['Dice'], 75)
    iqr = q3 - q1
    mean = np.mean(metrics_dict['Dice'])
    stats_text.append(
        f"  Q1: {q1:.2f}\n"
        f"  Median: {median:.2f}\n"
        f"  Q3: {q3:.2f}\n"
        f"  IQR: {iqr:.2f}\n"
        f"  Mean: {mean:.2f}\n"
        f"  N° 0s: {np.sum(np.array(metrics_dict['Dice']) == 0)}"
    )

    # saving boxplot to dst
    f, a = plt.subplots(1, 1, figsize=(19.2, 10.8))
    a.boxplot(metrics_dict['Dice'], label='Dice')
    a.set_title(f'Test set Dice for {dst.name}')
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    text = "\n\n".join(stats_text)
    a.text(1.02, 0.9, text, transform=a.transAxes, fontsize=10, verticalalignment='center', bbox=props)

    plt.savefig(dst / 'test_dice_boxplot.png', dpi=100)
    plt.close()
    print('Dice boxplot saved!')

    # dice histogram
    f, a = plt.subplots(1, 1, figsize=(19.2, 10.8))

    a.hist(metrics_dict['Dice'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    a.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    a.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')

    a.set_title(f'Test set Dice for {dst.name}')
    a.legend()
    plt.savefig(dst / 'test_dice_hist.png', dpi=100)
    plt.close()
    print('Dice histogram saved!')

    metrics_dict['Dice'] = [np.mean(metrics_dict['Dice'])]

    # saving metrics test
    df = pd.DataFrame(metrics_dict)
    df.to_csv(dst / 'test_metrics.csv', index=False)
    print('CSV saved!')

    # plotting 4 random crops
    random.shuffle(np_batches)
    sampled = np_batches[:4]

    for n, s in enumerate(sampled):
        im, mask, prob, pred = s

        if flag3D:
            i = random.randint(0, len(im)-1)
            im, mask, prob, pred = im[i], mask[i], prob[i], pred[i]

        f, axs = plt.subplots(1, 3, figsize=(19.2, 10.8))
        spec = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.1], wspace=0.3)
        axs = axs.flatten()

        overlay_gt= np.zeros((mask.shape[0], mask.shape[1], 3))
        overlay_gt[:, :, 0] = mask

        axs[0].imshow(im, cmap='gray')
        axs[0].imshow(overlay_gt, cmap='Reds', alpha=0.5)
        axs[0].axis('off')
        axs[0].set_title('true mask')

        overlay_pred = np.zeros((pred.shape[0], pred.shape[1], 3))
        overlay_pred[:, :, 0] = pred

        axs[1].imshow(im, cmap='gray')
        axs[1].imshow(overlay_pred, cmap='Reds', alpha=0.5)
        axs[1].axis('off')
        axs[1].set_title('predicted mask')

        axs[2].imshow(prob, cmap='jet')
        axs[2].axis('off')
        axs[2].set_title('prediction probability')
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.5)
        cbar = f.colorbar(axs[2].images[0], cax=cax, orientation='vertical')

        plt.tight_layout()
        plt.savefig(dst / f'test_exemple_{n}.png', dpi=100)
        plt.close()
    print('Random crop saved!')

if __name__ == '__main__':
    # do things

    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('--model', type=str, required=True, help='name of model to train')
    parser.add_argument('--weights', type=str, required=True, help='path to weights')
    parser.add_argument('--data_path', type=str, default='ASOCA_DATASET', help='path to ASOCA dataset')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--reshape_mode', type=str, default='crop', choices=['crop', 'grid'], help=" how to handle resize")
    parser.add_argument('--crop_size', type=int, default=128, help='the finel shape input to model')


    args = parser.parse_args()

    main(args)