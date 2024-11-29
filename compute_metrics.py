import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ASOCA_handler.general import load_centerline, load_single_volume, align_centerline_to_image, get_slices_with_centerline
from ASOCA_handler.clustering import get_slice_centroids

from utils.augmentation import square_crop, mask_square


def get_left_out_stats(crop_size, data_path=Path('ASOCA')):
    percenteges = []
    count = 0
    remaining = []
    tot_crops = 0
    tot_masks = 0
    after = 0
    masks_pxls = 0

    for f in os.listdir(data_path):
        if os.path.isdir(data_path / f):
            # Normal or Diseased
            for i in range(len(os.listdir(data_path / f / 'CTCA'))):
                print(f'{data_path / f / "CTCA" / os.listdir(data_path / f / "CTCA")[i]}')
                # iterating over patients
                volume, masks = load_single_volume(data_path, f, i)
                g_name = volume.name.replace('ASOCA/', '')
                graph = load_centerline(data_path / f / 'Centerlines_graphs' / f'{g_name}_0.5mm.GML')
                graph = align_centerline_to_image(volume, graph, 'ijk')

                # bringing to batch first (BxHxW)
                masks = np.transpose(masks, (2, 0, 1))

                # only images with centerline in
                idxs = get_slices_with_centerline(graph)
                masks = masks[idxs]

                for j, n_slice in enumerate(idxs):
                    tot_masks += 1
                    m = masks[j].copy()
                    masks_pxls += np.sum(m == 1)
                    centroids = get_slice_centroids(n_slice, graph)
                    for c in centroids:
                        tot_crops += 1
                        masks[j] = mask_square(masks[j], crop_size, (c[0], c[1]))
                    if np.max(masks[j]) == 1:
                        count += 1

                        after += np.sum(masks[j] == 1)

                        p = (np.sum(masks[j] == 1) / np.sum(m == 1)) * 100
                        percenteges.append(p)
                        remaining.append((masks[j], n_slice, g_name))

    singles_ratio = f'{after / tot_masks}/{masks_pxls / tot_masks}'
    ratio = f'{after/masks_pxls}'

    print(f'masks out ratio: {count}/{tot_masks}, crops out ratio: {count}/{tot_crops}, avg percentage pxls: {ratio}')
    print(f'single masks pixels ratio:\n {singles_ratio}')

    dst = Path(f'data/masks_out_of_crops_{crop_size}')
    if not os.path.isdir(dst):
        for m, n_slice, vol in remaining:
            image = Image.fromarray(m.astype(np.uint8) * 255)
            image.save(dst / f'{vol}_{n_slice}.png')


def get_crops_snr(crop_size, data_path=Path('ASOCA'), save=False):

    SNR = []
    tot_crops = 0
    tot_ones = 0
    tot_zeros = 0

    for f in os.listdir(data_path):
        if os.path.isdir(data_path / f):
            # Normal or Diseased
            for i in range(len(os.listdir(data_path / f / 'CTCA'))):
                print(f'{data_path / f / "CTCA" / os.listdir(data_path / f / "CTCA")[i]}')
                # iterating over patients
                volume, masks = load_single_volume(data_path, f, i)
                g_name = volume.name.replace('ASOCA/', '')
                graph = load_centerline(data_path / f / 'Centerlines_graphs' / f'{g_name}_0.5mm.GML')
                graph = align_centerline_to_image(volume, graph, 'ijk')

                # bringing to batch first (BxHxW)
                vol = np.transpose(volume.data, (2, 0, 1))
                masks = np.transpose(masks, (2, 0, 1))

                # only images with centerline in
                idxs = get_slices_with_centerline(graph)
                vol = vol[idxs]
                masks = masks[idxs]

                for j, n_slice in enumerate(idxs):
                    centroids = get_slice_centroids(n_slice, graph)
                    for c in centroids:
                        tot_crops += 1

                        crop = square_crop(masks[j], crop_size, (c[0], c[1]))

                        tot_ones += np.sum(crop == 1)
                        tot_zeros += np.sum(crop == 0)

                        SNR.append(np.sum(crop == 1) / np.sum(crop == 0))

    print(f'crops: {crop_size}x{crop_size}:\nmean SNR: {np.mean(SNR)} = ({int(tot_ones/tot_crops)}/{int(tot_zeros/tot_crops)}), '
          f'std SNR: {np.std(SNR)}')

    if save:
        dst = Path('data')
        os.makedirs(dst, exist_ok=True)
        plt.figure(figsize=(10.8, 19.2))
        plt.boxplot(SNR)
        plt.title(f'SNR {crop_size}x{crop_size} crops')
        plt.savefig(dst / f'boxplot_{crop_size}', dpi=100)
        plt.close()

    return SNR




if __name__ == "__main__":
    data_path = Path('ASOCA')
    crop_size = 64

    # get_left_out_stats(crop_size, data_path)

    snr1 = get_crops_snr(128, data_path, True)
    snr2 = get_crops_snr(256, data_path, True)
    snr3 = get_crops_snr(64, data_path, True)

    dst = Path('data')
    os.makedirs(dst, exist_ok=True)
    plt.figure(figsize=(10.8, 19.2))
    plt.boxplot([snr3, snr1, snr2], tick_labels=['64x64', '128x128', '256x256'])
    plt.title(f'SNR crops')
    plt.savefig(dst / f'boxplot_all', dpi=100)
    plt.close()