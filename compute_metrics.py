import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.ASOCA_handler.general import load_centerline, load_single_volume, align_centerline_to_image, get_slices_with_centerline
from utils.ASOCA_handler.clustering import get_slice_centroids

from utils.augmentation import square_crop, mask_square


def get_left_out_stats(crop_size, data_path=Path('ASOCA'), dst='data/graph', save_crops=False):
    percenteges = []
    count = 0
    remaining = []
    tot_crops = 0
    tot_masks = 0
    after = 0
    masks_pxls = 0

    dst = Path(dst)
    graph_type = dst.name.replace('graph', '')
    os.makedirs(dst, exist_ok=True)
    txt_path = dst / 'analysis.txt'

    for f in os.listdir(data_path):
        if os.path.isdir(data_path / f):
            # 'test', 'train', or 'val'
            for category in ['Normal', 'Diseased']:  #
                ctca_path = data_path / f / category / 'CTCA'
                for i in os.listdir(ctca_path):  
                    print(f"Processing file {i} in directory {ctca_path}")
                    volume, masks = load_single_volume(ctca_path / i)

                    g_name = volume.name.replace('ASOCA/', '')
                    graph = load_centerline(data_path / f /category / 'Centerlines_graphs' / f'{g_name}{graph_type}.GML')
                    graph = graph.resample(0.2,True)
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
                            masks[j] = mask_square(masks[j], crop_size, c[0])
                        if np.max(masks[j]) == 1:
                            count += 1

                            after += np.sum(masks[j] == 1)

                            p = (np.sum(masks[j] == 1) / np.sum(m == 1)) * 100
                            percenteges.append(p)
                            remaining.append((masks[j], n_slice, g_name))

    # singles_ratio = f'{after / tot_masks}/{masks_pxls / tot_masks}'
    # ratio = f'{after/masks_pxls}'

    # print(f'masks out ratio: {count}/{tot_masks}, crops out ratio: {count}/{tot_crops}, avg percentage pxls: {ratio}')
    # print(f'single masks pixels ratio:\n {singles_ratio}')


    with open(txt_path, 'a') as file:
        # Write new lines to the file
        file.write(f'\nconsidering {crop_size}x{crop_size} crops\n')
        file.write(f'masks out ratio: {count/tot_masks: .3f} (a/b = {count}/{tot_masks})\n')
        file.write(f'crops out ratio:: {count / tot_crops: .3f} (a/c = {count}/{tot_crops})\n')
        file.write(f'pixels out ratio: {after / masks_pxls: .3f} (d/e = {after}/{masks_pxls})\n')

    if save_crops:
        dst = dst / f'masks_out_of_crops_{crop_size}'
        if not os.path.isdir(dst):
            for m, n_slice, vol in remaining:
                image = Image.fromarray(m.astype(np.uint8) * 255)
                image.save(dst / f'{vol}_{n_slice}.png')



def get_crops_snr(crop_size, data_path=Path('ASOCA'), save=False, dst='data/graph'):

    SNR = []
    tot_crops = 0
    tot_ones = 0
    tot_zeros = 0

    dst = Path(dst)
    graph_type = dst.name.replace('graph', '')

        # test, train, val
    for dataset in ['test', 'train', 'val']:
        # Normal, Diseased
        for category in ['Normal', 'Diseased']:
            ctca_path = data_path / dataset / category / 'CTCA'
            graphs_path = data_path / dataset / category / 'Centerlines_graphs'

        
            if not ctca_path.exists() or not graphs_path.exists():
                print(f"file not found: {ctca_path} o {graphs_path}")
                continue

            
            for i in os.listdir(ctca_path):
                file_ctca = ctca_path / i

                if not file_ctca.is_file():
                    print(f"File not found  or not valido: {file_ctca}")
                    continue

                print(f"Processing file {i} in directory {ctca_path}")

                
                volume, masks = load_single_volume(file_ctca)

                
                g_name = volume.name.replace('ASOCA/', '')
                graph_file = graphs_path / f'{g_name}{graph_type}.GML'

                if not graph_file.exists():
                    print(f"graph file not found: {graph_file}")
                    continue

                
                graph = load_centerline(graph_file)
                graph = graph.resample(0.2,True)
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

                        crop = square_crop(masks[j], crop_size, (c[0]))

                        tot_ones += np.sum(crop == 1)
                        tot_zeros += np.sum(crop == 0)

                        SNR.append(np.sum(crop == 1) / np.sum(crop == 0))

    # print(f'crops: {crop_size}x{crop_size}:\nmean SNR: {np.mean(SNR)} = ({int(tot_ones/tot_crops)}/{int(tot_zeros/tot_crops)}), '
    #       f'std SNR: {np.std(SNR)}')
    txt_path = Path(dst) / 'analysis.txt'
    with open(txt_path, 'a') as file:
        # Write new lines to the file
        file.write(f'\nmean SNR: {np.mean(SNR): .3f} (f/g = {int(tot_ones/tot_crops)}/{int(tot_zeros/tot_crops)})\n')
        file.write(f'std SNR: {np.std(SNR): .3f}\n')

    if save:
        plt.figure(figsize=(19.2, 10.8))
        plt.boxplot(SNR)
        plt.title(f'SNR {crop_size}x{crop_size} crops')
        plt.savefig(dst / f'boxplot_{crop_size}', dpi=100)
        plt.close()

    return SNR


def get_stats_box(crop_sizes, graph_type, data_path=Path('ASOCA'), dst='data'):

    assert graph_type in ['']

    dst = Path(dst) / f'graph{graph_type}'
    os.makedirs(dst, exist_ok=True)

    with open(dst / 'analysis.txt', 'a') as file:
        # Write new lines to the file
        file.write(f'\nUSING: graph{graph_type}\n')

    snr = []
    for c in crop_sizes:
        print(f'computing stats for {c}x{c}')
        get_left_out_stats(c, data_path, dst)
        print(f'computing SNR for {c}x{c}')
        snr.append(get_crops_snr(c, data_path, True, dst))

    lab = [f'{c}x{c}' for c in crop_sizes]

    plt.figure(figsize=(19.2, 10.8))
    plt.boxplot(snr, tick_labels=lab)
    plt.title(f'SNR crops')
    plt.savefig(dst / f'boxplot_all', dpi=100)
    plt.close()




if __name__ == "__main__":
    data_path = Path('ASOCA')
    crop_size = [64, 128, 256]

    graph_type = ['']

    for g in graph_type:
        print(f'computing stats for graph{g}')
        get_stats_box(crop_size, g)

