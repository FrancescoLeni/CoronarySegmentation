import os
from pathlib import Path
import numpy as np
# from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch

from . import my_logger

from .ASOCA_handler.general import load_centerline, load_single_volume, align_centerline_to_image, \
                                    get_slices_with_centerline, floor_or_ceil

from .ASOCA_handler.clustering import get_slice_centroids, build_centerline_per_slice_dict

from .augmentation import square_crop, get_grid_patches


class LoaderFromPath:
    def __init__(self, data_path, reshape_mode=None, crop_size=128, scaler='standard', test_flag=False, store_imgs=True,
                 flag_3D=False):
        """
        gets the data path and loads images or path-to-images split datasets

        args:
            - data_path: path to dataset
            - reshape_mode: how to get square imgs
            - reshaped_size: target size as input to model
            - scaler: the type of scaler for normalizing data
            - test_flag: whether to return the test set (it always split for it, simply it is not returned)
            - use_label: whether to load also labels
            - load_imgs: if True it directly loads images to RAM, else it stores paths-to-images
        """
        accepted_reshape_types = [None, 'crop', 'grid']
        assert reshape_mode in accepted_reshape_types, f'"""{reshape_mode}" not valid, chose from {accepted_reshape_types}'
        self.reshape_mode = reshape_mode
        self.crop_size = crop_size

        accepted_scalers = ['standard', 'min_max']
        assert scaler in accepted_scalers, f'"{scaler}" scaler not valid, chose from {accepted_scalers}'
        self.scaler = scaler
        self.mean, self.var = (-440.90224, 269087.00438)
        self.vmax, self.vmin = (3526, -1024)

        self.data_path = Path(data_path)

        self.store_imgs = store_imgs

        self.test_flag = test_flag
        self.flag_3D = flag_3D
        self.train, self.val, self.test = self.load_imgs()

    def load_imgs(self):

        imgs = defaultdict(list)

        for folder in os.listdir(self.data_path):
            # train, valid or test
            if not self.test_flag and folder == 'test':
                continue

            my_logger.info(f'loading images from {self.data_path / folder}...')
            for f in os.listdir(self.data_path / folder):
                # Normal or Diseased
                for i in os.listdir(self.data_path / folder / f / 'CTCA'):
                    # iterating over patients
                    if self.store_imgs:  # loads all dataset to ram
                        volume, masks = load_single_volume(self.data_path / folder / f / 'CTCA' / i)

                        g_name = volume.name.replace('ASOCA/', '')
                        graph = load_centerline(self.data_path / folder / f / 'Centerlines_graphs' / f'{g_name}.GML')
                        graph = align_centerline_to_image(volume, graph, 'ijk')

                        # bringing to batch first (BxHxW)
                        vol = np.transpose(volume.data, (2, 0, 1))
                        masks = np.transpose(masks, (2, 0, 1))

                        # only images with centerline in
                        idxs = get_slices_with_centerline(graph)
                        vol = vol[idxs]
                        masks = masks[idxs]

                        imgs[folder].extend(self.preprocess((vol, masks), graph, idxs))

                    else:  # folders to only load batches
                        raise AttributeError('TO DO')

            my_logger.info(f'loaded {len(imgs[folder])} images for {folder}')
        if self.test_flag:
            return imgs['train'], imgs['val'], imgs['test']
        else:
            return imgs['train'], imgs['val'], []

    def preprocess(self, data, graph, idxs, eps=5, closeness=50.):

        vol, masks = data

        # scaling data
        vol = self.scale_data(vol)

        # cropping around centerline centroids
        if self.reshape_mode == 'crop':
            out = []
            if not self.flag_3D:
                for i, n_slice in enumerate(idxs):
                    centroids, xy = get_slice_centroids(n_slice, graph, eps, closeness)
                    for c in centroids:
                        g_ch = np.zeros(vol[i].shape).astype(np.uint8)
                        g_ch[xy[:, 1], xy[:, 0]] = 1
                        g_ch = square_crop(g_ch, self.crop_size, (c[0], c[1]))
                        v = square_crop(vol[i].squeeze(), self.crop_size, (c[0], c[1]))
                        m = square_crop(masks[i].squeeze(), self.crop_size, (c[0], c[1]))
                        v_in = np.stack((v, g_ch), axis=0)
                        out.append((v_in, m))
            else:
                # TO DO
                pass

        # cropping in grid-like patches
        elif self.reshape_mode == 'grid':
            out = []
            if not self.flag_3D:
                for i in range(len(idxs)):
                    z_dict = build_centerline_per_slice_dict(graph)
                    xy = np.array([[floor_or_ceil(graph.nodes[i]['x']), floor_or_ceil(graph.nodes[i]['y'])] for i in z_dict[n_slice]])
                    g_ch = np.zeros(vol[i].shape).astype(np.uint8)
                    g_ch[xy[:, 1], xy[:, 0]] = 1
                    g_ch = get_grid_patches(self.crop_size, g_ch)
                    v_patches = get_grid_patches(self.crop_size, vol[i].squeeze())
                    m_patches = get_grid_patches(self.crop_size, masks[i].squeeze())
                    # using patch ONLY if it contains at least 1 labeled point
                    out += [(np.stack((v, g_ch), axis=0), m) for v, m, g in zip(v_patches, m_patches, g_ch) if np.max(m.ravel()) == 1]
            else:
                # to do
                pass

        else:
            raise AttributeError(f'"{self.reshape_mode}" reshape_mode not valid')

        return out

    def scale_data(self, x):
        if self.scaler == 'standard':
            std = np.sqrt(self.var)
            return (x - self.mean) / std
        elif self.scaler == 'min_max':
            return (x - self.vmin) / (self.vmax - self.vmin)
        else:
            raise ValueError(f'provided scaler not valid...')


class LoaderFromData(torch.utils.data.Dataset):
    def __init__(self, data, augmentation=None):
        """
            gets the data loaded by the LoadersFromPaths and creates an iterable torch-like dataloader

            args:
                - data: list of images as np.array or tuple (img, lab)
                - augmentation: probably I'll create an apposite object
        """
        super().__init__()

        self.data = data

        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x, y = self.data[idx]  # imgs, labs, polys

        return self.transform(x, y)

    def transform(self, x, y=None):
        if self.augmentation:
            pass
            # do something for augmentation...

        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x.astype('float32'))
        y = torch.LongTensor(y)

        return x, y


def load_all(data_path, reshape_mode=None, crop_size=128, scaler='standard', batch_size=4, test_flag=False, n_workers=7,
             pin_memory=True):
    """
        loads all data in img_paths and returns the dataloaders for train, val and eventually test

        args:
            - data_path: path to dataset
            - reshape_mode: how to get crops
            - crop_size: crops_shape
            - scaler: which scaler to use
            - batch_size: self explained
            - test_flag: whether to return test set
            - n_workers: number of workers for parallel dataloading (rule of thumb: n° cpu core - 1)
            - pin_memory: whether to pin memory for more efficient passage to gpu
        returns:
            - train_loader, val_loader, (optional) test_loader : torch iterable dataloaders
    """

    accepted_reshape_types = [None, 'crop', 'grid']
    assert reshape_mode in accepted_reshape_types, f'{reshape_mode} not valid, chose from {accepted_reshape_types}'

    train = []
    val = []
    test = []

    # loading from all paths and splitting
    loader = LoaderFromPath(data_path, reshape_mode=reshape_mode, crop_size=crop_size, test_flag=test_flag,
                            scaler=scaler)
    train += loader.train
    val += loader.val
    test += loader.test


    train_loader = torch.utils.data.DataLoader(LoaderFromData(train), batch_size=batch_size, shuffle=True,
                                               pin_memory=pin_memory, num_workers=n_workers)
    val_loader = torch.utils.data.DataLoader(LoaderFromData(val), batch_size=batch_size, shuffle=True,
                                             pin_memory=pin_memory, num_workers=n_workers)
    if test_flag:
        test_loader = torch.utils.data.DataLoader(LoaderFromData(test), batch_size=batch_size, shuffle=False,
                                                  pin_memory=pin_memory, num_workers=n_workers)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader
