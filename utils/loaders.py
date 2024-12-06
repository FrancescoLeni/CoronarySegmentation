import os
from pathlib import Path
import numpy as np
# from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch

from . import my_logger

from .ASOCA_handler.general import get_slices_with_centerline, floor_or_ceil, load_vol_lab_graph_and_align

from .ASOCA_handler.clustering import get_slice_centroids, build_centerline_per_slice_dict, get_subvolumes_centroid

from .augmentation import square_crop, get_grid_patches, get_grid_patches_3d, adjust_idxs, square_crop_3d


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
        self.crop_depth = 32
        self.train, self.val, self.test = self.load_imgs()

    def load_imgs(self):

        imgs = defaultdict(list)

        for folder in os.listdir(self.data_path):
            # train, valid
            if not self.test_flag and folder == 'test':
                continue

            # only test
            if self.test_flag and folder != 'test':
                continue

            my_logger.info(f'loading images from {self.data_path / folder}...')
            for f in os.listdir(self.data_path / folder):
                # Normal or Diseased
                for i in os.listdir(self.data_path / folder / f / 'CTCA'):
                    # iterating over patients
                    if self.store_imgs:  # loads all dataset to ram
                        volume, masks, graph = load_vol_lab_graph_and_align(self.data_path / folder / f / 'CTCA' / i, 'ijk')

                        # bringing to batch first (BxHxW)
                        vol = np.transpose(volume, (2, 0, 1))
                        masks = np.transpose(masks, (2, 0, 1))

                        # only images with centerline in
                        idxs = get_slices_with_centerline(graph)

                        if not self.flag_3D:
                            vol = vol[idxs]
                            masks = masks[idxs]

                            imgs[folder].extend(self.preprocess_2d((vol, masks), graph, idxs))
                        else:
                            imgs[folder].extend(self.preprocess_3d((vol, masks), graph, idxs))

                    else:  # folders to only load batches
                        raise AttributeError('TO DO')
            x, _ = imgs[folder][0]
            my_logger.info(f'loaded {len(imgs[folder])} samples with shape {x.shape} for "{folder}"')
        if self.test_flag:
            return [], [], imgs['test']
        else:
            return imgs['train'], imgs['val'], []

    def preprocess_2d(self, data, graph, idxs, eps=5, closeness=50.):

        vol, masks = data

        # scaling data
        vol = self.scale_data(vol)

        # cropping around centerline centroids
        if self.reshape_mode == 'crop':
            out = []
            for i, n_slice in enumerate(idxs):
                centroids, ij = get_slice_centroids(n_slice, graph, eps, closeness)
                for c in centroids:
                    g_ch = np.zeros(vol[i].shape).astype(np.uint8)
                    g_ch[ij[:, 0], ij[:, 1]] = 1
                    g_ch = square_crop(g_ch, self.crop_size, (c[0], c[1]))
                    v = square_crop(vol[i].squeeze(), self.crop_size, (c[0], c[1]))
                    m = square_crop(masks[i].squeeze(), self.crop_size, (c[0], c[1]))
                    v_in = np.stack((v, g_ch), axis=0)
                    out.append((v_in, m))

        # cropping in grid-like patches
        elif self.reshape_mode == 'grid':
            out = []
            for i in range(len(idxs)):
                z_dict = build_centerline_per_slice_dict(graph)
                ij = np.array([[floor_or_ceil(graph.nodes[j]['x']), floor_or_ceil(graph.nodes[j]['y'])] for j in z_dict[idxs[i]]])
                g_ch = np.zeros(vol[i].shape).astype(np.uint8)
                g_ch[ij[:, 0], ij[:, 1]] = 1
                g_ch = get_grid_patches(self.crop_size, g_ch)
                v_patches = get_grid_patches(self.crop_size, vol[i].squeeze())
                m_patches = get_grid_patches(self.crop_size, masks[i].squeeze())
                # using patch ONLY if it contains at least 1 labeled point
                out += [(np.stack((v, g), axis=0), m) for v, m, g in zip(v_patches, m_patches, g_ch) if np.max(m.ravel()) == 1]
        else:
            raise AttributeError(f'"{self.reshape_mode}" reshape_mode not valid')
        return out

    def preprocess_3d(self, data, graph, idxs, eps=5, closeness=64.):

        vol, masks = data

        # scaling data
        vol = self.scale_data(vol)

        # extra channel for graph
        g_ch = np.zeros(vol.shape).astype(np.uint8)

        # cropping around centerline centroids
        if self.reshape_mode == 'crop':
            # adjusted start and last ids for volume cropping
            start_id, last_id = adjust_idxs(vol, idxs, self.crop_depth)
            clusters_list, kij_list = get_subvolumes_centroid(graph, start_id, last_id, self.crop_depth,
                                                              eps=eps, closeness=closeness)
            kij = np.array([x for sub in kij_list for x in sub])
            g_ch[kij[:, 0], kij[:, 1], kij[:, 2]] = 1

            out = []
            for i, start in enumerate(range(start_id, last_id, self.crop_depth)):
                for c in clusters_list[i]:

                    v_crop = square_crop_3d(vol, self.crop_size, start, self.crop_depth, (c[0], c[1]))
                    g_crop = square_crop_3d(g_ch, self.crop_size, start, self.crop_depth, (c[0], c[1]))
                    m_crop = square_crop_3d(masks, self.crop_size, start, self.crop_depth, (c[0], c[1]))

                    out.append((np.stack((v_crop, g_crop), axis=0), m_crop))

        # cropping in grid-like patches
        elif self.reshape_mode == 'grid':
            out = []
            for i in idxs:
                z_dict = build_centerline_per_slice_dict(graph)
                ij = np.array([[floor_or_ceil(graph.nodes[j]['x']), floor_or_ceil(graph.nodes[j]['y'])] for j in z_dict[i]])
                g_ch[i, ij[:, 0], ij[:, 1]] = 1

            g_ch = get_grid_patches_3d((self.crop_size, self.crop_depth), g_ch, idxs)
            v_patches = get_grid_patches_3d((self.crop_size, self.crop_depth), vol, idxs)
            m_patches = get_grid_patches_3d((self.crop_size, self.crop_depth), masks, idxs)

            # keeping only volumes with at least 1 pixel label
            out += [(np.stack((v, g), axis=0), m) for v, m, g in zip(v_patches, m_patches, g_ch) if np.max(m.ravel()) == 1]
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

        x = torch.from_numpy(x.astype('float32'))
        y = torch.LongTensor(y)

        return x, y


def load_all(data_path, reshape_mode=None, crop_size=128, scaler='standard', batch_size=4, test_flag=False, n_workers=0,
             pin_memory=True, flag_3D=False):
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
                            scaler=scaler, flag_3D=flag_3D)
    train += loader.train
    val += loader.val
    test += loader.test

    if test_flag:
        test_loader = torch.utils.data.DataLoader(LoaderFromData(test), batch_size=batch_size, shuffle=False,
                                                  pin_memory=pin_memory, num_workers=n_workers)
        return test_loader
    else:
        train_loader = torch.utils.data.DataLoader(LoaderFromData(train), batch_size=batch_size, shuffle=True,
                                                   pin_memory=pin_memory, num_workers=n_workers)
        val_loader = torch.utils.data.DataLoader(LoaderFromData(val), batch_size=batch_size, shuffle=True,
                                                 pin_memory=pin_memory, num_workers=n_workers)
        return train_loader, val_loader
