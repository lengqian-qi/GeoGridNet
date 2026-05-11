import os
import h5py
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
from scipy import ndimage
from scipy.interpolate import griddata

def adaptive_spatial_grid_mapping_multiview(features, grid_size=(16,16),
                                            use_interpolation=True,
                                            normalize_by_fiber=True):
    """
    Multi-view ASGM
    Convert [N,15,3] -> [N,H,W,3]

    channel0 : xy projection store z
    channel1 : xz projection store y
    channel2 : yz projection store x
    """

    N, num_points, _ = features.shape
    H, W = grid_size

    fibermaps = np.zeros((N, H, W, 3))

    for i in range(N):

        fiber = features[i]

        # normalization
        if normalize_by_fiber:
            min_vals = fiber.min(axis=0)
            max_vals = fiber.max(axis=0)

            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1

            norm = (fiber - min_vals) / ranges
        else:
            norm = fiber.copy()

        x = norm[:,0]
        y = norm[:,1]
        z = norm[:,2]

        # ===== projection indices =====
        xy_x = (x*(H-1)).astype(int)
        xy_y = (y*(W-1)).astype(int)

        xz_x = (x*(H-1)).astype(int)
        xz_z = (z*(W-1)).astype(int)

        yz_y = (y*(H-1)).astype(int)
        yz_z = (z*(W-1)).astype(int)

        # initialize
        maps = np.zeros((3,H,W))
        counts = np.zeros((3,H,W))

        for p in range(num_points):

            # xy -> store z
            maps[0, xy_x[p], xy_y[p]] += fiber[p,2]
            counts[0, xy_x[p], xy_y[p]] += 1

            # xz -> store y
            maps[1, xz_x[p], xz_z[p]] += fiber[p,1]
            counts[1, xz_x[p], xz_z[p]] += 1

            # yz -> store x
            maps[2, yz_y[p], yz_z[p]] += fiber[p,0]
            counts[2, yz_y[p], yz_z[p]] += 1

        mask = counts > 0
        maps[mask] /= counts[mask]

        # interpolation
        if use_interpolation:

            for c in range(3):

                sparse = maps[c]
                mask = counts[c] > 0

                if np.sum(mask) < 3:
                    continue

                known_points = np.argwhere(mask)
                known_values = sparse[mask]

                grid_x, grid_y = np.mgrid[0:H,0:W]
                grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

                interp = griddata(
                    known_points,
                    known_values,
                    grid_points,
                    method='linear',
                    fill_value=np.nan
                )

                nan_mask = np.isnan(interp)

                if np.any(nan_mask):

                    nearest = griddata(
                        known_points,
                        known_values,
                        grid_points[nan_mask],
                        method='nearest'
                    )

                    interp[nan_mask] = nearest

                maps[c] = interp.reshape(H,W)

        fibermaps[i] = np.transpose(maps,(1,2,0))

    return fibermaps