import numpy as np

from medpy.metric import assd
from medpy.metric.binary import __surface_distances

from utils.general import to_numpy

__all__ = ["hausdorff_distance", "mod_hausdorff_distance", "average_surface_distance"]



def hausdorff_distance(data1, data2, voxel_spacing=None):
    data1, data2 = to_numpy(data1), to_numpy(data2)
    hd1 = __surface_distances(data1, data2, voxel_spacing, connectivity=1)
    hd2 = __surface_distances(data2, data1, voxel_spacing, connectivity=1)
    hd = max(hd1.max(), hd2.max())
    return hd


def mod_hausdorff_distance(data1, data2, voxel_spacing=None, percentile=95):
    data1, data2 = to_numpy(data1), to_numpy(data2)
    hd1 = __surface_distances(data1, data2, voxel_spacing, connectivity=1)
    hd2 = __surface_distances(data2, data1, voxel_spacing, connectivity=1)
    hd95_1 = np.percentile(hd1, percentile)
    hd95_2 = np.percentile(hd2, percentile)
    mhd = max(hd95_1, hd95_2)
    return mhd


def average_surface_distance(data1, data2, voxel_spacing=None):
    data1, data2 = to_numpy(data1), to_numpy(data2)
    return assd(data1, data2, voxel_spacing)
