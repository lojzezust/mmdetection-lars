import os.path as osp

import mmcv
import cv2
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES


@PIPELINES.register_module()
class ObjectCenters(object):
    """Creates object centers masks for panoptic segmentation.

    Added key is "gt_centers".

    Args:
        sigma (float): Standard deviation of Gaussians on centers.
        normalize (bool): Whether to normalize the masks to range (0, 1).
    """

    def __init__(self, sigma=3., normalize=False):
        self.sigma = sigma
        self.normalize = normalize

    def __call__(self, results):
        """Call function create object centers.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        center_mask = np.zeros((results['img'].shape[0], results['img'].shape[1]), dtype=np.float32)
        for mask in results['gt_masks'].masks:
            x,y = np.where(mask)
            if np.isnan(x).any() or np.isnan(y).any():
                print('WARN: Nan in mask')

            if len(x) == 0:
                continue # TODO: this should not happen

            xm, ym = int(np.mean(x).round()), int(np.mean(y).round())

            center_mask[xm, ym] = 1. # Seed point

        # Create Gaussians on seed points
        center_mask = cv2.GaussianBlur(center_mask, (0,0), self.sigma)

        # Normalize
        if self.normalize and np.max(center_mask) > 0:
            center_mask = center_mask / np.max(center_mask)

        results['gt_centers'] = center_mask
        results['mask_fields'].append('gt_centers')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(sigma={self.sigma}, normalize={self.normalize})'
        return repr_str