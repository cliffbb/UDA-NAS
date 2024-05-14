# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Support synchronized crop and valid_pseudo_mask
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import numpy as np
from mmcv.parallel import DataContainer as DC

from . import OEMDataset
from .builder import DATASETS


def get_crop_bbox(img_size, crop_size):
    """Randomly get a crop bounding box."""
    assert len(img_size) == len(crop_size)
    assert len(img_size) == 2
    margin_h = max(img_size[0] - crop_size[0], 0)
    margin_w = max(img_size[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


@DATASETS.register_module()
class OEM_UDADataset(object):
    def __init__(self, source, target, cfg):
        self.source = source
        self.target = target
        self.ignore_index = target.ignore_index
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE
        assert target.ignore_index == source.ignore_index
        assert target.CLASSES == source.CLASSES
        assert target.PALETTE == source.PALETTE

        self.sync_crop_size = cfg.get('sync_crop_size')
        
    def synchronized_crop(self, s1, s2):
        if self.sync_crop_size is None:
            return s1, s2
        orig_crop_size = s1['img'].data.shape[1:]
        crop_y1, crop_y2, crop_x1, crop_x2 = get_crop_bbox(
            orig_crop_size, self.sync_crop_size)
        for i, s in enumerate([s1, s2]):
            for key in ['img', 'gt_semantic_seg', 'valid_pseudo_mask']:
                if key not in s:
                    continue
                s[key] = DC(
                    s[key].data[:, crop_y1:crop_y2, crop_x1:crop_x2],
                    stack=s[key]._stack)
        return s1, s2

    def __getitem__(self, idx):
        s1 = self.source[idx // len(self.target)]
        s2 = self.target[idx % len(self.target)]
        s1, s2 = self.synchronized_crop(s1, s2)
        out = {
            **s1, 'target_img_metas': s2['img_metas'],
            'target_img': s2['img']}
        
        if 'valid_pseudo_mask' in s2:
            out['valid_pseudo_mask'] = s2['valid_pseudo_mask']
        return out

    def __len__(self):
        return len(self.source) * len(self.target)
