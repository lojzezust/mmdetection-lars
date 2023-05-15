# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import PLUGIN_LAYERS, Conv2d, ConvModule, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import BaseModule, ModuleList


def _get_nms_kernel2d(kx: int, ky: int) -> torch.Tensor:
    """Get conv kernel for NMS."""
    numel = ky * kx
    center = numel // 2
    weight = torch.eye(numel)
    weight[center, center] = 0
    return weight.view(numel, 1, ky, kx)

def _select_and_pad_feats(feats, masks, num_samples=512):
    """Sample features using masks and pad/crop to fixed number of samples."""
    feats_sel = []
    feats_sel_valid = []
    for feat_i, mask_i in zip(feats, masks):
        c,h,w = feat_i.shape
        # Flatten spatial dimensions
        mask_i = mask_i.view(h * w)
        feat_i = feat_i.permute(1,2,0).view(h * w, c)

        # Select masked features
        feat_sel = feat_i[mask_i]
        feat_valid_mask = torch.ones(feat_sel.shape[0], dtype=torch.bool, device=feat_sel.device)

        # Pad to fixed number of features
        if feat_sel.shape[0] < num_samples:
            feat_sel = F.pad(feat_sel, (0, 0, 0, num_samples - feat_sel.shape[0]))
            feat_valid_mask = F.pad(feat_valid_mask, (0, num_samples - feat_valid_mask.shape[0]))
        elif feat_sel.shape[0] > num_samples:
            feat_sel = feat_sel[:num_samples] # TODO: sample instead of cutting
            feat_valid_mask = feat_valid_mask[:num_samples]

        feats_sel.append(feat_sel)
        feats_sel_valid.append(feat_valid_mask)

    feats_sel = torch.stack(feats_sel, dim=0)
    feats_sel_valid = torch.stack(feats_sel_valid, dim=0)

    return feats_sel, feats_sel_valid



class NonMaximaSuppression2d(nn.Module):
    r"""Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size: Tuple[int, int]):
        super(NonMaximaSuppression2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.padding: Tuple[int, int,
                            int, int] = self._compute_zero_padding2d(kernel_size)
        self.kernel = _get_nms_kernel2d(*kernel_size)

    @staticmethod
    def _compute_zero_padding2d(
            kernel_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 2, kernel_size

        def pad(x):
            return (x - 1) // 2  # zero padding function

        ky, kx = kernel_size     # we assume a cubic kernel
        return (pad(ky), pad(ky), pad(kx), pad(kx))

    def forward(self, x: torch.Tensor, mask_only: bool = False) -> torch.Tensor:  # type: ignore
        assert len(x.shape) == 4, x.shape
        B, CH, H, W = x.size()
        # find local maximum values
        max_non_center = F.conv2d(F.pad(x, list(self.padding)[::-1], mode='replicate'),
                                  self.kernel.repeat(CH, 1, 1, 1).to(x.device, x.dtype),
                                  stride=1,
                                  groups=CH).view(B, CH, -1, H, W).max(dim=2)[0]
        mask = x > max_non_center
        if mask_only:
            return mask
        return x * (mask.to(x.dtype))


@PLUGIN_LAYERS.register_module()
class ProposalGenerator(BaseModule):
    """Proposal generator head.

    Args:
        in_channels (List[int]): Number of input channels for inputs.
        feat_channels (int): Number of feature channels.
        num_layers (int): Number of layers.
        level (int): Index of level to generate mask.
        nms_kernel_size (tuple[int]): Kernel size of NMS.
        max_samples (int): Max number of selected features.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_layers,
                 level=1,
                 nms_kernel_size=(3, 3),
                 max_samples=512,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_inputs = len(in_channels)
        self.num_layers = num_layers
        self.level = level
        self.max_samples = max_samples
        self.nms_conv = NonMaximaSuppression2d(nms_kernel_size)

        self.layers = ModuleList()
        self.use_bias = norm_cfg is None

        for i in range(num_layers-1):
            in_ch = in_channels[i] if i == 0 else feat_channels
            conv = ConvModule(
                in_ch,
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.layers.append(conv)

        conv_out = ConvModule(
            feat_channels,
            1,
            kernel_size=3,
            padding=1,
            bias=self.use_bias,
            norm_cfg=None,
            act_cfg=dict(type='Sigmoid'))
        self.layers.append(conv_out)

    def init_weights(self):
        """Initialize weights."""
        for i in range(self.num_layers):
            caffe2_xavier_init(self.layers[i].conv, bias=0)


    def forward(self, multilevel_feats, positional_encodings):
        """
        Args:
            multilevel_feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).
            positional_encodings (list[Tensor]): Positional encodings of each level.
        Returns:
            tuple: a tuple containing the following:
                feats_sel (Tensor): Selected features with shape of
                    (batch_size, num_samples, c).
                pos_sel (Tensor): Selected positional encodings with shape of
                    (batch_size, num_samples, c).
                valid_mask (Tensor): Valid mask of selected features with shape of
                    (batch_size, num_samples).
        """
        feats_orig = multilevel_feats[self.level]
        feats = feats_orig
        pos = positional_encodings[self.level]
        for i in range(self.num_layers):
            feats = self.layers[i](feats)

        # Non-maximum suppression on feats
        center_mask = self.nms_conv(feats, mask_only=True)

        # Sample features
        feats_sel, valid_mask = _select_and_pad_feats(feats_orig, center_mask, self.max_samples)
        pos_sel, _ = _select_and_pad_feats(pos, center_mask, self.max_samples)

        return feats_sel, pos_sel, valid_mask

