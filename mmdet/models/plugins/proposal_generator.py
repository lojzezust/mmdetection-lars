# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from mmcv.cnn import PLUGIN_LAYERS, Conv2d, ConvModule, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import BaseModule, ModuleList
from ..builder import build_loss

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


class PointSinePositionalEncoding(BaseModule):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, points, size):
        N, _ = points.shape
        if N == 0:
            return torch.zeros((0, self.num_feats * 2), device=points.device)

        H, W = size
        x_embed = points[:,0]
        y_embed = points[:,1]
        if self.normalize:
            y_embed = (y_embed + self.offset) / (H + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (W + self.eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=points.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:,None] / dim_t
        pos_y = y_embed[:,None] / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).view(N, -1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).view(N, -1)

        pos = torch.cat((pos_y, pos_x), dim=1)
        return pos


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
                 max_samples=50,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),
                 loss_center=dict(type='CrossEntropyLoss',
                                  use_sigmoid=True,
                                  loss_weight=1.0),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_inputs = len(in_channels)
        self.num_layers = num_layers
        self.level = level
        self.max_samples = max_samples
        self.nms_conv = NonMaximaSuppression2d(nms_kernel_size)
        self.pos_encoder = PointSinePositionalEncoding(feat_channels//2, normalize=True)

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
            act_cfg=None)
        self.layers.append(conv_out)

        self.loss_center = build_loss(loss_center)

    def init_weights(self):
        """Initialize weights."""
        for i in range(self.num_layers):
            caffe2_xavier_init(self.layers[i].conv, bias=0)

    def loss(self, center_preds, gt_centers):
        """Compute proposal generator (center prediction) loss."""

        center_preds = TF.resize(center_preds, gt_centers.shape[2:])

        return self.loss_center(center_preds, gt_centers)

    def forward_train(self, gt_bboxes_list, size):
        """Forward function during training."""

        pos_enc_list = []
        gt_i_list = []
        obj_masks = torch.zeros((len(gt_bboxes_list), self.max_samples, size[0], size[1]), device=gt_bboxes_list[0].device, dtype=torch.float32)
        for batch_i, gt_bboxes in enumerate(gt_bboxes_list):
            N = gt_bboxes.shape[0]

            gt_centers = (gt_bboxes[:,:2] + gt_bboxes[:,2:])/2.
            pos_enc = self.pos_encoder(gt_centers, size) # N, C

            gt_i = torch.arange(N, device=pos_enc.device, dtype=torch.long)

            # Randomly shuffle the order of the points (this should not matter)
            if N > 0:
                perm = torch.randperm(N, device=pos_enc.device)
                pos_enc = pos_enc[perm]
                gt_i = gt_i[perm]
                gt_bboxes = gt_bboxes[perm]

            # Create object masks
            for obj_i,bbox in enumerate(gt_bboxes):
                bbox = bbox.round().clamp(min=0).int()
                obj_masks[batch_i, obj_i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.

            if N < self.max_samples:
                pos_enc = F.pad(pos_enc, (0, 0, 0, self.max_samples - N))
                gt_i = F.pad(gt_i, (0, self.max_samples - N), value=-1)
                N = self.max_samples

            pos_enc_list.append(pos_enc[:self.max_samples])
            gt_i_list.append(gt_i[:self.max_samples])

        pos_enc = torch.stack(pos_enc_list, dim=1)
        gt_i = torch.stack(gt_i_list)

        return pos_enc.detach(), gt_i.detach(), obj_masks.detach()


    def forward(self, multilevel_feats, positional_encodings, gt_center_mask=None):
        """
        Args:
            multilevel_feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).
            positional_encodings (list[Tensor]): Positional encodings of each level.
                Each has shape of (batch_size, c, h, w).
            gt_center_mask (list[Tensor], optional): GT center mask (only during training).
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
        preds = feats_orig
        pos = positional_encodings[self.level]
        for i in range(self.num_layers):
            preds = self.layers[i](preds)


        if gt_center_mask is None:
            # Non-maximum suppression on predicted centers
            probs = torch.sigmoid(preds)
            center_mask = self.nms_conv(preds, mask_only=True)
        else:
            center_mask = TF.resize(gt_center_mask, preds.shape[-2:])
            center_mask = self.nms_conv(center_mask, mask_only=True)

        # Sample features
        feats_sel, valid_mask = _select_and_pad_feats(feats_orig, center_mask, self.max_samples)
        pos_sel, _ = _select_and_pad_feats(pos, center_mask, self.max_samples)

        return feats_sel, pos_sel, valid_mask, preds

