import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import normal_init
from mmcv.runner import force_fp32
from mmseg.ops import resize

from .decode_head import BaseDecodeHead
from .psp_head import PPM
from ..builder import HEADS, build_loss
from ..losses import accuracy


@HEADS.register_module()
class MUPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(MUPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.loss_dep = build_loss(dict(type='DepthLoss', dataset='NYU', height=464, max_depth=10.0, loss_weight=1.0))

        # PSP Module
        self.psp_modules_se = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck_se = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.psp_modules_de = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck_de = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs_se = nn.ModuleList()
        self.fpn_convs_se = nn.ModuleList()
        self.lateral_convs_de = nn.ModuleList()
        self.fpn_convs_de = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv_se = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv_se = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs_se.append(l_conv_se)
            self.fpn_convs_se.append(fpn_conv_se)
            l_conv_de = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv_de = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs_de.append(l_conv_de)
            self.fpn_convs_de.append(fpn_conv_de)

        self.fpn_bottleneck_se = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.fpn_bottleneck_de = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv_de = nn.Conv2d(self.channels, 1, kernel_size=1)

    def init_weights(self):
        """Initialize weights of classification layer."""
        super(MUPerHead, self).init_weights()
        normal_init(self.conv_de, mean=0, std=0.01)

    def psp_se_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules_se(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck_se(psp_outs)

        return output

    def psp_de_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules_de(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck_de(psp_outs)

        return output

    def cls_de(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_de(feat)
        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals_se = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs_se)
        ]

        laterals_se.append(self.psp_se_forward(inputs))

        laterals_de = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs_de)
        ]

        laterals_de.append(self.psp_de_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals_se)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals_se[i - 1].shape[2:]
            laterals_se[i - 1] += resize(
                laterals_se[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            laterals_de[i - 1] += resize(
                laterals_de[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs_se = [
            self.fpn_convs_se[i](laterals_se[i])
            for i in range(used_backbone_levels - 1)
        ]
        fpn_outs_de = [
            self.fpn_convs_de[i](laterals_de[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs_se.append(laterals_se[-1])
        fpn_outs_de.append(laterals_de[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs_se[i] = resize(
                fpn_outs_se[i],
                size=fpn_outs_se[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            fpn_outs_de[i] = resize(
                fpn_outs_de[i],
                size=fpn_outs_de[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs_se = torch.cat(fpn_outs_se, dim=1)
        output_se = self.fpn_bottleneck_se(fpn_outs_se)
        output_se = self.cls_seg(output_se)
        fpn_outs_de = torch.cat(fpn_outs_de, dim=1)
        output_de = self.fpn_bottleneck_de(fpn_outs_de)
        output_de = self.cls_de(output_de)
        return output_se, output_de

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, output, gt):
        """Compute segmentation loss."""
        seg_logit, dep_out = output[0], output[1]
        seg_label, dep_map = gt[0], gt[1]

        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        dep_out = resize(
            input=dep_out,
            size=dep_map.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
            dep_weight = self.sampler.sample(dep_out, dep_map)
        else:
            seg_weight = None
            dep_weight = None
        loss['loss_dep'] = self.loss_dep(
            dep_out,
            dep_map,
            weight=dep_weight,
            ignore_index=self.ignore_index)
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
