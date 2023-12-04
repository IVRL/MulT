import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from mmseg.ops import resize

from .decode_head import BaseDecodeHead
from ..builder import HEADS, build_loss
from ..losses import accuracy


class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
                                       1], nn.ReLU(),
                                   nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
                                       1], nn.ReLU(),
                                   nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
                                       1], nn.ReLU(),
                                   nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
                                       1], nn.ReLU(),
                                   nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        # head2 = self.head2(mla_p2)
        head2 = F.interpolate(self.head2(
            mla_p2), 4 * mla_p2.shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head3(
            mla_p3), 4 * mla_p3.shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), 4 * mla_p4.shape[-1], mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), 4 * mla_p5.shape[-1], mode='bilinear', align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)


@HEADS.register_module()
class METR_Head(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128,
                 norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(METR_Head, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels

        self.loss_dep = build_loss(dict(type='DepthLoss', dataset='NYU', height=464, max_depth=10.0, loss_weight=1.0))

        self.mlahead_se = MLAHead(mla_channels=self.mla_channels,
                                  mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls_se = nn.Conv2d(4 * self.mlahead_channels,
                                self.num_classes, 3, padding=1)
        self.mlahead_de = MLAHead(mla_channels=self.mla_channels,
                                  mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls_de = nn.Conv2d(4 * self.mlahead_channels, 1, 3, padding=1)

    def forward(self, inputs):
        se, de = inputs[0], inputs[1]
        se = self.mlahead_se(se[0], se[1], se[2], se[3])
        se = self.cls_se(se)
        se = F.interpolate(se, size=self.img_size, mode='bilinear',
                           align_corners=self.align_corners)
        de = self.mlahead_de(de[0], de[1], de[2], de[3])
        de = self.cls_de(de)
        de = F.interpolate(de, size=self.img_size, mode='bilinear',
                           align_corners=self.align_corners)
        return se, de

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
